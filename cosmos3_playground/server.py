# SPDX-License-Identifier: Apache-2.0
"""FastAPI backend for the unified Cosmos3 Playground.

Routes a playground mode to the right surface:
* generate -> vLLM-Omni Cosmos3 server (/v1/images, /v1/videos)
* reason   -> OpenAI-compatible vLLM reasoner (/v1/chat/completions, media in -> text)
Also proxies async video jobs and serves the built React frontend.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import os
import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cosmos3_playground import __version__, modes
from cosmos3_playground.cosmos_client import CosmosClient, ReasonerClient

STATIC_DIR = Path(__file__).parent / "static"


def _gen_topology() -> dict[str, Any]:
    """Deployment parallel layout of the Cosmos generator, from env (set to match
    how the vLLM-Omni server was launched). world_size = cfg·ulysses·ring·tp·pp·dp."""
    def _d(name: str) -> int:
        try:
            return max(1, int(os.environ.get(name, "1") or 1))
        except ValueError:
            return 1
    dims = {"cfg": _d("COSMOS3_CFG_PARALLEL"), "ulysses": _d("COSMOS3_ULYSSES"),
            "ring": _d("COSMOS3_RING"), "tp": _d("COSMOS3_TP"),
            "pp": _d("COSMOS3_PP"), "dp": _d("COSMOS3_DP"), "vae": _d("COSMOS3_VAE_PP")}
    world = dims["cfg"] * dims["ulysses"] * dims["ring"] * dims["tp"] * dims["pp"] * dims["dp"]
    return {"dims": dims, "world_size": world}


def create_app(
    server_url: str,
    model: str | None = None,
    reasoner_url: str | None = None,
    reasoner_model: str | None = None,
    api_key: str = "EMPTY",
    policy_url: str | None = None,
    policy_model: str | None = None,
) -> FastAPI:
    gen = CosmosClient(server_url, api_key=api_key)
    reasoner = ReasonerClient(reasoner_url, api_key=api_key)
    # Optional dedicated policy checkpoint (Cosmos3-Nano-Policy-DROID) on a separate
    # server; the policy mode routes here, everything else stays on the base server.
    policy_gen = CosmosClient(policy_url, api_key=api_key) if policy_url else gen

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await gen.aclose()
            await reasoner.aclose()
            if policy_url:
                await policy_gen.aclose()

    app = FastAPI(title="Cosmos3 Playground", version=__version__, lifespan=lifespan)
    app.state.model = model
    app.state.policy_model = policy_model
    app.state.rollouts = {}  # rollout_id -> {status, chunk, total, error, path}

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        gen_model = model
        try:
            available = await gen.models()
            if not gen_model and available:
                gen_model = available[0]
        except Exception:
            available = []
        rmodel, ravail = reasoner_model, False
        if reasoner.configured:
            try:
                rmods = await reasoner.models()
                ravail = True
                if not rmodel and rmods:
                    rmodel = rmods[0]
            except Exception:
                ravail = False
        return {
            "version": __version__,
            "generator": {"url": server_url, "model": gen_model, "available_models": available,
                          "topology": _gen_topology()},
            "reasoner": {"url": reasoner_url, "model": rmodel, "available": ravail},
            **modes.catalog(),
        }

    @app.post("/api/generate")
    async def generate(
        mode: str = Form(...),
        params: str = Form("{}"),
        reference: UploadFile | None = None,
    ) -> Response:
        try:
            m = modes.mode(mode)
            p = json.loads(params or "{}")
        except (ValueError, json.JSONDecodeError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        if m["reference"] != "none" and reference is None:
            raise HTTPException(status_code=400, detail=f"mode '{mode}' needs a {m['reference']} reference")

        # ---------- REASON surface ----------
        if m["surface"] == "reason":
            if not reasoner.configured:
                raise HTTPException(status_code=503, detail="reasoner not connected (start with --reasoner-url)")
            media = await reference.read() if reference is not None else None
            payload = modes.build_reason_messages(
                mode, p, media,
                reference.filename if reference else None,
                reference.content_type if reference else None,
            )
            try:
                text = await reasoner.chat(payload)
            except httpx.HTTPError as err:
                raise HTTPException(status_code=502, detail=f"reasoner error: {err}") from err
            return JSONResponse({"kind": "text", "text": text})

        # ---------- GENERATE surface ----------
        # Transfer is locked to the control clip: output length is frame-aligned (paper Eq. 6,
        # one output frame per control frame) and the output aspect ratio must match the clip,
        # else structure distorts. Both are derived from the clip — not knobs the user must match.
        ref_bytes = None
        if mode == "transfer" and reference is not None:
            ref_bytes = await reference.read()
            try:
                frames = (await asyncio.to_thread(_decode_frames, ref_bytes))[0]
                n, h, w = (int(x) for x in frames.shape[:3])
                p["num_frames"] = max(5, min(n, 153))  # clamp to the supported transfer range
                # snap the output to the 720p bucket whose aspect ratio matches the control clip
                buckets = {16 / 9: "1280x720", 4 / 3: "1104x832", 1.0: "960x960",
                           3 / 4: "832x1104", 9 / 16: "720x1280"}
                p["size"] = min(buckets.items(), key=lambda kv: abs(kv[0] - w / h))[1]
            except Exception:  # noqa: BLE001 — fall back to the requested values on decode failure
                pass
        try:
            req = modes.build_request(mode, p)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        try:
            if req["kind"] == "image":
                payload = dict(req["fields"])
                if req["extra_params"]:
                    payload["extra_params"] = req["extra_params"]
                if app.state.model:
                    payload["model"] = app.state.model
                result = await gen.generate_image(payload)
                b64 = (result.get("data") or [{}])[0].get("b64_json")
                if not b64:
                    raise HTTPException(status_code=502, detail="cosmos returned no image")
                fmt = result.get("output_format", "png")
                # deliver the raw image bytes (not a base64 JSON blob ~33% larger) so the
                # browser can stream and progressively paint it — a 1024² PNG is multi-MB.
                return Response(content=base64.b64decode(b64), media_type=f"image/{fmt}")

            ref = None
            if reference is not None:
                data = ref_bytes if ref_bytes is not None else await reference.read()
                ref = (reference.filename or "ref", data,
                       reference.content_type or "application/octet-stream")
            job = await gen.create_video(modes.to_multipart_fields(req, app.state.model), ref)
            return JSONResponse({"kind": "video", "job_id": job.get("id"), "status": job.get("status"),
                                 "async_action": req.get("async_action", False)})
        except (httpx.HTTPError, RuntimeError) as err:
            raise HTTPException(status_code=502, detail=f"cosmos server error: {err}") from err

    async def _gen_one_chunk(req: dict[str, Any], ref: tuple[str, bytes, str], client, model_name) -> bytes:
        job = await client.create_video(modes.to_multipart_fields(req, model_name), ref)
        jid = job.get("id")
        res = await client.get_video(jid)
        for _ in range(300):
            if res.get("status") in ("completed", "succeeded", "failed", "error"):
                break
            await asyncio.sleep(2)
            res = await client.get_video(jid)
        if res.get("status") in ("failed", "error"):
            raise RuntimeError(str(res.get("error") or "chunk generation failed"))
        return b"".join([c async for c in client.stream_video_content(jid)])

    def _decode_frames(video: bytes):
        import imageio.v3 as iio
        frames = iio.imread(io.BytesIO(video), index=None, plugin="pyav")  # [T, H, W, 3]
        buf = io.BytesIO()
        iio.imwrite(buf, frames[-1], extension=".png")
        return frames, buf.getvalue()

    def _encode_concat(frame_lists: list, fps: int) -> str:
        import imageio.v3 as iio
        import numpy as np
        stitched = np.concatenate(frame_lists, axis=0)
        fd, path = tempfile.mkstemp(suffix=".mp4", dir="/home/zjy/cosmos3_storage")
        os.close(fd)
        iio.imwrite(path, stitched, fps=fps, plugin="pyav", codec="libx264")
        return path

    async def _run_rollout(mode: str, rid: str, params: dict[str, Any], n: int, img: bytes, iname: str, ctype: str) -> None:
        st = app.state.rollouts[rid]
        try:
            frame_lists = []
            prev_video: bytes | None = None
            fps = 10
            # The robot (bridge) policy runs on the base generator, like forward dynamics;
            # the dedicated DROID checkpoint (policy_gen) is not used by the current example.
            client = gen
            model_name = app.state.model
            for i in range(n):
                st["chunk"] = i + 1
                # policy predicts its own actions each chunk; forward dynamics replays chunk i.
                req = (modes.policy_single_chunk_request(params) if mode == "policy"
                       else modes.fd_single_chunk_request(params, i))
                fps = int(req["fields"]["fps"])
                if i == 0:
                    ref = (iname, img, ctype)  # first chunk: image first-frame (P=1, I2V start)
                else:
                    # subsequent chunks: condition on the PREVIOUS chunk's video tail (P>1, V2V
                    # continuation, paper Eq. 5) — preserves motion history, far less drift than
                    # re-seeding from a single last frame.
                    req["extra_params"]["condition_video_keep"] = "last"
                    ref = ("prev.mp4", prev_video, "video/mp4")
                video = await _gen_one_chunk(req, ref, client, model_name)
                frames, _ = await asyncio.to_thread(_decode_frames, video)
                frame_lists.append(frames)
                prev_video = video
            st["path"] = await asyncio.to_thread(_encode_concat, frame_lists, fps)
            st["status"] = "completed"
        except Exception as err:  # noqa: BLE001 — surface any failure to the client
            st["status"] = "error"
            st["error"] = str(err)

    @app.post("/api/rollout")
    async def rollout_start(mode: str = Form(...), params: str = Form("{}"),
                            reference: UploadFile | None = None) -> dict[str, Any]:
        """Start an autoregressive forward-dynamics rollout: generate one action chunk at a
        time, conditioning each on the previous chunk's last frame, then stitch the clips."""
        if mode != "fwd_dynamics":
            raise HTTPException(status_code=400, detail="rollout is only for forward dynamics")
        if reference is None:
            raise HTTPException(status_code=400, detail="rollout needs a first-frame image")
        try:
            p = json.loads(params or "{}")
        except json.JSONDecodeError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        # forward dynamics is bounded by the available action chunks; policy predicts its own,
        # so it can roll out as far as requested.
        n = max(1, int(p.get("rollout_chunks") or 1))
        if mode == "fwd_dynamics":
            n = min(n, modes.fd_chunk_count())
        img = await reference.read()
        rid = uuid.uuid4().hex
        app.state.rollouts[rid] = {"status": "running", "chunk": 0, "total": n, "error": None, "path": None}
        asyncio.create_task(_run_rollout(mode, rid, p, n, img, reference.filename or "frame.png",
                                         reference.content_type or "image/png"))
        return {"rollout_id": rid, "total": n}

    @app.get("/api/rollout/{rid}")
    async def rollout_status(rid: str) -> dict[str, Any]:
        st = app.state.rollouts.get(rid)
        if st is None:
            raise HTTPException(status_code=404, detail="unknown rollout")
        return {k: st[k] for k in ("status", "chunk", "total", "error")}

    @app.get("/api/rollout/{rid}/content")
    async def rollout_content(rid: str) -> FileResponse:
        st = app.state.rollouts.get(rid)
        if st is None or not st.get("path"):
            raise HTTPException(status_code=404, detail="rollout video not ready")
        return FileResponse(st["path"], media_type="video/mp4")

    @app.post("/api/validate")
    async def validate(payload: dict[str, Any]) -> dict[str, Any]:
        """Round-trip validation for forward dynamics: re-read the action out of the generated
        video with inverse dynamics (same domain) and score it against the original plan."""
        mode_id = payload.get("mode", "")
        params = payload.get("params") or {}
        job_id = payload.get("job_id")
        if mode_id != "fwd_dynamics":
            raise HTTPException(status_code=400, detail="round-trip validation only supports forward dynamics")
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id of the forward-dynamics video is required")
        try:
            original = modes.fd_action(params)
            id_req = modes.roundtrip_id_request(mode_id, params)
        except (ValueError, KeyError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        # pull the forward-dynamics video back out of the job store as the inverse-dynamics input
        try:
            chunks = [c async for c in gen.stream_video_content(job_id)]
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"could not read forward-dynamics video: {err}") from err
        video = b"".join(chunks)
        if not video:
            raise HTTPException(status_code=502, detail="forward-dynamics video is empty (job not finished?)")

        try:
            ref = (f"{job_id}.mp4", video, "video/mp4")
            id_job = await gen.create_video(modes.to_multipart_fields(id_req, app.state.model), ref)
        except (httpx.HTTPError, RuntimeError) as err:
            raise HTTPException(status_code=502, detail=f"inverse-dynamics run failed: {err}") from err

        id_job_id = id_job.get("id")
        result = await gen.get_video(id_job_id)
        for _ in range(200):
            if result.get("status") in ("completed", "succeeded", "failed", "error"):
                break
            await asyncio.sleep(2)
            result = await gen.get_video(id_job_id)
        if result.get("status") in ("failed", "error"):
            raise HTTPException(status_code=502, detail=f"inverse-dynamics job failed: {result.get('error')}")

        recovered = (result.get("action") or {}).get("data")
        if recovered is None:
            raise HTTPException(status_code=502, detail="inverse dynamics returned no action to compare")
        score = modes.compare_actions(original, recovered)
        return {"method": "forward→inverse round-trip", "domain": id_req["extra_params"].get("domain_name"),
                "score": score, "original_shape": [len(original), len(original[0]) if original else 0]}

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        try:
            return await gen.get_video(job_id)
        except Exception as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    @app.get("/api/jobs/{job_id}/content")
    async def job_content(job_id: str) -> StreamingResponse:
        return StreamingResponse(gen.stream_video_content(job_id), media_type="video/mp4")

    @app.post("/api/request-preview")
    async def request_preview(payload: dict[str, Any]) -> dict[str, Any]:
        """Reveal the exact normalized request for the current mode + settings (no run)."""
        try:
            return modes.request_preview(payload.get("mode", ""), payload.get("params") or {})
        except (ValueError, KeyError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

    @app.get("/api/example/{mode_id}/action")
    async def example_action(mode_id: str) -> dict[str, Any]:
        data = modes.example_action(mode_id)
        if data is None:
            raise HTTPException(status_code=404, detail="no action plan for this mode")
        return data

    @app.get("/api/example/{mode_id}/reference")
    async def example_reference(mode_id: str) -> FileResponse:
        path = modes.example_reference_path(mode_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no example reference for this mode")
        # no-store so swapping an example asset takes effect immediately (path is constant).
        return FileResponse(str(path), headers={"Cache-Control": "no-store"})

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


def app_from_env() -> FastAPI:
    return create_app(
        os.environ.get("COSMOS3_SERVER_URL", "http://127.0.0.1:8000"),
        os.environ.get("COSMOS3_MODEL") or None,
        os.environ.get("COSMOS3_REASONER_URL") or None,
        os.environ.get("COSMOS3_REASONER_MODEL") or None,
        os.environ.get("COSMOS3_API_KEY", "EMPTY"),
        os.environ.get("COSMOS3_POLICY_URL") or None,
        os.environ.get("COSMOS3_POLICY_MODEL") or None,
    )
