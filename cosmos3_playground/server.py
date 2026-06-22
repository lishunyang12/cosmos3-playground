# SPDX-License-Identifier: Apache-2.0
"""FastAPI backend for the unified Cosmos3 Playground.

Routes a playground mode to the right surface:
* generate -> vLLM-Omni Cosmos3 server (/v1/images, /v1/videos)
* reason   -> OpenAI-compatible vLLM reasoner (/v1/chat/completions, media in -> text)
Also proxies async video jobs and serves the built React frontend.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cosmos3_playground import __version__, modes
from cosmos3_playground.cosmos_client import CosmosClient, ReasonerClient

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    server_url: str,
    model: str | None = None,
    reasoner_url: str | None = None,
    reasoner_model: str | None = None,
    api_key: str = "EMPTY",
) -> FastAPI:
    gen = CosmosClient(server_url, api_key=api_key)
    reasoner = ReasonerClient(reasoner_url, api_key=api_key)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await gen.aclose()
            await reasoner.aclose()

    app = FastAPI(title="Cosmos3 Playground", version=__version__, lifespan=lifespan)
    app.state.model = model

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
            "generator": {"url": server_url, "model": gen_model, "available_models": available},
            "reasoner": {"url": reasoner_url, "model": rmodel, "available": ravail},
            **modes.catalog(),
        }

    @app.post("/api/generate")
    async def generate(
        mode: str = Form(...),
        params: str = Form("{}"),
        reference: UploadFile | None = None,
    ) -> JSONResponse:
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
            payload = modes.build_reason_messages(mode, p, media, reference.filename if reference else None)
            try:
                text = await reasoner.chat(payload)
            except httpx.HTTPError as err:
                raise HTTPException(status_code=502, detail=f"reasoner error: {err}") from err
            return JSONResponse({"kind": "text", "text": text})

        # ---------- GENERATE surface ----------
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
                return JSONResponse({"kind": "image", "b64": b64, "format": result.get("output_format", "png")})

            ref = None
            if reference is not None:
                ref = (reference.filename or "ref", await reference.read(),
                       reference.content_type or "application/octet-stream")
            job = await gen.create_video(modes.to_multipart_fields(req, app.state.model), ref)
            return JSONResponse({"kind": "video", "job_id": job.get("id"), "status": job.get("status"),
                                 "async_action": req.get("async_action", False)})
        except (httpx.HTTPError, RuntimeError) as err:
            raise HTTPException(status_code=502, detail=f"cosmos server error: {err}") from err

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        try:
            return await gen.get_video(job_id)
        except Exception as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    @app.get("/api/jobs/{job_id}/content")
    async def job_content(job_id: str) -> StreamingResponse:
        return StreamingResponse(gen.stream_video_content(job_id), media_type="video/mp4")

    @app.get("/api/example/{mode_id}/reference")
    async def example_reference(mode_id: str) -> FileResponse:
        path = modes.example_reference_path(mode_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no example reference for this mode")
        return FileResponse(str(path))

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
    )
