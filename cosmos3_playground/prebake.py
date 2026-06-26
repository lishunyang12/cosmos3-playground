# SPDX-License-Identifier: Apache-2.0
"""Pre-bake every mode's example output into a static gallery.

Drives the *running* playground HTTP API exactly like the browser's Generate button
(``onGenerate`` + ``pollJob``), then saves each result under ``prebaked/``:

* ``<id>.png`` / ``<id>.mp4`` — the generated media (omitted for action-only modes)
* ``<id>.json``               — metadata the frontend loads to render the cached hero
                                ({kind, media, text, action, profiling})

The server serves these via ``/api/example/{id}/result`` so opening the URL shows the
cached output by default; pressing Generate re-runs the mode live and replaces it.

Usage (with the playground already running):
    python -m cosmos3_playground.prebake --base http://127.0.0.1:8800
    python -m cosmos3_playground.prebake --only t2i,policy      # subset
    python -m cosmos3_playground.prebake --skip transfer        # all but these
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PREBAKE_DIR = Path(__file__).parent / "prebaked"
EXAMPLES_DIR = Path(__file__).parent / "examples"

# Terminal job states reported by the vLLM-Omni video job store.
_DONE = {"completed", "succeeded"}
_FAILED = {"failed", "error"}


def _params_for(mode: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the exact params the UI submits on mode load (App.jsx example effect):
    knob defaults, then extra defaults, then the example overrides."""
    knobs = cfg["reason_knobs"] if mode["surface"] == "reason" else cfg["gen_knobs"]
    ex = mode.get("example") or {}
    out: dict[str, Any] = {"prompt": ex.get("prompt") or ""}
    for k in knobs:
        if k.get("default") is not None:
            out[k["key"]] = k["default"]
    for e in mode.get("extra") or []:
        if e.get("default") is not None:
            out[e["key"]] = e["default"]
    out.update(ex.get("params") or {})
    return out


def _reference(mode: dict[str, Any]) -> tuple[str, bytes, str] | None:
    ref = (mode.get("example") or {}).get("reference")
    if not ref:
        return None
    path = EXAMPLES_DIR / ref
    if not path.is_file():
        print(f"  ! reference {ref} missing — skipping", file=sys.stderr)
        return None
    ctype = "video/mp4" if path.suffix.lower() == ".mp4" else (
        "image/png" if path.suffix.lower() == ".png" else "image/jpeg")
    return ref, path.read_bytes(), ctype


def _poll_job(client: httpx.Client, base: str, job_id: str, timeout_s: int = 1800) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        j = client.get(f"{base}/api/jobs/{job_id}").json()
        if j.get("status") in _DONE | _FAILED:
            return j
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def _bake_one(client: httpx.Client, base: str, mode: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    mid = mode["id"]
    params = _params_for(mode, cfg)
    files = {"mode": (None, mid), "params": (None, json.dumps(params))}
    ref = _reference(mode)
    if ref is not None:
        files["reference"] = ref

    resp = client.post(f"{base}/api/generate", files=files, timeout=120)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")

    # ---- image: raw bytes returned synchronously ----
    if ctype.startswith("image/"):
        ext = ctype.split("/", 1)[1].split(";")[0] or "png"
        media = f"{mid}.{ext}"
        (PREBAKE_DIR / media).write_bytes(resp.content)
        return {"kind": "image", "media": media}

    data = resp.json()
    kind = data.get("kind")

    # ---- text (reason surface) ----
    if kind == "text":
        return {"kind": "text", "media": None, "text": data.get("text", "")}

    # ---- video / action (async job) ----
    if kind == "video":
        job = _poll_job(client, base, data["job_id"])
        if job.get("status") in _FAILED:
            raise RuntimeError(f"job failed: {job.get('error')}")
        meta: dict[str, Any] = {"kind": "video", "media": None}
        if job.get("action") is not None:
            meta["action"] = job["action"]
        prof = {k: job[k] for k in ("inference_time_s", "peak_memory_mb") if job.get(k) is not None}
        if prof:
            meta["profiling"] = prof
        # action-only modes (inverse dynamics) render an overlay on the *reference* clip,
        # not a generated video — store just the action, no media.
        if mode.get("primary_output") == "action":
            return meta
        content = client.get(f"{base}/api/jobs/{data['job_id']}/content", timeout=300).content
        if content:
            media = f"{mid}.mp4"
            (PREBAKE_DIR / media).write_bytes(content)
            meta["media"] = media
        return meta

    raise RuntimeError(f"unexpected response kind: {kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-bake playground mode outputs into a static gallery.")
    ap.add_argument("--base", default="http://127.0.0.1:8800", help="running playground base URL")
    ap.add_argument("--only", default="", help="comma-separated mode ids to bake (default: all)")
    ap.add_argument("--skip", default="", help="comma-separated mode ids to skip")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    PREBAKE_DIR.mkdir(exist_ok=True)

    with httpx.Client() as client:
        cfg = client.get(f"{args.base}/api/config", timeout=30).json()
        modes = cfg["modes"]
        if only:
            modes = [m for m in modes if m["id"] in only]
        modes = [m for m in modes if m["id"] not in skip]

        ok, failed = 0, []
        for m in modes:
            mid = m["id"]
            print(f"• {mid} ({m['surface']}) …", flush=True)
            t0 = time.monotonic()
            try:
                meta = _bake_one(client, args.base, m, cfg)
                meta["mode"] = mid
                meta["baked_seconds"] = round(time.monotonic() - t0, 1)
                (PREBAKE_DIR / f"{mid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
                ok += 1
                print(f"  ✓ {meta['kind']}"
                      + (f" → {meta['media']}" if meta.get("media") else "")
                      + f"  ({meta['baked_seconds']}s)", flush=True)
            except Exception as err:  # noqa: BLE001 — keep going; report the rest at the end
                failed.append((mid, str(err)))
                print(f"  ✗ {err}", file=sys.stderr, flush=True)

    print(f"\nbaked {ok}/{len(modes)} modes into {PREBAKE_DIR}")
    if failed:
        print("failed:")
        for mid, err in failed:
            print(f"  - {mid}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
