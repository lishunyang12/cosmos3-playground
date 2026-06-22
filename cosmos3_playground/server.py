# SPDX-License-Identifier: Apache-2.0
"""FastAPI backend for the Cosmos3 Playground.

Thin layer over a vLLM-Omni Cosmos3 server: it maps a playground mode + params to
the right OpenAI-compatible request, dispatches it, proxies async video jobs, and
serves the built React frontend.
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
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cosmos3_playground import __version__, modes
from cosmos3_playground.cosmos_client import CosmosClient

STATIC_DIR = Path(__file__).parent / "static"


def create_app(server_url: str, model: str | None = None, api_key: str = "EMPTY") -> FastAPI:
    client = CosmosClient(server_url, api_key=api_key)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    app = FastAPI(title="Cosmos3 Playground", version=__version__, lifespan=lifespan)
    app.state.model = model

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        resolved = model
        try:
            available = await client.models()
            if not resolved and available:
                resolved = available[0]
        except Exception:
            available = []
        return {
            "version": __version__,
            "server_url": server_url,
            "model": resolved,
            "available_models": available,
            **modes.catalog(),
        }

    @app.post("/api/generate")
    async def generate(
        mode: str = Form(...),
        params: str = Form("{}"),
        reference: UploadFile | None = None,
    ) -> JSONResponse:
        try:
            req = modes.build_request(mode, json.loads(params or "{}"))
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        if req["reference"] != "none" and reference is None:
            raise HTTPException(status_code=400, detail=f"mode '{mode}' needs a {req['reference']} reference")

        try:
            if req["kind"] == "image":
                payload = dict(req["fields"])
                if req["extra_params"]:
                    payload["extra_params"] = req["extra_params"]
                if app.state.model:
                    payload["model"] = app.state.model
                result = await client.generate_image(payload)
                b64 = (result.get("data") or [{}])[0].get("b64_json")
                return JSONResponse({"kind": "image", "b64": b64, "format": result.get("output_format", "png")})

            ref = None
            if reference is not None:
                ref = (reference.filename or "ref", await reference.read(), reference.content_type or "application/octet-stream")
            fields = modes.to_multipart_fields(req, app.state.model)
            job = await client.create_video(fields, ref)
            return JSONResponse({"kind": "video", "job_id": job.get("id"), "status": job.get("status")})
        except httpx.HTTPError as err:
            raise HTTPException(status_code=502, detail=f"cosmos server error: {err}") from err

    @app.get("/api/jobs/{job_id}")
    async def job(job_id: str) -> dict[str, Any]:
        try:
            return await client.get_video(job_id)
        except Exception as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    @app.get("/api/jobs/{job_id}/content")
    async def job_content(job_id: str) -> StreamingResponse:
        return StreamingResponse(client.stream_video_content(job_id), media_type="video/mp4")

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


def app_from_env() -> FastAPI:
    return create_app(
        os.environ.get("COSMOS3_SERVER_URL", "http://127.0.0.1:8000"),
        os.environ.get("COSMOS3_MODEL") or None,
        os.environ.get("COSMOS3_API_KEY", "EMPTY"),
    )
