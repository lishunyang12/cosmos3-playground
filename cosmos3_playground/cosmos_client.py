# SPDX-License-Identifier: Apache-2.0
"""Thin async client for a vLLM-Omni server serving Cosmos3 (OpenAI-compatible)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx


class CosmosClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, headers={"authorization": f"Bearer {api_key}"})

    async def aclose(self) -> None:
        await self._client.aclose()

    async def models(self) -> list[str]:
        r = await self._client.get(f"{self.base_url}/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def generate_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = {"response_format": "b64_json", **payload}
        r = await self._client.post(f"{self.base_url}/v1/images/generations", json=payload)
        r.raise_for_status()
        return r.json()

    async def create_video(self, fields: dict[str, str], reference: tuple[str, bytes, str] | None) -> dict[str, Any]:
        files = {}
        if reference is not None:
            filename, data, content_type = reference
            files["input_reference"] = (filename, data, content_type)
        # multipart when there's a file, else plain form fields
        r = await self._client.post(f"{self.base_url}/v1/videos", data=fields, files=files or None)
        r.raise_for_status()
        return r.json()

    async def get_video(self, video_id: str) -> dict[str, Any]:
        # A failed job returns HTTP 500 with a valid JSON body (status="failed", error=...).
        # Return the body rather than raising, so the UI can surface the real error.
        r = await self._client.get(f"{self.base_url}/v1/videos/{video_id}")
        try:
            return r.json()
        except ValueError:
            r.raise_for_status()
            raise

    async def stream_video_content(self, video_id: str) -> AsyncIterator[bytes]:
        async with self._client.stream("GET", f"{self.base_url}/v1/videos/{video_id}/content") as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk


class ReasonerClient:
    """OpenAI-compatible chat client for the REASON surface (image/video -> text)."""

    def __init__(self, base_url: str | None, api_key: str = "EMPTY", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self._client = httpx.AsyncClient(timeout=timeout, headers={"authorization": f"Bearer {api_key}"})

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def models(self) -> list[str]:
        if not self.base_url:
            return []
        r = await self._client.get(f"{self.base_url}/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def chat(self, payload: dict[str, Any]) -> str:
        if not self.base_url:
            raise RuntimeError("reasoner not configured (set --reasoner-url)")
        r = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
