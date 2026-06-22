# SPDX-License-Identifier: Apache-2.0
"""The Cosmos3 mode catalog — the single source of truth shared by the backend
(to build vLLM-Omni requests) and the frontend (to render the UI).

Cosmos3 is one pipeline with the task chosen *per request*; each "mode" here maps
a user-facing task to an endpoint + reference kind + the extra knobs that fold into
the request's ``extra_params``.
"""

from __future__ import annotations

import json
from typing import Any

# kind: "image" -> POST /v1/images/generations (sync, b64);
#       "video" -> POST /v1/videos (async job) + GET /v1/videos/{id}/content
# reference: what the user attaches (drives the upload widget)
MODES: list[dict[str, Any]] = [
    {"id": "t2i", "label": "Text → Image", "group": "Imagine", "kind": "image", "reference": "none",
     "blurb": "Generate a still image from a prompt."},
    {"id": "t2v", "label": "Text → Video", "group": "Imagine", "kind": "video", "reference": "none",
     "blurb": "Imagine a video world from a prompt."},
    {"id": "i2v", "label": "Image → Video", "group": "Animate", "kind": "video", "reference": "image",
     "blurb": "Animate a still image into a video."},
    {"id": "v2v", "label": "Video → Video", "group": "Edit", "kind": "video", "reference": "video",
     "blurb": "Continue / re-imagine a reference video.",
     "extra": [
         {"key": "condition_video_keep", "label": "Keep", "type": "select",
          "options": ["first", "last"], "default": "first"},
     ]},
    {"id": "transfer", "label": "Transfer (control)", "group": "Edit", "kind": "video", "reference": "video",
     "blurb": "Structure-guided generation (ControlNet-style) from a control video.",
     "extra": [
         {"key": "control", "label": "Control", "type": "select",
          "options": ["edge", "blur", "depth", "seg", "wsm"], "default": "edge"},
         {"key": "control_guidance", "label": "Control guidance", "type": "number", "default": 1.0,
          "min": 0.0, "max": 2.0, "step": 0.05},
     ]},
]

# Common diffusion knobs (rendered by the frontend; video-only ones flagged).
KNOBS: list[dict[str, Any]] = [
    {"key": "negative_prompt", "label": "Negative prompt", "type": "text", "default": ""},
    {"key": "size", "label": "Resolution", "type": "select",
     "options": ["1280x720", "720x1280", "960x720", "720x960", "1024x1024", "832x480", "480x832"],
     "default": "1280x720"},
    {"key": "num_frames", "label": "Frames", "type": "int", "default": 93, "min": 1, "max": 257, "video": True},
    {"key": "fps", "label": "FPS", "type": "int", "default": 24, "min": 4, "max": 60, "video": True},
    {"key": "num_inference_steps", "label": "Steps", "type": "int", "default": 35, "min": 1, "max": 100},
    {"key": "guidance_scale", "label": "Guidance", "type": "number", "default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5},
    {"key": "flow_shift", "label": "Flow shift", "type": "number", "default": 10.0, "min": 0.0, "max": 20.0,
     "step": 0.5, "video": True},
    {"key": "seed", "label": "Seed", "type": "int", "default": None, "min": 0, "max": 2**31 - 1},
    {"key": "generate_sound", "label": "Generate sound", "type": "bool", "default": False, "video": True},
    {"key": "sound_duration", "label": "Sound duration (s)", "type": "number", "default": 5.0, "video": True},
]

_MODE_BY_ID = {m["id"]: m for m in MODES}


def catalog() -> dict[str, Any]:
    return {"modes": MODES, "knobs": KNOBS}


def _num(params: dict[str, Any], key: str, cast):
    v = params.get(key)
    if v in (None, ""):
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def build_request(mode_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Turn a mode + params into a normalized request descriptor.

    Returns ``{"kind": "image"|"video", "fields": {...}, "extra_params": {...}}``.
    The caller attaches any reference file and dispatches to the right endpoint.
    """
    mode = _MODE_BY_ID.get(mode_id)
    if mode is None:
        raise ValueError(f"unknown mode: {mode_id}")

    extra_params: dict[str, Any] = {}
    fields: dict[str, Any] = {"prompt": (params.get("prompt") or "").strip()}
    if not fields["prompt"]:
        raise ValueError("prompt is required")

    if params.get("negative_prompt"):
        fields["negative_prompt"] = params["negative_prompt"]
    if params.get("size"):
        fields["size"] = params["size"]
    for key, cast in (("num_inference_steps", int), ("guidance_scale", float), ("seed", int)):
        val = _num(params, key, cast)
        if val is not None:
            fields[key] = val

    if mode["kind"] == "video":
        for key, cast in (("num_frames", int), ("fps", int), ("flow_shift", float)):
            val = _num(params, key, cast)
            if val is not None:
                fields[key] = val
        if params.get("generate_sound"):
            fields["generate_sound"] = True
            dur = _num(params, "sound_duration", float)
            if dur is not None:
                fields["sound_duration"] = dur

    # mode-specific extras -> extra_params
    if mode_id == "v2v":
        extra_params["condition_video_keep"] = params.get("condition_video_keep", "first")
        extra_params["condition_frame_indexes_vision"] = [0, 1]
    elif mode_id == "transfer":
        control = params.get("control", "edge")
        extra_params[control] = True
        cg = _num(params, "control_guidance", float)
        if cg is not None:
            extra_params["control_guidance"] = cg

    return {"kind": mode["kind"], "reference": mode["reference"], "fields": fields, "extra_params": extra_params}


def to_multipart_fields(req: dict[str, Any], model: str | None) -> dict[str, str]:
    """Flatten a video request descriptor into multipart form fields (strings)."""
    out: dict[str, str] = {}
    for k, v in req["fields"].items():
        out[k] = v if isinstance(v, str) else json.dumps(v) if isinstance(v, bool) else str(v)
    if req["extra_params"]:
        out["extra_params"] = json.dumps(req["extra_params"])
    if model:
        out["model"] = model
    return out
