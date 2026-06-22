# SPDX-License-Identifier: Apache-2.0
"""Unified Cosmos3 mode catalog — one interface over two surfaces:

* ``generate`` — diffusion world generation -> media (image / video [+ sound] / action
  rollout). Served by a vLLM-Omni Cosmos3 server (``/v1/images``, ``/v1/videos``).
* ``reason``  — world understanding -> text (captioning, temporal localization, 2D
  grounding, physical-plausibility, free Q&A). Served by an OpenAI-compatible vLLM
  reasoner (``/v1/chat/completions`` with image/video in).

This is the single source of truth: the backend builds requests from it and the
frontend renders the whole UI from it (GET /api/config).
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).parent / "examples"

_VIDEO_DEFAULTS = {
    "size": "1280x720", "num_frames": 93, "fps": 24,
    "num_inference_steps": 35, "guidance_scale": 6.0, "flow_shift": 10.0,
}
_IMAGE_DEFAULTS = {"size": "1024x1024", "num_inference_steps": 50, "guidance_scale": 7.0}
# Action / dynamics: low guidance, low flow-shift, small clips (model card).
_ACTION_DEFAULTS = {"size": "832x480", "fps": 10, "num_inference_steps": 30, "guidance_scale": 1.0, "flow_shift": 5.0}

# ----------------------------------------------------------------------------- modes
MODES: list[dict[str, Any]] = [
    # ---- GENERATE ----
    {"id": "t2i", "label": "Text → Image", "surface": "generate", "group": "Imagine",
     "kind": "image", "reference": "none", "blurb": "Generate a still image from a prompt.",
     "example": {"prompt": "A modern industrial robotic arm with a polished silver body cleaning a white "
                 "ceramic plate in a bright kitchen, photorealistic, sharp detail",
                 "params": _IMAGE_DEFAULTS, "reference": None}},
    {"id": "t2v", "label": "Text → Video", "surface": "generate", "group": "Imagine",
     "kind": "video", "reference": "none", "blurb": "Imagine a video world from a prompt.",
     "example": {"prompt": "A robotic arm in a bright kitchen picks up a green sponge and cleans a white plate, "
                 "smooth realistic motion, photorealistic. Audio description: soft servo whirs, gentle "
                 "sponge-on-ceramic sounds, faint water drips.", "params": _VIDEO_DEFAULTS, "reference": None}},
    {"id": "i2v", "label": "Image → Video", "surface": "generate", "group": "Animate",
     "kind": "video", "reference": "image", "blurb": "Animate a still image into a video.",
     "example": {"prompt": "The scene comes to life with gentle, natural motion and a slow cinematic push-in.",
                 "params": _VIDEO_DEFAULTS, "reference": "i2v_input.jpg"}},
    {"id": "v2v", "label": "Video → Video", "surface": "generate", "group": "Edit",
     "kind": "video", "reference": "video", "blurb": "Continue / re-imagine a reference video.",
     "extra": [{"key": "condition_video_keep", "label": "Keep", "type": "select",
                "options": ["first", "last"], "default": "first"}],
     "example": {"prompt": "Continue the scene with natural, physically plausible motion, preserving style.",
                 "params": {**_VIDEO_DEFAULTS, "condition_video_keep": "first"}, "reference": "ref_video.mp4"}},
    {"id": "transfer", "label": "Transfer (control)", "surface": "generate", "group": "Edit",
     "kind": "video", "reference": "video", "blurb": "Structure-guided generation (ControlNet-style).",
     "extra": [{"key": "control", "label": "Control", "type": "select",
                "options": ["edge", "blur", "depth", "seg", "wsm"], "default": "edge"},
               {"key": "control_guidance", "label": "Control guidance", "type": "number",
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}],
     "example": {"prompt": "A photorealistic video that follows the structure and motion of the control input.",
                 "params": {**_VIDEO_DEFAULTS, "control": "edge", "control_guidance": 1.0},
                 "reference": "ref_video.mp4"}},
    {"id": "fwd_dynamics", "label": "Forward dynamics", "surface": "generate", "group": "Simulate",
     "kind": "video", "reference": "image", "blurb": "Action-conditioned future prediction: roll out a video "
     "from a first frame + an action trajectory.",
     "example": {"prompt": "Roll out the future given the action trajectory.",
                 "params": _ACTION_DEFAULTS, "reference": "fd_first_frame.png", "action": "fd_action_chunks.json"}},
    {"id": "inv_dynamics", "label": "Inverse dynamics", "surface": "generate", "group": "Simulate",
     "kind": "video", "reference": "video", "blurb": "Recover the ego-motion / action trajectory from a video.",
     "example": {"prompt": "Recover the action trajectory from the video.",
                 "params": {**_ACTION_DEFAULTS, "num_frames": 61}, "reference": "id_av_input.mp4"}},
    # ---- REASON ----
    {"id": "caption", "label": "Captioning", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "image", "blurb": "Detailed description of an image or video.",
     "example": {"prompt": "Describe this scene in detail.", "params": {}, "reference": "reason_image.png"}},
    {"id": "temporal", "label": "Temporal localization", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "video", "blurb": "When does an event happen? (timestamps)",
     "example": {"prompt": "Identify the key event in this video and report its start and end timestamps.",
                 "params": {}, "reference": "id_av_input.mp4"}},
    {"id": "grounding", "label": "2D grounding", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "image", "blurb": "Locate an object and return its 2D coordinates.",
     "example": {"prompt": "Locate the main object in the image and return its 2D bounding box [x1,y1,x2,y2].",
                 "params": {}, "reference": "reason_image.png"}},
    {"id": "physical", "label": "Physical reasoning", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "video", "blurb": "Is what happens physically plausible?",
     "example": {"prompt": "Is what happens in this video physically plausible? Explain your reasoning step by step.",
                 "params": {}, "reference": "id_av_input.mp4"}},
    {"id": "vqa", "label": "Ask anything", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "image", "blurb": "Free-form question about an image or video.",
     "example": {"prompt": "What is happening in this image, and what might happen next?",
                 "params": {}, "reference": "reason_image.png"}},
]

GEN_KNOBS: list[dict[str, Any]] = [
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
]

REASON_KNOBS: list[dict[str, Any]] = [
    {"key": "max_tokens", "label": "Max tokens", "type": "int", "default": 512, "min": 16, "max": 4096},
    {"key": "temperature", "label": "Temperature", "type": "number", "default": 0.2, "min": 0.0, "max": 1.5, "step": 0.1},
]

_MODE_BY_ID = {m["id"]: m for m in MODES}


def catalog() -> dict[str, Any]:
    return {"modes": MODES, "gen_knobs": GEN_KNOBS, "reason_knobs": REASON_KNOBS}


def mode(mode_id: str) -> dict[str, Any]:
    m = _MODE_BY_ID.get(mode_id)
    if m is None:
        raise ValueError(f"unknown mode: {mode_id}")
    return m


def example_reference_path(mode_id: str) -> Path | None:
    ref = (_MODE_BY_ID.get(mode_id) or {}).get("example", {}).get("reference")
    if not ref:
        return None
    p = EXAMPLES_DIR / ref
    return p if p.is_file() else None


def _num(params: dict[str, Any], key: str, cast):
    v = params.get(key)
    if v in (None, ""):
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------- generate requests
def build_request(mode_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Normalized GENERATE request: {kind, reference, fields, extra_params}."""
    m = mode(mode_id)
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

    if m["kind"] == "video":
        for key, cast in (("num_frames", int), ("fps", int), ("flow_shift", float)):
            val = _num(params, key, cast)
            if val is not None:
                fields[key] = val
        if params.get("generate_sound"):
            fields["generate_sound"] = True
            nf, fps = _num(params, "num_frames", int), _num(params, "fps", int)
            if nf and fps:
                fields["sound_duration"] = round(nf / fps, 3)

    if mode_id == "v2v":
        extra_params["condition_video_keep"] = params.get("condition_video_keep", "first")
        extra_params["condition_frame_indexes_vision"] = [0, 1]
    elif mode_id == "transfer":
        extra_params[params.get("control", "edge")] = True
        cg = _num(params, "control_guidance", float)
        if cg is not None:
            extra_params["control_guidance"] = cg
    elif mode_id == "fwd_dynamics":
        spec = json.loads((EXAMPLES_DIR / "fd_action_chunks.json").read_text())
        chunk_size = int(spec.get("action_chunk_size", 16))
        extra_params.update({"action_mode": "forward_dynamics", "domain_name": spec.get("domain_name", "agibotworld"),
                             "action_chunk_size": chunk_size, "action": spec["action_chunks"][0]})
        fields["num_frames"] = chunk_size + 1
    elif mode_id == "inv_dynamics":
        extra_params.update({"action_mode": "inverse_dynamics", "domain_name": "av",
                             "action_chunk_size": 60, "raw_action_dim": 9})
        fields["num_frames"] = 61

    # forward/inverse dynamics read the action back from the async job, like the cookbook.
    return {"kind": m["kind"], "reference": m["reference"], "fields": fields, "extra_params": extra_params,
            "async_action": mode_id in ("fwd_dynamics", "inv_dynamics")}


def to_multipart_fields(req: dict[str, Any], model_name: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in req["fields"].items():
        out[k] = v if isinstance(v, str) else json.dumps(v) if isinstance(v, bool) else str(v)
    if req["extra_params"]:
        out["extra_params"] = json.dumps(req["extra_params"])
    if model_name:
        out["model"] = model_name
    return out


# ------------------------------------------------------------------- reason requests
def build_reason_messages(mode_id: str, params: dict[str, Any], media: bytes | None,
                          media_name: str | None) -> dict[str, Any]:
    """Build an OpenAI chat request for the REASON surface (media in -> text out)."""
    m = mode(mode_id)
    prompt = (params.get("prompt") or "").strip() or (m.get("example", {}).get("prompt") or "Describe this.")
    content: list[dict[str, Any]] = []
    if media is not None:
        mime = mimetypes.guess_type(media_name or "")[0] or "application/octet-stream"
        data_uri = f"data:{mime};base64," + base64.b64encode(media).decode()
        if mime.startswith("video"):
            content.append({"type": "video_url", "video_url": {"url": data_uri}})
        else:
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
    content.append({"type": "text", "text": prompt})
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": _num(params, "max_tokens", int) or 512,
        "temperature": _num(params, "temperature", float) if _num(params, "temperature", float) is not None else 0.2,
    }
    return payload
