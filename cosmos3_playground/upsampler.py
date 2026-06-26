"""Cosmos3 prompt upsampler — a thin port of cosmos_framework/inference/prompt_upsampling.py.

Cosmos3's official text/image/video pipeline is two-stage: a short natural-language prompt
is "upsampled" by the Reasoner into a dense structured-caption JSON, and the generator is
conditioned on that JSON (the model is trained on these captions). This module builds the
upsampler chat request and parses the response; the caller runs it through the connected
Reasoner (an OpenAI-compatible /v1/chat/completions endpoint).
"""

from __future__ import annotations

import json
import re
from string import Template
from typing import Any

# Valid (resolution, aspect_ratio) -> (W, H) buckets (verbatim from the framework).
RESOLUTION_RATIO_DICT: dict[str, dict[str, dict[str, int]]] = {
    "256": {"1,1": {"W": 256, "H": 256}, "4,3": {"W": 320, "H": 256}, "3,4": {"W": 256, "H": 320},
            "16,9": {"W": 320, "H": 192}, "9,16": {"W": 192, "H": 320}},
    "480": {"1,1": {"W": 640, "H": 640}, "4,3": {"W": 736, "H": 544}, "3,4": {"W": 544, "H": 736},
            "16,9": {"W": 832, "H": 480}, "9,16": {"W": 480, "H": 832}},
    "720": {"1,1": {"W": 960, "H": 960}, "4,3": {"W": 1104, "H": 832}, "3,4": {"W": 832, "H": 1104},
            "16,9": {"W": 1280, "H": 720}, "9,16": {"W": 720, "H": 1280}},
    "768": {"1,1": {"W": 1024, "H": 1024}, "4,3": {"W": 1184, "H": 880}, "3,4": {"W": 880, "H": 1184},
            "16,9": {"W": 1360, "H": 768}, "9,16": {"W": 768, "H": 1360}},
}

SYSTEM_MESSAGE = {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}

_T2I_SCHEMA = """{
  "subjects": [{"description": "", "appearance_details": "", "relationship": "", "location": "",
    "relative_size": "", "orientation": "", "pose": "", "clothing": "", "expression": "", "gender": "",
    "age": "", "skin_tone_and_texture": "", "facial_features": "", "number_of_subjects": 0,
    "number_of_arms": 0, "number_of_legs": 0, "number_of_hands": 0, "number_of_fingers": 0}],
  "subject_details": {}, "background_setting": "",
  "lighting": {"conditions": "", "direction": "", "shadows": "", "illumination_effect": ""},
  "aesthetics": {"composition": "", "color_scheme": "", "mood_atmosphere": "", "patterns": ""},
  "cinematography": {"framing": "", "camera_angle": "", "depth_of_field": "", "focus": "", "lens_focal_length": ""},
  "style_medium": "", "artistic_style": "", "context": "",
  "text_and_signage_elements": [{"text": "", "category": "", "appearance": "", "spatial": "", "context": ""}],
  "quadrant_scan": {"top_left": "", "top_right": "", "bottom_left": "", "bottom_right": "", "absolute_center": ""},
  "comprehensive_t2i_caption": "",
  "resolution": {"H": 0, "W": 0}, "aspect_ratio": "16,9"}"""

_T2V_SCHEMA = """{
  "subjects": [{"description": "", "appearance_details": "", "relationship": "", "location": "",
    "relative_size": "", "orientation": "", "pose": "", "action": "", "state_changes": "", "clothing": "",
    "expression": "", "gender": "", "age": "", "skin_tone_and_texture": "", "facial_features": "",
    "number_of_subjects": 0, "number_of_arms": 0, "number_of_legs": 0}],
  "background_setting": "",
  "lighting": {"conditions": "", "direction": "", "shadows": "", "illumination_effect": ""},
  "aesthetics": {"composition": "", "color_scheme": "", "mood_atmosphere": "", "patterns": ""},
  "cinematography": {"camera_motion": "", "framing": "", "camera_angle": "", "depth_of_field": "", "focus": "", "lens_focal_length": ""},
  "style_medium": "", "artistic_style": "", "context": "",
  "actions": [{"time": "0:00-0:05", "description": ""}],
  "text_and_signage_elements": [{"text": "", "category": "", "appearance": "", "spatial_temporal": "", "context": ""}],
  "segments": [{"segment_index": 0, "time_range": "0:00-0:05", "description": "", "key_changes": "", "camera": ""}],
  "transitions": [], "temporal_caption": "", "audio_description": "",
  "resolution": {"H": 0, "W": 0}, "aspect_ratio": "16,9", "duration": "5s", "fps": 24}"""

_T2I_TEMPLATE = Template(
    "Given the user's natural-language request below, generate a dense structured JSON that fully "
    "describes the image to be produced. The JSON must strictly follow the template provided after the "
    "request, including every top-level key and every nested sub-field.\n\n"
    "The output is always DENSE. Even when the request is brief, you must infer plausible, scene-consistent "
    "details for every field. Be creative but stay grounded: your additions must be physically plausible and "
    "internally consistent with the request.\n\n"
    "Return only the JSON object wrapped in a ```json code fence.\n\n$nl_description\n\n"
    "All top-level keys must always be present in the output; fill unused fields with \"\", 0, {}, or [] as "
    "appropriate.\n\n$json_template\n\nresolution_ratio_dict = $resolution_ratio_dict")

_T2V_TEMPLATE = Template(
    "$intro, generate a dense structured JSON that fully describes the video to be produced. The JSON must "
    "strictly follow the template provided after the request, including every top-level key and every nested "
    "sub-field.\n\nThe output is always DENSE. Even when the request is brief, you must infer plausible, "
    "scene-consistent details for every field. Be creative but stay grounded: your additions must be "
    "physically plausible and internally consistent with the request.\n\n"
    "Return only the JSON object wrapped in a ```json code fence.\n$image_note\n$nl_description\n\n"
    "All top-level keys must always be present in the output; fill unused fields with \"\", 0, {}, or [] as "
    "appropriate.\n\n$json_template\n\nresolution_ratio_dict = $resolution_ratio_dict")

_I2V_IMAGE_NOTE = (
    "\nIMPORTANT - IMAGE INPUT: The attached image is the first frame of the video. Use it as visual ground "
    "truth for subject appearance, setting, lighting, and colors. The natural-language request primarily "
    "describes temporal/action intent. Your JSON must be consistent with what is visible in the image.\n")


def size_to_res_aspect(size: str) -> tuple[str, str]:
    """Map a 'WxH' size to the (resolution, aspect_ratio) bucket the upsampler expects."""
    try:
        w, h = (int(x) for x in str(size).lower().split("x"))
    except (ValueError, AttributeError):
        return "720", "16,9"
    for res, ar_dict in RESOLUTION_RATIO_DICT.items():
        for ar, wh in ar_dict.items():
            if wh["W"] == w and wh["H"] == h:
                return res, ar
    ratio = w / max(1, h)
    best = min(RESOLUTION_RATIO_DICT["720"].items(), key=lambda kv: abs(kv[1]["W"] / kv[1]["H"] - ratio))
    return "720", best[0]


def _duration_label(num_frames: int, fps: int) -> str:
    return f"{int(num_frames / max(1, fps))}s"


def _nl(prompt: str, resolution: str, aspect: str, duration: str | None = None, fps: int | None = None) -> str:
    params = [f"resolution {resolution}", f"aspect_ratio {aspect}"]
    if duration is not None:
        params.append(f"duration {duration}")
    if fps is not None:
        params.append(f"fps {fps}")
    return f"{prompt.strip()}\n\nOutput parameters: {', '.join(params)}."


def is_upsampled_prompt(prompt: str) -> bool:
    """True if the prompt already looks like upsampler output (so we skip re-upsampling)."""
    s = (prompt or "").strip()
    if not s:
        return False
    if s.startswith("```json") or s.startswith("```\n{"):
        return True
    if s.startswith("{"):
        try:
            return isinstance(json.loads(s), dict)
        except (json.JSONDecodeError, ValueError):
            return False
    return False


def build_payload(mode: str, prompt: str, size: str, fps: int, num_frames: int,
                  model: str | None, image_data_url: str | None = None) -> dict[str, Any]:
    """Build the OpenAI-compatible chat payload for the given mode (text2image/text2video/image2video)."""
    res, aspect = size_to_res_aspect(size)
    res_text = json.dumps(RESOLUTION_RATIO_DICT, indent=2)
    if mode == "text2image":
        text = _T2I_TEMPLATE.substitute(json_template=_T2I_SCHEMA, nl_description=_nl(prompt, res, aspect),
                                        resolution_ratio_dict=res_text)
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    else:
        dur = _duration_label(num_frames, fps)
        img = mode == "image2video"
        intro = ("Given the attached starting frame image and the user's natural-language request below"
                 if img else "Given the user's natural-language request below")
        text = _T2V_TEMPLATE.substitute(
            image_note=_I2V_IMAGE_NOTE if img else "", intro=intro, json_template=_T2V_SCHEMA,
            nl_description=_nl(prompt, res, aspect, dur, fps), resolution_ratio_dict=res_text)
        if img and image_data_url:
            content = [{"type": "image_url", "image_url": {"url": image_data_url}}, {"type": "text", "text": text}]
        else:
            content = [{"type": "text", "text": text}]
    # Deterministic upsampling (greedy + fixed seed): the framework default (temp 0.7) re-invents a
    # different dense caption every call, which made the same prompt produce a completely different
    # image each time. Greedy keeps the same prompt -> same caption -> reproducible generation.
    payload: dict[str, Any] = {"messages": [SYSTEM_MESSAGE, {"role": "user", "content": content}],
                               "max_tokens": 8192, "temperature": 0.0, "seed": 0}
    if model:
        payload["model"] = model
    return payload


def parse_upsampled(content: str, mode: str, size: str, fps: int, num_frames: int) -> str:
    """Extract the JSON object from the model response and pin the output parameters.
    Returns a compact JSON string (the generator's prompt). Raises if no JSON is found."""
    cleaned = (content or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("upsampler response JSON must be an object")
    res, aspect = size_to_res_aspect(size)
    wh = RESOLUTION_RATIO_DICT[res][aspect]
    data["resolution"] = {"H": wh["H"], "W": wh["W"]}
    data["aspect_ratio"] = aspect
    if mode != "text2image":
        data["duration"] = _duration_label(num_frames, fps)
        data["fps"] = int(fps)
    return json.dumps(data, ensure_ascii=False)
