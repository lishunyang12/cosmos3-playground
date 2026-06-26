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
import math
import mimetypes
import os
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).parent / "examples"

# Sampling defaults follow the Cosmos 3 technical report, Table 21 ("Default sampling
# configurations ... for each generator and generation modality"). Base generator
# audio-visual: steps=50, guidance=6, shift=10, full-range CFG.
_VIDEO_DEFAULTS = {
    "size": "1280x720", "num_frames": 93, "fps": 24,
    "num_inference_steps": 50, "guidance_scale": 6.0, "flow_shift": 10.0,
}
# Image is the audio-visual model's still-frame mode → same guidance=6 as Table 21.
_IMAGE_DEFAULTS = {"size": "1024x1024", "num_inference_steps": 50, "guidance_scale": 6.0}
# Forward/inverse dynamics (Table 21): steps=50, guidance=1, shift=5, full-range CFG,
# null negative prompt. Action envelope: 10-30 FPS, 16-400 frame horizon (§6.3.1).
_ACTION_DEFAULTS = {"size": "832x480", "fps": 10, "num_inference_steps": 50, "guidance_scale": 1.0, "flow_shift": 5.0}

# Action-mode prompt formatting — mirror cosmos-framework's ActionPromptJsonFormatter +
# action.py so the checkpoints see their trained input distribution: the IMAGE system prompt
# (is_video=False for action modes) plus a structured JSON caption with the per-domain
# viewpoint framing. Verified to give a more stable, on-distribution policy rollout than raw
# free text. The system-prompt string is verbatim from the framework (trained typo "give").
_ACTION_SYSTEM_PROMPT = "You are a helpful assistant who will generate images from a give prompt."
_VIEWPOINT_TEMPLATES = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}
# Canonical viewpoint per embodiment domain (from the cosmos-framework action datasets).
_DOMAIN_VIEWPOINT = {"droid_lerobot": "concat_view", "av": "ego_view",
                     "agibotworld": "concat_view", "bridge_orig_lerobot": "ego_view"}
# Domain-specific framing detail appended after the viewpoint template, verbatim from the dataset.
_DOMAIN_VIEW_DESC = {
    "droid_lerobot": ("The top row is from the wrist-mounted camera. The bottom row contains two horizontally "
                      "concatenated third-person perspective views of the scene from opposite sides, with the robot visible."),
}

# Default negative prompt for video generation, verbatim from the report's Appendix B.3
# (the natural-language "DEFAULT NEGATIVE PROMPT"). Table 21 uses the null string for
# Text2Image and for all action/dynamics modes, so only the video examples carry this.
_NEG_VIDEO = (
    "The video captures a series of frames showing macroblocking artifacts, chromatic "
    "aberration, high-frequency noise, and rolling shutter distortion. It includes static "
    "with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy "
    "texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor "
    "color balance, washed out colors, choppy sequences, jerky movements, low frame rate, "
    "bit-depth compression artifacts, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, hard cut, "
    "visual noise, and flickering. It features moire patterns, edge halos, and temporal "
    "aliasing. Furthermore, the content defies common sense, generating illogical scenarios, "
    "nonsensical entities, absurd character behaviors, and conceptual paradoxes that violate "
    "basic human reasoning and everyday reality. The video looks like a surreal or glitchy "
    "hallucination. Overall, the video is of poor quality."
)

# Forward-dynamics example metadata — drives the Rollout control and the derived frame count.
# Duration is bound to the action trajectory (1 frame per action step), so the user picks how
# many chunks to roll out rather than a raw frame count.
_FD_SPEC = json.loads((EXAMPLES_DIR / "fd_action_chunks.json").read_text())
_FD_CHUNK = int(_FD_SPEC.get("action_chunk_size", 16))
_FD_NCHUNKS = int(_FD_SPEC.get("num_chunks", len(_FD_SPEC.get("action_chunks", [[]]))))

# ----------------------------------------------------------------------------- modes
MODES: list[dict[str, Any]] = [
    # ---- GENERATE ----
    {"id": "t2i", "label": "Text → Image", "surface": "generate", "group": "World Model", "primary": True,
     "kind": "image", "reference": "none", "blurb": "Generate a still image from a prompt.",
     "io": "Prompt → image", "key_knobs": ["size", "guidance_scale"],
     "example": {"prompt": "Photorealistic close-up of a brushed-titanium robotic hand with exposed servos "
                 "gently cradling a fresh dewy strawberry, soft window light, razor-sharp focus on the fruit's "
                 "seeds and the metal's micro-scratches, shallow depth of field, studio product photography",
                 "params": _IMAGE_DEFAULTS, "reference": None}},
    {"id": "t2v", "label": "Text → Video", "surface": "generate", "group": "World Model", "primary": True,
     "kind": "video", "reference": "none", "blurb": "Imagine a video world from a prompt.",
     "io": "Prompt → video (with optional sound)", "key_knobs": ["size", "num_frames", "generate_sound"],
     "example": {"prompt": "A lone surfer drops into a towering turquoise wave at golden hour; the lip throws "
                 "over into a glassy barrel, offshore wind feathering spray off the crest, water rushing past "
                 "with physically accurate fluid dynamics, cinematic tracking shot, photorealistic. Audio "
                 "description: the deep roar of the breaking wave, the hiss of wind-blown spray, the board "
                 "carving through water.",
                 # sound defaults OFF: under a sequence-parallel (ulysses) deployment the
                 # video+sound token count must be a multiple of ulysses_degree, which fails for
                 # many frame counts. Video-only is always safe; sound is an opt-in toggle.
                 # No negative_prompt → backend applies the paper B.6 structured negative prompt by default.
                 "params": {**_VIDEO_DEFAULTS, "generate_sound": False},
                 "reference": None}},
    {"id": "i2v", "label": "Image → Video", "surface": "generate", "group": "World Model", "primary": True,
     "kind": "video", "reference": "image", "blurb": "Animate a still image into a video.",
     "io": "Image + prompt → video", "key_knobs": ["num_frames"],
     "example": {"prompt": "Bring the waterfall to life: water cascades over the rock face and splashes into the "
                 "pool, the stream ripples over the mossy stones, ferns sway gently, fine mist drifts — natural, "
                 "physically consistent motion, photorealistic.",
                 "params": {**_VIDEO_DEFAULTS}, "reference": "i2v_input.jpg"}},
    {"id": "v2v", "label": "Video → Video", "surface": "generate", "group": "World Model",
     "kind": "video", "reference": "video", "blurb": "Future prediction: keep the opening frames, generate what happens next.",
     "io": "Video (opening frames) + prompt → continuation", "key_knobs": ["num_frames"],
     "purpose": "Predict the future of a clip — the model locks your opening frames as ground truth and "
                "generates what comes next from the prompt. It's continuation, NOT a restyle. "
                "(For “same structure, new look”, use Transfer.)",
     "flow": {"inputs": ["opening frames", "prompt"], "output": "continued video"},
     "notes": [["locks", "the opening frames (shown unchanged)"],
               ["generates", "the continuation from your prompt"]],
     "extra": [{"key": "condition_video_keep", "label": "Mode", "type": "select", "widget": "segmented",
                "options": ["first", "last"], "default": "first",
                "optionLabels": {"first": "predict from start", "last": "extend past end"},
                "hint": "Conditions on only ~5 frames (2 latent) — a short clip is enough; no need for a long video."}],
     "example": {"prompt": "A robotic arm continues its manipulation task over the white plate, the gripper moving "
                 "smoothly along a deliberate trajectory to handle the food and then retracting. Physically "
                 "consistent motion, the scene and tableware stay stable, bright soft lighting, photorealistic.",
                 "params": {**_VIDEO_DEFAULTS, "size": "1280x720", "num_frames": 121, "fps": 24,
                            "condition_video_keep": "first"},
                 "reference": "ref_video.mp4"}},
    {"id": "transfer", "label": "Transfer · Sim→Real", "surface": "generate", "group": "Sim2Real (SDG)",
     "kind": "video", "reference": "video", "blurb": "Sim-to-real: turn a simulated / CG clip into a photorealistic video, keeping exact geometry & motion.",
     "io": "Sim (or control) clip + prompt → photorealistic video", "key_knobs": [],
     # length + aspect ratio are derived from the control clip automatically — not user knobs;
     # transfer has no audio, so hide the sound toggle too.
     "hide_knobs": ["num_frames", "fps", "size", "generate_sound"],
     "purpose": "Take a low-fidelity simulation/render and make it photorealistic — same layout, objects and "
                "motion, but real materials, textures and lighting. The core Physical-AI use: generate labeled, "
                "photorealistic training data from sim (the geometry stays ground-truth-correct).",
     "flow": {"inputs": ["sim/control clip", "prompt"], "output": "photorealistic video"},
     "notes": [["keeps", "geometry · layout · motion (ground-truth)"],
               ["changes", "materials · textures · lighting → photoreal"]],
     # Only edge/blur can be derived on-the-fly from an RGB clip; depth/seg/wsm need a
     # precomputed control-map video (control_path), which this playground doesn't generate.
     "control_defaults": {"edge": 1.5, "blur": 1.5},
     "extra": [{"key": "control", "label": "Control type", "type": "select", "widget": "segmented",
                "options": ["edge", "blur"], "default": "edge"},
               {"key": "control_guidance", "label": "Control strength", "type": "number", "widget": "slider",
                "default": 1.5, "min": 0.0, "max": 3.0, "step": 0.05}],
     # Prompt is video-matched (generated by the reasoner from transfer_sim_robot.mp4): describing the
     # actual scene keeps the model from fighting the control input, which removes grain. Paired with the
     # reference transfer CFG (guidance 3, not 5) — the dominant grain reducer for structure-locked transfer.
     "example": {"prompt": "In a meticulously rendered kitchen with a white marble countertop and dark wood "
                 "flooring, a robotic hand with black-gloved fingers gently grasps a ripe, glossy red tomato "
                 "resting on a checkered wooden cutting board, while the other hand hovers near an empty beige "
                 "frying pan with dual handles, positioned centrally on the counter. To the left, a vibrant orange "
                 "plate sits beside a colorful cereal box, and a potted aloe vera plant with long, spiky green "
                 "leaves stands beside a sleek silver toaster, all under soft, diffused overhead lighting that "
                 "casts subtle shadows and highlights the textures of the countertop, the tomato's skin, and the "
                 "wood grain of the cutting board. The robot's movements are precise and deliberate, as if "
                 "preparing to slice the tomato with a knife that's not yet visible, maintaining the exact spatial "
                 "arrangement and camera angle of the original simulation.",
                 "params": {**_VIDEO_DEFAULTS, "size": "1280x720", "control": "edge", "control_guidance": 1.5,
                            "guidance_scale": 3.0},
                 "reference": "transfer_sim_robot.mp4"}},
    {"id": "fwd_dynamics", "label": "Forward dynamics", "surface": "generate", "group": "Robotics", "action": True,
     "kind": "video", "reference": "image", "blurb": "Action-conditioned future prediction: roll out a video "
     "from a first frame + an action trajectory.", "chunk_size": _FD_CHUNK,
     "io": "First frame + action trajectory → video", "key_knobs": [],
     "purpose": "Simulate the future — roll out a video from a first frame plus an action plan. "
                "“press play on the physics.”",
     "flow": {"inputs": ["first frame", "actions ▸▸"], "output": "predicted video"},
     "notes": [["obeys", "physics & contact"],
               ["driven by", "your action plan — pick how far to roll out"]],
     # Duration = how much of the action trajectory you roll out (1 frame per action step). The Rollout
     # control picks how many chunks to feed; num_frames is derived from it (chunk_size · n + 1).
     "extra": [{"key": "rollout_chunks", "label": f"Rollout (×{_FD_CHUNK}-step chunks)", "type": "int",
                "widget": "slider", "min": 1, "max": _FD_NCHUNKS, "step": 1, "default": _FD_NCHUNKS, "unit": "chunks"}],
     "example": {"prompt": "Predict the future frames produced by executing the given action trajectory, with "
                 "physically consistent contact and object motion.",
                 "params": {**_ACTION_DEFAULTS, "size": "960x960", "rollout_chunks": 4},
                 "reference": "fd_first_frame.png", "action": "fd_action_chunks.json"}},
    {"id": "inv_dynamics", "label": "Inverse dynamics", "surface": "generate", "group": "Autonomous Driving", "action": True,
     "kind": "video", "reference": "video", "blurb": "Recover the ego-motion / action trajectory from a video.",
     "io": "Video → action trajectory (numbers, not a clip)", "key_knobs": [],
     # The model's real output here is the action tensor, not a clip — tell the UI to surface the
     # trajectory (numbers + plot) instead of the echoed reconstruction video.
     "primary_output": "action",
     "example": {"prompt": "Recover the per-frame ego-motion behind this race-car onboard clip — the camera's "
                 "translation and rotation through the scene, frame by frame. The output is a 32×9 action "
                 "trajectory, not a video.",
                 "params": {**_ACTION_DEFAULTS, "num_frames": 33, "action_chunk_size": 32}, "reference": "race_pov.mp4"}},
    {"id": "policy", "label": "Policy", "surface": "generate", "group": "Autonomous Driving", "action": True,
     "kind": "video", "reference": "image", "blurb": "Planning policy: from a single first frame + a role "
     "instruction the model predicts its own action trajectory and rolls out the future.",
     "io": "First frame + instruction → predicted actions + video",
     "purpose": "Give it a role, not a script — the model decides the actions itself and rolls out the "
                "future. Here it acts as an autonomous-vehicle planner: from one front-camera frame it "
                "predicts a 60-step driving trajectory and the resulting ~6s of forward driving.",
     "flow": {"inputs": ["first frame", "instruction"], "output": "predicted driving + actions"},
     "notes": [["model decides", "the 60-step action trajectory"],
               ["rolls out", "the predicted future drive (single shot)"]],
     # Official AV policy example (cosmos-framework inputs/omni/action_policy_av.json): av domain,
     # ego_view front camera, 9-D action, image_size 480 (832x480), 10 fps, 60-step chunk. Runs on the
     # base generator (NOT the dedicated DROID checkpoint). First frame is the official 832x480 driving clip.
     "example": {"prompt": "You are an autonomous vehicle planning system.",
                 "params": {"size": "832x480", "fps": 10, "num_inference_steps": 50, "guidance_scale": 1.0,
                            "flow_shift": 5.0, "domain_name": "av", "raw_action_dim": 9},
                 "reference": "policy_av_first_frame.png", "action": "policy"}},
    # ---- REASON ----
    {"id": "caption", "label": "Captioning", "surface": "reason", "group": "Reason", "primary": True,
     "kind": "text", "reference": "image", "blurb": "Detailed description of an image or video.",
     "io": "Image / video → description", "key_knobs": ["max_tokens"],
     "example": {"prompt": "Describe this scene in vivid, concrete detail — the objects and their materials, "
                 "the spatial layout, the lighting, and any action taking place.",
                 "params": {}, "reference": "reason_image.png"}},
    {"id": "temporal", "label": "Temporal localization", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "video", "blurb": "When does an event happen? (timestamps)",
     "io": "Video → event timestamps", "key_knobs": ["max_tokens"],
     "example": {"prompt": "Identify the key events in this video and report each one's start and end timestamps.",
                 "params": {}, "reference": "id_av_input.mp4"}},
    {"id": "grounding", "label": "2D grounding", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "image", "blurb": "Locate an object and return its 2D coordinates.",
     "io": "Image → object bounding box", "key_knobs": ["max_tokens"],
     "example": {"prompt": "Locate the dog in the image and return its 2D bounding box as [x1, y1, x2, y2].",
                 "params": {}, "reference": "ground_input.jpg"}},
    {"id": "physical", "label": "Physical reasoning", "surface": "reason", "group": "Reason",
     "kind": "text", "reference": "video", "blurb": "Is what happens physically plausible?",
     "io": "Video → plausibility analysis", "key_knobs": ["max_tokens"],
     "example": {"prompt": "Examine the physics of this video. Is every motion physically plausible? Flag any "
                 "violation of gravity, momentum, or contact dynamics and explain your reasoning step by step.",
                 "params": {}, "reference": "id_av_input.mp4"}},
    {"id": "vqa", "label": "Ask anything", "surface": "reason", "group": "Reason", "primary": True,
     "kind": "text", "reference": "image", "blurb": "Free-form question about an image or video.",
     "io": "Image / video → answer", "key_knobs": ["max_tokens"],
     "example": {"prompt": "What's on the plate, and what meal is this? List the food items and say whether it "
                 "looks like a balanced meal.", "params": {}, "reference": "vqa_input.jpg"}},
]

GEN_KNOBS: list[dict[str, Any]] = [
    {"key": "negative_prompt", "label": "Negative prompt", "type": "text", "default": ""},
    {"key": "size", "label": "Resolution", "type": "select", "widget": "segmented",
     "options": ["1280x720", "720x1280", "960x720", "720x960", "1024x1024", "832x480", "480x832"],
     "default": "1280x720"},
    {"key": "num_frames", "label": "Length", "type": "int", "widget": "slider", "unit": "frames",
     "default": 93, "min": 1, "max": 257, "step": 4, "video": True},
    {"key": "fps", "label": "FPS", "type": "int", "widget": "slider", "default": 24, "min": 10, "max": 30, "video": True},
    {"key": "num_inference_steps", "label": "Steps", "type": "int", "widget": "slider", "default": 50, "min": 1, "max": 100},
    {"key": "guidance_scale", "label": "Guidance", "type": "number", "widget": "slider",
     "default": 6.0, "min": 0.0, "max": 20.0, "step": 0.5},
    {"key": "flow_shift", "label": "Flow shift", "type": "number", "widget": "slider",
     "default": 10.0, "min": 0.0, "max": 20.0, "step": 0.5, "video": True},
    {"key": "seed", "label": "Seed", "type": "int", "default": None, "min": 0, "max": 2**31 - 1},
    {"key": "generate_sound", "label": "Generate sound", "type": "bool", "default": False, "video": True},
]

REASON_KNOBS: list[dict[str, Any]] = [
    {"key": "max_tokens", "label": "Answer length", "type": "int", "widget": "slider",
     "default": 512, "min": 16, "max": 4096, "step": 16, "unit": "tokens"},
    {"key": "temperature", "label": "Temperature", "type": "number", "widget": "slider",
     "default": 0.2, "min": 0.0, "max": 1.5, "step": 0.1},
]

_MODE_BY_ID = {m["id"]: m for m in MODES}

# Action embodiment domains. ``domain_id`` mirrors vllm_omni cosmos3 action.py
# (EMBODIMENT_TO_DOMAIN_ID); the rest is metadata the playground surfaces so an
# explainability user reads the action *context*, not just opaque numbers. dims/fps
# are filled only where we can verify them (av, agibotworld); null = not yet sourced.
ACTION_DOMAINS: dict[str, dict[str, Any]] = {
    "agibotworld": {"domain_id": 15, "kind": "bimanual robot manipulation",
                    "raw_action_dim": 29, "fps": 10, "viewpoint": "wrist + two third-person views"},
    "av": {"domain_id": 1, "kind": "ego-vehicle motion",
           "raw_action_dim": 9, "fps": 10, "viewpoint": "front-facing camera"},
    "droid_lerobot": {"domain_id": 8, "kind": "single-arm manipulation",
                      "raw_action_dim": None, "fps": 15, "viewpoint": "third-person"},
    "libero": {"domain_id": 5, "kind": "tabletop manipulation",
               "raw_action_dim": None, "fps": None, "viewpoint": "third-person"},
}


def action_domain(name_or_id: Any) -> dict[str, Any] | None:
    """Look up a domain by name or numeric id; returns metadata + name, or None."""
    if name_or_id is None:
        return None
    if isinstance(name_or_id, str) and name_or_id in ACTION_DOMAINS:
        return {"name": name_or_id, **ACTION_DOMAINS[name_or_id]}
    for name, info in ACTION_DOMAINS.items():
        if info["domain_id"] == name_or_id:
            return {"name": name, **info}
    return None


def catalog() -> dict[str, Any]:
    return {"modes": MODES, "gen_knobs": GEN_KNOBS, "reason_knobs": REASON_KNOBS,
            "action_domains": ACTION_DOMAINS}


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
        # honor the explicit size / frame-count / fps instead of letting the server rewrite the
        # prompt from its resolution/duration templates (matches the official cookbook request).
        extra_params["use_resolution_template"] = False
        extra_params["use_duration_template"] = False
        for key, cast in (("num_frames", int), ("fps", int), ("flow_shift", float)):
            val = _num(params, key, cast)
            if val is not None:
                fields[key] = val
        if params.get("generate_sound"):
            fields["generate_sound"] = True
            nf, fps = _num(params, "num_frames", int), _num(params, "fps", int)
            if nf and fps:
                fields["sound_duration"] = _sound_duration_for_sp(params.get("size"), nf, nf / fps)

    if mode_id == "v2v":
        extra_params["condition_video_keep"] = params.get("condition_video_keep", "first")
        extra_params["condition_frame_indexes_vision"] = [0, 1]
    elif mode_id == "transfer":
        extra_params[params.get("control", "edge")] = True
        cg = _num(params, "control_guidance", float)
        if cg is not None:
            extra_params["control_guidance"] = cg
        # transfer requires a supported resolution bucket; derive it from the requested
        # size's short side, otherwise the server falls back to an unsupported value.
        buckets = (256, 480, 704, 720)
        try:
            w, h = (int(x) for x in (params.get("size") or "1280x720").lower().split("x"))
            short = min(w, h)
        except (ValueError, AttributeError):
            short = 720
        extra_params["resolution"] = min(buckets, key=lambda b: abs(b - short))
    elif mode_id == "fwd_dynamics":
        action = fd_action(params)
        extra_params.update({"action_mode": "forward_dynamics",
                             "domain_name": _FD_SPEC.get("domain_name", "agibotworld"),
                             "action_chunk_size": len(action), "action": action})
        fields["num_frames"] = len(action) + 1  # one frame per action step (+ the first frame)
    elif mode_id == "inv_dynamics":
        # Defaults describe the standalone av example; a round-trip overrides domain/dims
        # so inverse dynamics runs in the SAME domain as the forward-dynamics plan it checks.
        chunk = int(params.get("action_chunk_size") or 60)
        extra_params.update({"action_mode": "inverse_dynamics",
                             "domain_name": params.get("domain_name") or "av",
                             "action_chunk_size": chunk,
                             "raw_action_dim": int(params.get("raw_action_dim") or 9)})
        # The pipeline requires num_frames == chunk or chunk+1; ignore any leaked UI
        # value (e.g. the hidden video knob's 93) that falls outside that range.
        nf = _num(params, "num_frames", int)
        fields["num_frames"] = nf if nf in (chunk, chunk + 1) else (chunk + 1)
    elif mode_id == "policy":
        # Single-shot planning policy (official AV example): the model predicts its own
        # action trajectory + rolls out the future drive. Runs on the base generator.
        chunk = int(params.get("action_chunk_size") or 60)
        extra_params.update({"action_mode": "policy",
                             "domain_name": params.get("domain_name") or "av",
                             "action_chunk_size": chunk,
                             "raw_action_dim": int(params.get("raw_action_dim") or 9)})
        # num_frames is fixed by the prediction horizon (first frame + chunk steps); the
        # pipeline requires it to equal chunk or chunk+1, so ignore any leaked UI value.
        fields["num_frames"] = chunk + 1

    # action modes read the predicted/echoed action back from the async job, like the cookbook.
    req = {"kind": m["kind"], "reference": m["reference"], "fields": fields, "extra_params": extra_params,
           "async_action": mode_id in ("fwd_dynamics", "inv_dynamics", "policy")}
    return _apply_action_prompt_format(req)


def fd_action(params: dict[str, Any]) -> list[list[float]]:
    """The flat [T, D] action plan that forward dynamics rolls out for the given settings."""
    all_chunks = _FD_SPEC["action_chunks"]
    n = max(1, min(int(params.get("rollout_chunks") or 1), len(all_chunks)))
    return [step for ch in all_chunks[:n] for step in ch]  # concatenate n chunks of actions


def fd_chunk_count() -> int:
    return len(_FD_SPEC.get("action_chunks", []))


def _action_framing(domain: str) -> str | None:
    """Resolve the cinematography framing text for a domain (viewpoint template +
    optional domain-specific detail), mirroring ActionPromptJsonFormatter._get_viewpoint_caption."""
    tmpl = _VIEWPOINT_TEMPLATES.get(_DOMAIN_VIEWPOINT.get(domain, ""))
    desc = _DOMAIN_VIEW_DESC.get(domain)
    if tmpl is None:
        return desc
    if desc:
        return tmpl + (" " if tmpl.endswith(".") else ". ") + desc
    return tmpl


def _action_json_caption(prompt: str, domain: str, fps: int, width: int, height: int, num_frames: int) -> str:
    """Build the structured action JSON caption the action checkpoints were trained on."""
    secs = (num_frames / fps) if fps else 0.0
    mm, ss = divmod(int(round(secs)), 60)
    desc = prompt if prompt[-1:] in ".!?" else prompt + "."
    cap: dict[str, Any] = {}
    framing = _action_framing(domain)
    if framing:
        cap["cinematography"] = {"framing": framing}
    cap["actions"] = [{"time": f"0:00-{mm}:{ss:02d}", "description": desc}]
    cap["duration"] = f"{int(secs)}s"
    cap["fps"] = float(fps)
    cap["resolution"] = {"H": int(height), "W": int(width)}
    g = math.gcd(int(width), int(height)) or 1
    cap["aspect_ratio"] = f"{int(width) // g},{int(height) // g}"
    return json.dumps(cap)


def _apply_action_prompt_format(req: dict[str, Any]) -> dict[str, Any]:
    """In-place: make an action request follow the paper's inference format — IMAGE system
    prompt + structured JSON caption. No-op for non-action requests."""
    ep = req.get("extra_params") or {}
    if not ep.get("action_mode"):
        return req
    f = req["fields"]
    raw = (f.get("prompt") or "").strip()
    if raw and raw != ".":
        try:
            w, h = (int(x) for x in str(f.get("size") or _ACTION_DEFAULTS["size"]).lower().split("x"))
        except (ValueError, AttributeError):
            w, h = 640, 480
        f["prompt"] = _action_json_caption(
            raw, ep.get("domain_name", ""), int(f.get("fps") or 10), w, h, int(f.get("num_frames") or 1)
        )
    # Action modes are tokenized with is_video=False → the IMAGE system prompt.
    ep["use_system_prompt"] = True
    ep["system_prompt"] = _ACTION_SYSTEM_PROMPT
    req["extra_params"] = ep
    return req


def fd_single_chunk_request(params: dict[str, Any], idx: int) -> dict[str, Any]:
    """A forward-dynamics request for ONE action chunk — the unit of an autoregressive
    rollout (each chunk is conditioned on the previous chunk's last frame)."""
    chunk = _FD_SPEC["action_chunks"][idx]
    prompt = (params.get("prompt") or mode("fwd_dynamics").get("example", {}).get("prompt") or ".").strip()
    fields = {
        "prompt": prompt or ".",
        "size": params.get("size") or _ACTION_DEFAULTS["size"],
        "fps": int(_num(params, "fps", int) or _FD_SPEC.get("fps") or _ACTION_DEFAULTS["fps"]),
        "num_inference_steps": int(_num(params, "num_inference_steps", int) or _ACTION_DEFAULTS["num_inference_steps"]),
        "guidance_scale": float(_num(params, "guidance_scale", float) or _ACTION_DEFAULTS["guidance_scale"]),
        "flow_shift": float(_num(params, "flow_shift", float) or _ACTION_DEFAULTS["flow_shift"]),
        "num_frames": len(chunk) + 1,
    }
    extra = {"use_resolution_template": False, "use_duration_template": False,
             "action_mode": "forward_dynamics", "domain_name": _FD_SPEC.get("domain_name", "agibotworld"),
             "action_chunk_size": len(chunk), "action": chunk}
    return _apply_action_prompt_format(
        {"kind": "video", "reference": "image", "fields": fields, "extra_params": extra, "async_action": True})


def policy_single_chunk_request(params: dict[str, Any]) -> dict[str, Any]:
    """One policy chunk: the model PREDICTS a 16-step action from the first frame + the
    instruction and rolls out 17 frames. Chained autoregressively for a long video."""
    chunk = _FD_CHUNK
    prompt = (params.get("prompt") or mode("policy").get("example", {}).get("prompt") or ".").strip()
    fields = {
        "prompt": prompt or ".",
        "size": params.get("size") or _ACTION_DEFAULTS["size"],
        "fps": int(_num(params, "fps", int) or _FD_SPEC.get("fps") or _ACTION_DEFAULTS["fps"]),
        "num_inference_steps": int(_num(params, "num_inference_steps", int) or _ACTION_DEFAULTS["num_inference_steps"]),
        "guidance_scale": float(_num(params, "guidance_scale", float) or _ACTION_DEFAULTS["guidance_scale"]),
        "flow_shift": float(_num(params, "flow_shift", float) or _ACTION_DEFAULTS["flow_shift"]),
        "num_frames": chunk + 1,
    }
    extra = {"use_resolution_template": False, "use_duration_template": False,
             "action_mode": "policy", "domain_name": params.get("domain_name") or "droid_lerobot",
             "raw_action_dim": int(params.get("raw_action_dim") or 8), "action_chunk_size": chunk}
    return _apply_action_prompt_format(
        {"kind": "video", "reference": "image", "fields": fields, "extra_params": extra, "async_action": True})


def roundtrip_id_request(mode_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build the inverse-dynamics request that re-reads the action out of a forward-dynamics
    video, in the SAME domain as the forward plan — the recovered action is compared to the
    original to score round-trip consistency."""
    if mode_id != "fwd_dynamics":
        raise ValueError(f"round-trip validation is only defined for forward dynamics, not '{mode_id}'")
    action = fd_action(params)
    chunk, dim = len(action), (len(action[0]) if action else 0)
    id_params = {
        "prompt": (params.get("prompt") or mode("inv_dynamics").get("example", {}).get("prompt") or "."),
        "size": params.get("size") or _ACTION_DEFAULTS["size"],
        "fps": params.get("fps") or _FD_SPEC.get("fps") or _ACTION_DEFAULTS["fps"],
        "num_inference_steps": params.get("num_inference_steps") or _ACTION_DEFAULTS["num_inference_steps"],
        "guidance_scale": params.get("guidance_scale") or _ACTION_DEFAULTS["guidance_scale"],
        "flow_shift": params.get("flow_shift") or _ACTION_DEFAULTS["flow_shift"],
        "domain_name": _FD_SPEC.get("domain_name", "agibotworld"),
        "action_chunk_size": chunk, "raw_action_dim": dim, "num_frames": chunk + 1,
    }
    return build_request("inv_dynamics", id_params)


def compare_actions(original: Any, recovered: Any) -> dict[str, Any]:
    """Score how well a recovered [T, D] action matches the original plan. Pure-Python so it
    has no numpy dependency. Headline is cosine similarity (stable for short horizons where
    per-channel range can collapse to ~0 and blow up a normalized RMSE)."""
    def as2d(a: Any) -> list[list[float]]:
        while isinstance(a, list) and len(a) == 1 and a and isinstance(a[0], list) and a[0] \
                and isinstance(a[0][0], list):
            a = a[0]  # squeeze leading batch dims
        return [[float(x) for x in row] for row in a]

    A, B = as2d(original), as2d(recovered)
    t = min(len(A), len(B))
    if t == 0 or not A[0]:
        return {"ok": False, "reason": "empty action"}
    d = min(len(A[0]), len(B[0]))
    A = [row[:d] for row in A[:t]]
    B = [row[:d] for row in B[:t]]

    abs_err = [[abs(A[i][j] - B[i][j]) for j in range(d)] for i in range(t)]
    mae = sum(sum(r) for r in abs_err) / (t * d)
    per_channel_mae = [sum(abs_err[i][j] for i in range(t)) / t for j in range(d)]

    dot = sum(A[i][j] * B[i][j] for i in range(t) for j in range(d))
    na = math.sqrt(sum(A[i][j] ** 2 for i in range(t) for j in range(d)))
    nb = math.sqrt(sum(B[i][j] ** 2 for i in range(t) for j in range(d)))
    cosine = dot / (na * nb) if na > 0 and nb > 0 else 0.0
    return {
        "ok": True, "shape": [t, d],
        "cosine": round(cosine, 4),
        "mae": round(mae, 4),
        "consistency_pct": round(max(0.0, cosine) * 100, 1),
        "per_channel_mae": [round(v, 4) for v in per_channel_mae],
    }


def to_multipart_fields(req: dict[str, Any], model_name: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in req["fields"].items():
        out[k] = v if isinstance(v, str) else json.dumps(v) if isinstance(v, bool) else str(v)
    if req["extra_params"]:
        out["extra_params"] = json.dumps(req["extra_params"])
    if model_name:
        out["model"] = model_name
    return out


# --------------------------------------------------------------- pipeline transparency
def _summarize_request(req: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe view of a built request for the Pipeline panel — bulky action arrays
    are replaced by a shape summary so the request stays readable."""
    ep = dict(req.get("extra_params") or {})
    act = ep.get("action")
    if isinstance(act, list):
        dim = len(act[0]) if act and isinstance(act[0], list) else None
        ep["action"] = f"⟨{len(act)}×{dim} action array⟩" if dim else f"⟨{len(act)} action steps⟩"
    return {"kind": req["kind"], "reference": req["reference"],
            "async_action": req.get("async_action", False),
            "fields": dict(req["fields"]), "extra_params": ep}


def execution_stages(mode_id: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """The ordered stages a request runs through on the server, tagged with the
    parallel dimensions that act on each stage (``dims`` keys map to the
    deployment topology sizes the frontend reads from /api/config)."""
    m = mode(mode_id)
    if m["surface"] == "reason":
        return [{"name": "Encode", "detail": "media + question → tokens", "dims": ["tp"]},
                {"name": "LLM decode", "detail": "autoregressive answer", "dims": ["tp", "pp"]}]
    stages: list[dict[str, Any]] = [{"name": "Text encode", "detail": "prompt → conditioning (cached)", "dims": ["tp"]}]
    if mode_id == "i2v":
        stages.append({"name": "Image encode", "detail": "first frame → latent", "dims": []})
    elif mode_id in ("v2v", "transfer"):
        stages.append({"name": "Video encode", "detail": "reference → latents", "dims": []})
    elif mode_id == "inv_dynamics":
        stages.append({"name": "Video encode", "detail": "input clip → latents", "dims": []})
    steps = _num(params, "num_inference_steps", int) or 50
    stages.append({"name": "Denoise", "detail": f"{steps} steps · DiT", "dims": ["cfg", "ulysses", "ring"]})
    stages.append({"name": "VAE decode",
                   "detail": "latent → image" if m["kind"] == "image" else "latents → frames",
                   "dims": ["vae"]})
    if params.get("generate_sound"):
        stages.append({"name": "Sound decode", "detail": "latents → waveform", "dims": []})
    if mode_id == "inv_dynamics":
        stages.append({"name": "Action readout", "detail": "latents → [T×D] action", "dims": []})
    return stages


# Cosmos3 VAE/DiT factors (authoritative at runtime, mirrored here for offline shape
# computation): spatial downsample 16, temporal compression 4, DiT latent patch 2,
# audio latent rate 48000/1920 = 25 fps.
_VAE_SPATIAL, _VAE_TEMPORAL, _DIT_PATCH, _SOUND_LATENT_FPS = 16, 4, 2, 25.0


def _sound_duration_for_sp(size: Any, num_frames: int, duration: float) -> float:
    """Pick a sound duration whose latent length makes the packed video+sound sequence
    divisible by ulysses_degree, so video+sound runs under sequence parallelism without
    the server's divisibility error. Adds at most a few audio-latent frames (~40 ms each)
    beyond the requested duration; a no-op when SP is off. Mirrors the server-side fix."""
    try:
        uly = max(1, int(os.environ.get("COSMOS3_ULYSSES", "1") or 1))
    except ValueError:
        uly = 1
    sound_frames = max(1, math.ceil(duration * _SOUND_LATENT_FPS))
    if uly > 1:
        vtok = _latent_shape(size, num_frames)["tokens"]
        sound_frames += (-(vtok + sound_frames)) % uly
        duration = sound_frames / _SOUND_LATENT_FPS
    return round(duration, 4)


def _latent_shape(size: Any, num_frames: int) -> dict[str, Any]:
    try:
        w, h = (int(x) for x in str(size or "1280x720").lower().split("x"))
    except (ValueError, AttributeError):
        w, h = 1280, 720
    t_lat = (max(1, int(num_frames)) - 1) // _VAE_TEMPORAL + 1
    h_lat, w_lat = max(1, h // _VAE_SPATIAL), max(1, w // _VAE_SPATIAL)
    hp = -(-h_lat // _DIT_PATCH)
    wp = -(-w_lat // _DIT_PATCH)
    return {"px_h": h, "px_w": w, "t": t_lat, "h": h_lat, "w": w_lat, "tokens": t_lat * hp * wp}


def execution_graph(mode_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """The request as a TensorBoard-style computation graph: op-typed nodes grouped into
    namescopes, tensor shapes on the edges, and the classifier-free-guidance cond/uncond
    branches expanded so the parallel structure (CFG split, USP shard) is explicit."""
    m = mode(mode_id)
    if m["surface"] == "reason":
        return {"scopes": [{"id": "io", "label": "io"}, {"id": "vlm", "label": "Qwen3-VL"}],
                "nodes": [
                    {"id": "q", "label": "question", "op": "TextInput", "kind": "input", "scope": "io"},
                    {"id": "media", "label": "media", "op": "ImageInput", "kind": "input", "scope": "io"},
                    {"id": "enc", "label": "encode", "op": "vision_encode", "kind": "compute", "scope": "vlm", "dims": ["tp"]},
                    {"id": "llm", "label": "decode", "op": "generate", "kind": "compute", "scope": "vlm", "dims": ["tp", "pp"]},
                    {"id": "ans", "label": "answer", "op": "Text", "kind": "output", "scope": "io"},
                ],
                "edges": [{"from": "q", "to": "enc"}, {"from": "media", "to": "enc", "shape": "pixels"},
                          {"from": "enc", "to": "llm", "shape": "tokens"}, {"from": "llm", "to": "ans", "shape": "text"}]}

    steps = _num(params, "num_inference_steps", int) or 50
    guidance = _num(params, "guidance_scale", float) or 1.0
    lat = _latent_shape(params.get("size"), _num(params, "num_frames", int) or (1 if m["kind"] == "image" else 93))
    z = f"{lat['t']}×{lat['h']}×{lat['w']}"            # latent grid C×T×H×W (channels omitted)
    seq = f"{lat['tokens']}×d"                          # packed token sequence into the DiT
    px = f"{lat['px_h']}×{lat['px_w']}"

    L = "L"  # number of DiT blocks (symbolic)
    scopes = [{"id": "inputs", "label": "inputs"}, {"id": "cond", "label": "conditioning"},
              {"id": "denoise", "label": f"denoiser · {steps} steps"}, {"id": "decode", "label": "decode"},
              {"id": "out", "label": "outputs"}]
    nodes = [
        {"id": "prompt", "label": "prompt", "op": "TextInput", "kind": "input", "scope": "inputs"},
        {"id": "tok", "label": "tokenize", "op": "tokenizer", "kind": "compute", "scope": "cond"},
        {"id": "text", "label": "text encode", "op": "Qwen-LM (UND)", "kind": "compute", "scope": "cond", "dims": ["tp"]},
        {"id": "noise", "label": "noise", "op": "randn", "kind": "input", "scope": "inputs", "shape": z},
        {"id": "patch", "label": "patchify", "op": "patchify", "kind": "compute", "scope": "denoise", "shape": seq},
        {"id": "projin", "label": "proj_in", "op": "proj_in", "kind": "compute", "scope": "denoise", "shape": seq},
    ]
    edges = [
        {"from": "prompt", "to": "tok", "shape": "text"},
        {"from": "tok", "to": "text", "shape": "tokens"},
        {"from": "noise", "to": "patch", "shape": z},
        {"from": "patch", "to": "projin", "shape": seq},
    ]

    def _enc(node_id: str, in_id: str, in_label: str, into: list[str]) -> None:
        # reference pixels → VAE encode → conditioning latents joining the DiT input
        nodes.append({"id": in_id, "label": in_label, "op": "Input", "kind": "input", "scope": "inputs"})
        nodes.append({"id": node_id, "label": "VAE encode", "op": "vae.encode", "kind": "compute", "scope": "cond"})
        edges.append({"from": in_id, "to": node_id, "shape": "pixels"})
        for tgt in into:
            edges.append({"from": node_id, "to": tgt, "shape": "cond z"})

    # one DiT step expanded into blocks; classifier-free guidance splits cond/uncond lanes.
    cfg = guidance > 1.0
    lanes = ["dit_c", "dit_u"] if cfg else ["dit"]
    for lane in lanes:
        tag = "cond" if lane == "dit_c" else ("uncond" if lane == "dit_u" else "")
        nodes.append({"id": lane, "label": f"DiT block ×{L}" + (f" ({tag})" if tag else ""),
                      "op": "cross-attn + self-attn + MLP", "kind": "compute", "scope": "denoise",
                      "dims": ["ulysses", "ring"], "shape": seq})
        edges.append({"from": "projin", "to": lane, "shape": seq})
        edges.append({"from": "text", "to": lane, "shape": "K/V" if lane != "dit_u" else "K/V ∅"})
    if cfg:
        nodes.append({"id": "guide", "label": f"CFG combine ×{guidance:g}", "op": "apply_cfg",
                      "kind": "compute", "scope": "denoise", "dims": ["cfg"], "shape": seq})
        edges.append({"from": "dit_c", "to": "guide", "shape": "ε_c"})
        edges.append({"from": "dit_u", "to": "guide", "shape": "ε_u"})
    eps = "guide" if cfg else "dit"
    nodes.append({"id": "step", "label": f"scheduler.step ×{steps}", "op": "rectified-flow step",
                  "kind": "compute", "scope": "denoise", "shape": z})
    edges.append({"from": eps, "to": "step", "shape": "ε"})

    if mode_id == "i2v":
        _enc("imgenc", "img", "first frame", lanes)
    elif mode_id in ("v2v", "transfer"):
        _enc("viden", "vid", "reference video", lanes)
    elif mode_id == "inv_dynamics":
        _enc("viden", "vid", "input clip", lanes)
    elif mode_id == "fwd_dynamics":
        _enc("imgenc", "img", "first frame", lanes)
        nodes.append({"id": "act", "label": "action plan", "op": "ActionInput", "kind": "input", "scope": "inputs"})
        nodes.append({"id": "actin", "label": "action_proj_in", "op": "action_proj_in", "kind": "compute", "scope": "cond"})
        edges.append({"from": "act", "to": "actin", "shape": "T×D"})
        for lane in lanes:
            edges.append({"from": "actin", "to": lane, "shape": "act z"})

    if mode_id == "inv_dynamics":
        nodes.append({"id": "readout", "label": "action_proj_out", "op": "action_proj_out", "kind": "compute", "scope": "decode"})
        nodes.append({"id": "aout", "label": "action", "op": "Output", "kind": "output", "scope": "out", "shape": "T×D"})
        edges.extend([{"from": "step", "to": "readout", "shape": z}, {"from": "readout", "to": "aout", "shape": "T×D"}])
    else:
        nodes.append({"id": "projout", "label": "proj_out · unpatchify", "op": "proj_out + unpatchify", "kind": "compute", "scope": "decode", "shape": z})
        nodes.append({"id": "vae", "label": "VAE decode", "op": "vae.decode", "kind": "compute", "scope": "decode", "dims": ["vae"]})
        nodes.append({"id": "vout", "label": "image" if m["kind"] == "image" else "video", "op": "Output", "kind": "output", "scope": "out", "shape": px})
        edges.extend([{"from": "step", "to": "projout", "shape": z}, {"from": "projout", "to": "vae", "shape": z},
                      {"from": "vae", "to": "vout", "shape": px}])
        if params.get("generate_sound"):
            nodes.append({"id": "sndout", "label": "audio_proj_out", "op": "audio_proj_out", "kind": "compute", "scope": "decode"})
            nodes.append({"id": "snd", "label": "sound decode", "op": "sound_vae.decode", "kind": "compute", "scope": "decode"})
            nodes.append({"id": "sout", "label": "waveform", "op": "Output", "kind": "output", "scope": "out", "shape": "48kHz"})
            edges.extend([{"from": "step", "to": "sndout", "shape": "audio z"}, {"from": "sndout", "to": "snd", "shape": "audio z"},
                          {"from": "snd", "to": "sout", "shape": "48kHz"}])
    return {"scopes": scopes, "nodes": nodes, "edges": edges}


def request_preview(mode_id: str, params: dict[str, Any]) -> dict[str, Any]:
    """Show exactly what the playground will send downstream for the current settings,
    without running it — the core explainability hook for the Pipeline panel."""
    m = mode(mode_id)
    if m["surface"] == "reason":
        return {"surface": "reason",
                "prompt": (params.get("prompt") or m.get("example", {}).get("prompt") or ""),
                "max_tokens": _num(params, "max_tokens", int) or 512,
                "temperature": _num(params, "temperature", float),
                "stages": execution_stages(mode_id, params),
                "graph": execution_graph(mode_id, params)}
    p = dict(params)
    if not (p.get("prompt") or "").strip():
        p["prompt"] = m.get("example", {}).get("prompt", "(your prompt)")
    req = build_request(mode_id, p)
    out = _summarize_request(req)
    out["surface"] = "generate"
    out["stages"] = execution_stages(mode_id, p)
    out["graph"] = execution_graph(mode_id, p)
    ep = req["extra_params"]
    dom = action_domain(ep.get("domain_name", ep.get("domain_id")))
    if dom:
        out["domain"] = dom
    return out


def example_action(mode_id: str) -> dict[str, Any] | None:
    """The action plan that drives forward dynamics, exposed for visualization."""
    if mode_id != "fwd_dynamics":
        return None
    chunks = _FD_SPEC.get("action_chunks", [])
    flat = [step for ch in chunks for step in ch]
    return {"domain_name": _FD_SPEC.get("domain_name"), "fps": _FD_SPEC.get("fps"),
            "action_chunk_size": _FD_CHUNK, "num_chunks": _FD_NCHUNKS,
            "shape": [len(flat), len(flat[0]) if flat else 0], "data": flat,
            "domain": action_domain(_FD_SPEC.get("domain_name"))}


# ------------------------------------------------------------------- reason requests
def build_reason_messages(mode_id: str, params: dict[str, Any], media: bytes | None,
                          media_name: str | None, media_type: str | None = None) -> dict[str, Any]:
    """Build an OpenAI chat request for the REASON surface (media in -> text out)."""
    m = mode(mode_id)
    prompt = (params.get("prompt") or "").strip() or (m.get("example", {}).get("prompt") or "Describe this.")
    content: list[dict[str, Any]] = []
    if media is not None:
        # prefer the upload's declared content-type; fall back to the filename, then the
        # mode's expected reference kind — so a video is never mis-sent as an image.
        mime = media_type if (media_type or "").startswith(("image", "video")) else None
        mime = mime or mimetypes.guess_type(media_name or "")[0]
        if not mime:
            mime = "video/mp4" if m.get("reference") == "video" else "image/png"
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
