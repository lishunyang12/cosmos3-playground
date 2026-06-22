# Cosmos3 Playground

A polished web **playground** for NVIDIA **Cosmos3** world-foundation models, served by
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). Packaged as a vLLM-Omni **plugin**:
the model runs in your vLLM-Omni server (OpenAI-compatible), and this playground is a
decoupled UI you point at it.

> Imagine → animate → edit → hear a world, from one model. Cosmos3 picks the task per
> request; the playground just gives each task a nice front door.

![modes](docs/modes.png)

## What it gives you

| Group | Modes | Endpoint |
|---|---|---|
| **Imagine** | Text → Image, Text → Video | `/v1/images/generations`, `/v1/videos` |
| **Animate** | Image → Video | `/v1/videos` (+ image reference) |
| **Edit** | Video → Video, Transfer (edge / blur / depth / seg / wsm control) | `/v1/videos` (+ video reference, `extra_params`) |
| **Sound** | add muxed audio to any video mode (`generate_sound`) | `/v1/videos` |

Plus the core diffusion knobs (resolution, frames, fps, steps, guidance, flow-shift,
negative prompt, seed), a live **progress bar** for async video jobs, and the server's
profiling (inference time, peak VRAM) surfaced in the UI.

## Architecture

```
 Browser (React/Vite SPA)        cosmos3-playground (FastAPI)        vLLM-Omni server (Cosmos3)
 ┌────────────────────┐   /api   ┌───────────────────────────┐  /v1  ┌──────────────────────┐
 │ mode tabs · knobs  │ ───────▶ │ mode → request mapping     │ ────▶ │ /v1/images/generations│
 │ media · progress   │ ◀─────── │ ref upload · async poll    │ ◀──── │ /v1/videos (+ /content)│
 └────────────────────┘          └───────────────────────────┘       └──────────────────────┘
```
The backend is thin — its only real job is mapping a playground mode to the right
OpenAI-compatible request (folding mode-specific controls into `extra_params`),
uploading references, and proxying the async video job. Everything else is the stock API,
so the same playground works for any vLLM-Omni diffusion model.

## Quick start

**1. Serve Cosmos3 with vLLM-Omni** (separate process / GPU):
```bash
vllm serve nvidia/Cosmos3-Nano --omni --port 8000        # add --quantization fp8 for ~24GB GPUs
```

**2. Install + run the playground:**
```bash
pip install -e .                 # installs the backend + the `cosmos3-playground` command
# build the UI once (Node 18+):
cd frontend && npm install && npm run build && cd ..
cosmos3-playground --cosmos-url http://127.0.0.1:8000 --port 8800
```
Open **http://localhost:8800**.

`--model` is optional (defaults to the first model from `/v1/models`).

## Development

```bash
# terminal 1 — backend (proxies /v1 to your Cosmos3 server)
cosmos3-playground --cosmos-url http://127.0.0.1:8000 --port 8800
# terminal 2 — Vite dev server with HMR (proxies /api → :8800)
cd frontend && npm run dev    # http://localhost:5173
```

The **mode catalog lives in one place** (`cosmos3_playground/modes.py`): the backend builds
requests from it and the frontend renders the UI from it (`GET /api/config`). Add a mode
there and it appears in the UI automatically.

## As a vLLM-Omni plugin

Installing the package registers a light `vllm.general_plugins` entry point that only
advertises availability (it does **not** pull FastAPI into the engine process). The UI runs
out-of-process via the `cosmos3-playground` command against a running server.

## Roadmap

- [x] Imagine / Animate / Edit / Sound
- [ ] Simulate (action world-model: forward / inverse dynamics, policy) with action-tensor viewer
- [ ] Robot control loop (OpenPI WebSocket)
- [ ] Cosmos3-Super multi-GPU presets, fp8 toggle surfaced in UI

## License

Apache-2.0.
