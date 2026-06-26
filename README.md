# Cosmos3 Playground

A polished web **playground** for NVIDIA **Cosmos3** world-foundation models, served by
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). Packaged as a vLLM-Omni **plugin**:
the model runs in your vLLM-Omni server (OpenAI-compatible), and this playground is a
decoupled UI you point at it.

<img width="1875" height="870" alt="image" src="https://github.com/user-attachments/assets/6b40f8e3-8122-4c29-90db-d26a122ad94d" />

<img width="1861" height="766" alt="image" src="https://github.com/user-attachments/assets/651c7213-66e1-4c38-94b2-7d5e2ec75316" />

> Imagine → animate → edit → hear a world, from one model. Cosmos3 picks the task per
> request; the playground just gives each task a nice front door.

## What it gives you

One unified interface over the two Cosmos3 surfaces from the official cookbook:

**Generate** (diffusion → media), against a vLLM-Omni Cosmos3 server:
| Group | Modes |
|---|---|
| **Imagine** | Text → Image, Text → Video |
| **Animate** | Image → Video |
| **Edit** | Video → Video, Transfer (edge / blur / depth / seg / wsm) |
| **Simulate** | Forward dynamics (action → future video), Inverse dynamics (video → action trajectory) |
| — | + **Generate sound** toggle (muxed AAC) on any video mode |

**Reason** (world understanding → text), against an OpenAI-compatible vLLM reasoner:
| Group | Modes |
|---|---|
| **Reason** | Captioning · Temporal localization · 2D grounding · Physical reasoning · Ask anything |

Each mode **pre-loads its official example** (prompt + recommended settings + reference,
sourced from the `nvidia/Cosmos3-Nano` model card) so you can just hit **Generate** /
**Analyze**. Plus: live **progress bar** with ETA for async video jobs, an **action-tensor
viewer** for dynamics, a **reference preview** (image/video), and server profiling.

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

**1. Serve Cosmos3 (Generate surface)** with vLLM-Omni:
```bash
vllm serve nvidia/Cosmos3-Nano --omni --port 8000 --no-guardrails   # +--quantization fp8 for ~24GB GPUs
```
Note: video output writes to a storage dir — set `VLLM_OMNI_STORAGE_PATH=/some/writable/dir`
if `/tmp/storage` isn't writable.

**2. (Optional) Serve a reasoner (Reason surface)** — any OpenAI-compatible vLLM VLM:
```bash
vllm serve <cosmos3-reasoner-or-any-VLM> --port 8001
```
The Reason surface is model-agnostic; point `--reasoner-url` at the Cosmos3 reasoner for
true Cosmos reasoning, or any chat VLM. Omit it and the Reason tabs show "reasoner off".

**3. Install + run the playground:**
```bash
pip install -e .
cd frontend && npm install && npm run build && cd ..     # build the UI once (Node 18+)
cosmos3-playground --cosmos-url http://127.0.0.1:8000 --reasoner-url http://127.0.0.1:8001 --port 8800
```
Open **http://localhost:8800**. `--model` / `--reasoner-model` are optional (default to the
first model from each server's `/v1/models`).

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

- [x] Generate: Imagine / Animate / Edit / Sound
- [x] Simulate: forward / inverse dynamics (action-tensor viewer)
- [x] Reason: captioning / temporal localization / grounding / physical reasoning (decoupled reasoner)
- [ ] Robot control loop (OpenPI WebSocket)
- [ ] Cosmos3-Super multi-GPU presets, fp8 toggle surfaced in UI
- [ ] Grounding box overlay on the reasoned image

## License

Apache-2.0.
