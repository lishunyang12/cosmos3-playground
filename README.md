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
| **World Model** | Text → Image, Text → Video, Image → Video, Video → Video |
| **Sim2Real (SDG)** | Transfer · Sim→Real (edge / blur / depth / seg / wsm control) |
| **Robotics** | Forward dynamics (first frame + action plan → future video), Policy (first frame + instruction → predicted actions + manipulation video) |
| **Autonomous Driving** | Inverse dynamics (video → ego-motion / action trajectory) |
| — | + **Generate sound** toggle (muxed AAC) on any video mode |

**Reason** (world understanding → text), against an OpenAI-compatible vLLM reasoner:
| Group | Modes |
|---|---|
| **Reason** | Captioning · Temporal localization · 2D grounding · Physical reasoning · Ask anything |

Each mode **pre-loads its official example** (prompt + paper-aligned settings + reference,
from the Cosmos3 paper / `cosmos-framework` inference defaults) so you can just hit
**Generate** / **Analyze**. Video generation also applies the paper's **B.6 structured
negative prompt** by default (Text2Image and action/dynamics modes use the null string,
matching the framework's `sample_args.json`).

**Cached example gallery.** Opening the URL shows each mode's **pre-baked output by
default** — no waiting — with a *"cached example · press Generate to run live"* badge.
Pressing Generate re-runs the mode live and replaces it. Bake the gallery once with the
server running:
```bash
python -m cosmos3_playground.prebake --base http://127.0.0.1:8800   # all modes
# --only t2i,policy   subset    ·   --skip transfer   all but these
```
This drives the same API as the Generate button and saves each result under
`cosmos3_playground/prebaked/` (served via `GET /api/example/{mode}/result`).

Plus: live **progress bar** with ETA for async video jobs, an **action-tensor / trajectory
viewer** for dynamics and policy, a **reference preview** (image/video), and server profiling.

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

- [x] World Model: Text/Image/Video → Image/Video (+ sound), paper-faithful negative prompt
- [x] Sim2Real: video Transfer (edge / blur / depth / seg / wsm)
- [x] Robotics: forward dynamics + Policy (model-predicted action rollout)
- [x] Autonomous Driving: inverse dynamics (ego-motion trajectory viewer)
- [x] Reason: captioning / temporal localization / grounding / physical reasoning (decoupled reasoner)
- [x] Cached example gallery (pre-baked outputs, regenerate on demand)
- [ ] Robot control loop (OpenPI WebSocket)
- [ ] Cosmos3-Super multi-GPU presets, fp8 toggle surfaced in UI
- [ ] Grounding box overlay on the reasoned image

## License

Apache-2.0.
