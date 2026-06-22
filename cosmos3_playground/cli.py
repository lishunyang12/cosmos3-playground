# SPDX-License-Identifier: Apache-2.0
"""``cosmos3-playground`` console entry point."""

from __future__ import annotations

import argparse

import uvicorn

from cosmos3_playground import __version__, server


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos3 Playground — web UI for a vLLM-Omni Cosmos3 server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument(
        "--cosmos-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the vLLM-Omni server serving Cosmos3 (OpenAI-compatible).",
    )
    parser.add_argument("--model", default=None, help="Served model name (default: first from /v1/models).")
    parser.add_argument(
        "--reasoner-url",
        default=None,
        help="Base URL of an OpenAI-compatible vLLM reasoner server (Cosmos3 reasoner) for the "
        "Reason surface (captioning, grounding, physical reasoning). Optional.",
    )
    parser.add_argument("--reasoner-model", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--version", action="version", version=f"cosmos3-playground {__version__}")
    args = parser.parse_args()

    app = server.create_app(args.cosmos_url, args.model, args.reasoner_url, args.reasoner_model, args.api_key)
    print(f"Cosmos3 Playground {__version__} -> {args.cosmos_url}  (UI: http://{args.host}:{args.port})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
