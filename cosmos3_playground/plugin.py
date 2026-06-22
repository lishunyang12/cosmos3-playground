# SPDX-License-Identifier: Apache-2.0
"""vLLM general-plugin entry point.

This is intentionally import-light: it must not pull FastAPI/uvicorn into the vLLM
engine process. It only advertises that the Cosmos3 Playground is installed; the UI
runs out-of-process via the ``cosmos3-playground`` console command.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("cosmos3_playground")


def register() -> None:
    logger.info(
        "Cosmos3 Playground is installed. Launch the web UI with: "
        "`cosmos3-playground --cosmos-url <this-server>/`"
    )
