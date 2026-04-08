"""
server/app.py — OpenEnv-compliant entry point for MetaQuestOSEnv.

This module is the [project.scripts] entry point required by openenv-core.
It imports and re-exports the FastAPI app from main.py.
"""

import sys
import os

# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: F401 — re-exported for openenv-core


def main():
    """Entry point for 'uv run server' command."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()