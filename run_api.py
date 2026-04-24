"""API launcher — sets sys.path before uvicorn imports the app module.

Usage:
    uv run python run_api.py [--host HOST] [--port PORT] [--no-reload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Remove project root from sys.path before any import so that
# src/schemas.py takes priority over the root-level schemas.py.
# Python adds the script's directory (project root) automatically;
# we strip it here and prepend src/ instead.
_root = Path(__file__).parent.resolve()
_src = str(_root / "src")
sys.path = [p for p in sys.path if Path(p).resolve() != _root]
if _src not in sys.path:
    sys.path.insert(0, _src)

import uvicorn  # noqa: E402


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IUT Market Oracle API server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-reload", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )
