#!/usr/bin/env python3
"""Staircase Terminal — entry point.

Usage:
    cd projects/ && python -m p6.run              # Start on port 8420
    cd projects/ && python -m p6.run --port 9000  # Custom port
"""
from __future__ import annotations

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Staircase Terminal Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8420, help="Bind port")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from p6.server.config import config
    config.host = args.host
    config.port = args.port

    import uvicorn
    uvicorn.run(
        "p6.server.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
