#!/usr/bin/env python3
"""Launch the P6 Staircase Terminal server.

This script sets up the correct import context so that both the server
(server/) and core modules (pipeline.py, models.py, etc.) can use
relative imports within the p6-v2 package.

Usage:
    python3 run_server.py [--host 0.0.0.0] [--port 8420]
"""
import sys
import os

# Load .env file so DATABENTO_API_KEY and other secrets are available.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# Ensure the parent directory of p6-v2 is on sys.path so that
# `from p6v2.server.app import app` and `from p6v2.pipeline import ...`
# both resolve correctly.
project_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_dir)

# The project directory name may be "p6-v2" which isn't a valid Python
# package name. We need to create/verify a package alias.
package_name = os.path.basename(project_dir)
if package_name == "p6-v2":
    # Symlink p6v2 → p6-v2 should exist; if not, use direct path manipulation
    alias = os.path.join(parent_dir, "p6v2")
    if os.path.islink(alias) or os.path.isdir(alias):
        package_name = "p6v2"

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P6 Staircase Terminal Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    # Import through the package name so relative imports work
    uvicorn.run(
        f"{package_name}.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
