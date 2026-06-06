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

# The app uses package-relative imports (e.g. `from ..models import ...`), so it
# must be imported as a package. The project dir name may not be a valid Python
# identifier (e.g. "shadow-lab-demo"); create/verify a sanitized symlink alias.
project_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_dir)

package_name = os.path.basename(project_dir)
if not package_name.isidentifier():
    alias = package_name.replace("-", "_")
    alias_path = os.path.join(parent_dir, alias)
    if not (os.path.islink(alias_path) or os.path.isdir(alias_path)):
        try:
            os.symlink(project_dir, alias_path)
        except OSError:
            pass
    package_name = alias

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Make the bundled p6lab library importable (powers the optional correlation engine).
_p6lab_src = os.path.join(project_dir, "p6lab", "src")
if os.path.isdir(_p6lab_src) and _p6lab_src not in sys.path:
    sys.path.insert(0, _p6lab_src)

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
