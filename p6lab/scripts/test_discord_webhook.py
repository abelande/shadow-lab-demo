"""
Fire one synthetic tier-A match through the Discord webhook renderer.

Loads ``p6lab/.env`` itself via python-dotenv — no shell-state fiddling needed.
Run from anywhere:

    python3 scripts/test_discord_webhook.py

Prints sent / dropped counts and exits non-zero if the POST failed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Locate project root (parent of this script's dir) and add src/ to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "p6lab"))

# Load .env from the p6lab root. python-dotenv is already pulled in by
# the lab extras; if it's somehow missing, fall back to manual parsing.
env_path = ROOT / ".env"
if not env_path.is_file():
    print(f"ERROR: .env missing at {env_path}", file=sys.stderr)
    sys.exit(2)

# Load both the repo-root (p6-v2/.env) and lab (p6lab/.env) files so
# operators can split secrets across them. p6lab/.env wins on overlap.
_env_paths = [ROOT.parent / ".env", env_path]
for _p in _env_paths:
    if not _p.is_file():
        continue
    try:
        from dotenv import load_dotenv
        load_dotenv(_p, override=False)
    except ImportError:
        for line in _p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

url = os.environ.get("P6LAB_DISCORD_WEBHOOK_URL")
if not url:
    print("ERROR: P6LAB_DISCORD_WEBHOOK_URL not set in .env", file=sys.stderr)
    sys.exit(2)

print(f"loaded URL: {url!r}")

from p6lab.correlation.match_broker import MatchBroker
from p6lab.correlation.renderers import WebhookRenderer


class _FakeMatch:
    confidence_tier = "A"
    tier = "A"
    pattern_id = "smoketest_discord"
    ensemble_score = 0.92
    expected_direction = "bull"
    expected_move_atr = 1.35
    template_similarity = 0.88
    mahalanobis_score = 0.75
    contextual_score = 0.70
    stage1_score = 0.82
    match_window_start_ms = int(time.time() * 1000) - 60_000
    match_window_end_ms = int(time.time() * 1000)
    regime = "normal"
    instrument = "NQ"


def main() -> int:
    broker = MatchBroker()
    discord = WebhookRenderer(
        url, platform="discord",
        tier_filter={"A", "B"},
        min_score=0.80,
        max_per_minute=20,
    )
    broker.subscribe(discord)

    print("emitting synthetic tier-A match to broker...")
    broker.emit(_FakeMatch())

    # Wait for the background POST thread. The renderer uses a daemon thread,
    # so we need to give it time before the process exits.
    for _ in range(30):
        time.sleep(0.1)
        if discord.posts_sent or discord.posts_dropped:
            break

    print(f"posts_sent    = {discord.posts_sent}")
    print(f"posts_dropped = {discord.posts_dropped}")

    if discord.posts_sent >= 1:
        print("OK — check your Discord channel for the message.")
        return 0
    print("FAILED — no post went through. Likely HTTP error (webhook "
          "invalidated, permission denied, etc.). See warning log above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
