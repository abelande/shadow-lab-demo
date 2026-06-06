#!/usr/bin/env python3
"""Stress test for P6 server — replay feed, WebSocket frames, API responsiveness."""
import asyncio
import json
import time
import sys
import websockets
import aiohttp

BASE = "http://127.0.0.1:8420"
WS_URL = "ws://127.0.0.1:8420/ws"

# Test data files
FULL_EXCHANGE = "/home/bel/.openclaw/workspace-principal/projects/p6-v2/data/glbx-mdp3-20260223.mbo.dbn.zst"
SINGLE_NQ = "/home/bel/.openclaw/workspace-principal/projects/p6-v2/data/nq-mbo-2026-03-24.dbn.zst"
SINGLE_ES = None  # no single-ES file currently

results = []

def log(test, status, detail=""):
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    results.append((test, status, detail))
    print(f"  {icon} {test}: {detail}")


async def api(method, path, body=None, timeout=10):
    async with aiohttp.ClientSession() as s:
        kwargs = {"timeout": aiohttp.ClientTimeout(total=timeout)}
        if body:
            kwargs["json"] = body
        async with getattr(s, method)(BASE + path, **kwargs) as r:
            return r.status, await r.json()


async def test_api_responsive():
    """Test 1: API responds within 2s while idle."""
    t0 = time.time()
    code, data = await api("get", "/api/status")
    elapsed = time.time() - t0
    if code == 200 and elapsed < 2:
        log("API responsive (idle)", "PASS", f"{elapsed:.2f}s, mode={data.get('mode')}")
    else:
        log("API responsive (idle)", "FAIL", f"code={code}, elapsed={elapsed:.2f}s")


async def test_files_endpoint():
    """Test 2: /api/data/files returns files with valid metadata."""
    code, data = await api("get", "/api/data/files")
    if code != 200:
        log("Files endpoint", "FAIL", f"code={code}")
        return
    symbols_found = set(f.get("symbol") for f in data)
    has_multi = any(f.get("multi_instrument") for f in data)
    log("Files endpoint", "PASS", f"{len(data)} entries, symbols={symbols_found}, multi={has_multi}")

    # Check dates are valid (not "Invalid Date")
    for f in data:
        if f.get("start") and "Invalid" in str(f.get("start", "")):
            log("File dates valid", "FAIL", f"Invalid date in {f.get('file')}")
            return
    log("File dates valid", "PASS", "All dates parseable")


async def test_replay_single_instrument(file_path, symbol, label):
    """Test: Replay a single-instrument file and collect frames."""
    # Stop any running feed
    await api("post", "/api/feed/stop")
    await asyncio.sleep(0.5)

    # Start replay
    code, data = await api("post", "/api/feed/replay", {
        "file_path": file_path,
        "symbol": symbol,
        "snapshot_interval_ms": 100,
    }, timeout=15)

    if code != 200:
        log(f"Replay start ({label})", "FAIL", f"code={code}, detail={data}")
        return 0

    log(f"Replay start ({label})", "PASS", f"status={data.get('status')}")

    # Collect frames via WebSocket for 15 seconds
    frames = []
    api_responsive = True
    t0 = time.time()

    async def collect_frames():
        try:
            async with websockets.connect(WS_URL, close_timeout=2) as ws:
                while time.time() - t0 < 15:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        frame = json.loads(msg)
                        if "timestamp_ms" in frame:
                            frames.append(frame)
                    except asyncio.TimeoutError:
                        continue
        except Exception as e:
            log(f"WS collect ({label})", "WARN", str(e))

    async def check_api_during_replay():
        nonlocal api_responsive
        await asyncio.sleep(3)  # Wait for replay to be actively processing
        for i in range(5):
            try:
                t1 = time.time()
                code, _ = await api("get", "/api/status", timeout=5)
                elapsed = time.time() - t1
                if code != 200 or elapsed > 3:
                    api_responsive = False
                    log(f"API during replay ({label}) poll {i}", "FAIL", f"code={code}, {elapsed:.2f}s")
            except Exception as e:
                api_responsive = False
                log(f"API during replay ({label}) poll {i}", "FAIL", str(e))
            await asyncio.sleep(2)

    await asyncio.gather(collect_frames(), check_api_during_replay())

    # Stop feed
    await api("post", "/api/feed/stop")

    # Analyze frames
    if len(frames) == 0:
        log(f"Frames received ({label})", "FAIL", "0 frames in 15s")
        return 0

    # Check frame quality
    has_stats = sum(1 for f in frames if f.get("stats") and f["stats"].get("fill_count", 0) > 0)
    has_tape = sum(1 for f in frames if f.get("tape") and len(f["tape"]) > 0)
    has_dom = sum(1 for f in frames if f.get("dom_rows") and len(f["dom_rows"]) > 0)
    has_signal = sum(1 for f in frames if f.get("confidence", 0) > 0)
    has_force = sum(1 for f in frames if f.get("force_vector") and f["force_vector"].get("total_force", 0) != 0)
    has_game = sum(1 for f in frames if f.get("game_state") and f["game_state"].get("state") != "BALANCED")

    log(f"Frames received ({label})", "PASS", f"{len(frames)} frames in 15s ({len(frames)/15:.1f} fps)")
    log(f"Stats populated ({label})", "PASS" if has_stats > 0 else "FAIL", f"{has_stats}/{len(frames)} with fill_count>0")
    log(f"Tape populated ({label})", "PASS" if has_tape > 0 else "FAIL", f"{has_tape}/{len(frames)} with trades")
    log(f"DOM populated ({label})", "PASS" if has_dom > 0 else "FAIL", f"{has_dom}/{len(frames)} with dom_rows")
    log(f"Signals present ({label})", "PASS" if has_signal > 0 else "WARN", f"{has_signal}/{len(frames)} with confidence>0")
    log(f"Force vector ({label})", "PASS" if has_force > 0 else "WARN", f"{has_force}/{len(frames)} with force!=0")
    log(f"Game state ({label})", "PASS" if has_game > 0 else "WARN", f"{has_game}/{len(frames)} non-BALANCED")
    log(f"API responsive during ({label})", "PASS" if api_responsive else "FAIL", "All polls < 3s" if api_responsive else "Some polls timed out")

    # Print sample frame keys for debugging
    if frames:
        sample = frames[len(frames)//2]
        log(f"Sample frame keys ({label})", "PASS",
            f"keys={sorted(sample.keys())[:10]}")

    return len(frames)


async def test_replay_rth():
    """Test: Full-exchange file with RTH filter."""
    await api("post", "/api/feed/stop")
    await asyncio.sleep(0.5)

    code, data = await api("post", "/api/feed/replay", {
        "file_path": FULL_EXCHANGE,
        "symbol": "ES",
        "filter_symbol": "ES",
        "rth_only": True,
        "snapshot_interval_ms": 100,
    }, timeout=15)

    if code != 200:
        log("RTH replay start (ES full-exchange)", "FAIL", f"code={code}")
        return

    log("RTH replay start (ES full-exchange)", "PASS", f"status={data.get('status')}")

    # Wait 20s, check frames + API
    frames = []
    t0 = time.time()
    api_times = []

    async def collect():
        try:
            async with websockets.connect(WS_URL, close_timeout=2) as ws:
                while time.time() - t0 < 20:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        frame = json.loads(msg)
                        if "timestamp_ms" in frame:
                            frames.append(frame)
                    except asyncio.TimeoutError:
                        continue
        except Exception:
            pass

    async def poll_api():
        await asyncio.sleep(5)
        for _ in range(5):
            try:
                t1 = time.time()
                await api("get", "/api/status", timeout=5)
                api_times.append(time.time() - t1)
            except:
                api_times.append(99)
            await asyncio.sleep(2)

    await asyncio.gather(collect(), poll_api())
    await api("post", "/api/feed/stop")

    avg_api = sum(api_times) / len(api_times) if api_times else 99
    log(f"RTH frames (ES full-exchange)", "PASS" if len(frames) > 0 else "FAIL",
        f"{len(frames)} frames in 20s")
    log(f"API latency during RTH replay", "PASS" if avg_api < 3 else "FAIL",
        f"avg={avg_api:.2f}s, max={max(api_times):.2f}s")

    if frames:
        # Verify timestamps are in RTH range (13:30-20:00 UTC)
        from datetime import datetime, timezone
        rth_ok = 0
        for f in frames:
            ts = f.get("timestamp_ms", 0)
            if ts > 0:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                h, m = dt.hour, dt.minute
                if (h > 13 or (h == 13 and m >= 30)) and h < 20:
                    rth_ok += 1
        log(f"RTH timestamps valid", "PASS" if rth_ok == len(frames) else "WARN",
            f"{rth_ok}/{len(frames)} in RTH window")


async def test_stop_start_cycle():
    """Test: Rapid stop/start doesn't crash."""
    for i in range(3):
        await api("post", "/api/feed/stop")
        await asyncio.sleep(0.5)
        code, _ = await api("post", "/api/feed/replay", {
            "file_path": FULL_EXCHANGE,
            "symbol": "NQ",
            "filter_symbol": "NQ",
            "snapshot_interval_ms": 100,
        }, timeout=10)
        if code != 200:
            log(f"Stop/start cycle {i}", "FAIL", f"code={code}")
            return
        await asyncio.sleep(1)

    await api("post", "/api/feed/stop")
    code, data = await api("get", "/api/status", timeout=5)
    log("Stop/start cycling (3x)", "PASS" if code == 200 else "FAIL",
        f"Server stable, mode={data.get('mode')}")


async def main():
    print("\n" + "="*60)
    print("  P6 STRESS TEST")
    print("="*60 + "\n")

    print("── Test 1: API Baseline ──")
    await test_api_responsive()

    print("\n── Test 2: Files Endpoint ──")
    await test_files_endpoint()

    print("\n── Test 3: Single-Instrument NQ Replay (15s) ──")
    nq_frames = await test_replay_single_instrument(SINGLE_NQ, "NQ", "NQ-single")

    print("\n── Test 4: Full-Exchange NQ with Time Range (mid-RTH, 15s) ──")
    # Use time range to skip to active trading (not overnight book-building)
    await api("post", "/api/feed/stop")
    await asyncio.sleep(0.5)
    code, data = await api("post", "/api/feed/replay", {
        "file_path": FULL_EXCHANGE,
        "symbol": "NQ",
        "filter_symbol": "NQ",
        "time_start": "2026-02-23T15:00:00",
        "time_end": "2026-02-23T16:00:00",
        "snapshot_interval_ms": 100,
    }, timeout=15)
    if code == 200:
        log("NQ mid-RTH replay start", "PASS", f"status={data.get('status')}")
        # Collect frames for 15s
        frames = []
        t0 = time.time()
        try:
            async with websockets.connect(WS_URL, close_timeout=2) as ws:
                while time.time() - t0 < 15:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        frame = json.loads(msg)
                        if "timestamp_ms" in frame:
                            frames.append(frame)
                    except asyncio.TimeoutError:
                        continue
        except Exception:
            pass
        has_stats = sum(1 for f in frames if f.get("stats") and f["stats"].get("fill_count", 0) > 0)
        has_tape = sum(1 for f in frames if f.get("tape") and len(f["tape"]) > 0)
        log(f"NQ mid-RTH frames", "PASS" if len(frames) > 0 else "FAIL", f"{len(frames)} frames")
        log(f"NQ mid-RTH stats w/fills", "PASS" if has_stats > 0 else "FAIL", f"{has_stats}/{len(frames)}")
        log(f"NQ mid-RTH tape populated", "PASS" if has_tape > 0 else "FAIL", f"{has_tape}/{len(frames)}")
        await api("post", "/api/feed/stop")
    else:
        log("NQ mid-RTH replay start", "FAIL", f"code={code}")

    print("\n── Test 5: Full-Exchange ES RTH Replay (20s) ──")
    await test_replay_rth()

    print("\n── Test 6: Single-Instrument NQ Replay (15s) ──")
    if SINGLE_NQ:
        nq2 = await test_replay_single_instrument(SINGLE_NQ, "NQ", "NQ-single-file")
    else:
        log("NQ single replay", "WARN", "No single-NQ file available")

    print("\n── Test 7: Stop/Start Cycling ──")
    await test_stop_start_cycle()

    # Final cleanup
    await api("post", "/api/feed/stop")

    # Summary
    print("\n" + "="*60)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    warned = sum(1 for _, s, _ in results if s == "WARN")
    print(f"  RESULTS: {passed} passed, {failed} failed, {warned} warnings")
    print(f"  TOTAL:   {len(results)} tests")
    print("="*60 + "\n")

    if failed > 0:
        print("FAILED TESTS:")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  ❌ {name}: {detail}")
        print()

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
