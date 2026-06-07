# Architecture — Shadow Lab Microstructure Engine

A live order-book microstructure terminal: a feed is processed through a multi-layer
detection pipeline and streamed to a browser over a WebSocket at ~30 Hz. This public
build runs on a synthetic Level-3 feed (no live data, no credentials).

---

## System Overview

The engine consumes order-book snapshots, runs each through a deterministic detection
pipeline that emits a single `DepthIndicatorFrame`, and broadcasts those frames to all
connected clients. The frontend renders the chart, depth-of-market ladder, and a set of
overlays (cup-flip state, spoof/authenticity, regime, force) plus a paper exec sim.

## Component Map

- `ingestion/` — feed adapters. `synthetic.py` generates a deterministic L3 book with
  injectable microstructure events; the live/replay adapters are inert in the demo.
- `pipeline.py` — `OrderBookMetaPipeline.run(snapshot) -> DepthIndicatorFrame`; orchestrates
  the detector layers.
- `cup_flip/` — tape-reading state machine (BALANCED / BULL_STREAK / BEAR_STREAK / STALL / STOP_RUN).
- `spoof_detection/` — pull-before-touch, layering, phantom-wall, iceberg, and a weighted
  authenticity score.
- `staircase_analyzer/` — depth fragility scoring.
- `spectral_force/` — institutional force / energy bands.
- `regime_context/` — regime classification → layer weights.
- `level_tracker.py`, `depth_indicator/` — level lifecycle and DOM construction.
- `server/` — FastAPI app (`app.py`), the engine runner (`engine_runner.py`), and the
  WebSocket manager (`websocket.py`).
- `web/` — vanilla-JS frontend (no build step).
- `p6lab/` — research library; the engine imports `p6lab.patterns` / `p6lab.correlation` /
  `p6lab.live` for the optional pattern-correlation layer (runs unloaded in the demo).

## Data Flow

```
feed.next() ─▶ engine_runner queue ─▶ pipeline.run(snapshot) ─▶ DepthIndicatorFrame
                                                                      │
                                              ws_manager.broadcast ───┘ ─▶ /ws ─▶ web/
```

In `DEMO_MODE`, `engine_runner.start_demo_feed()` drives `ingestion/synthetic.py` as the
producer; a scripted 60-second loop injects momentum runs, walls, spoofs, and stop-runs.

## Key Interfaces

- Port **8420** — FastAPI server (`run_server.py`).
- `GET /` — terminal UI · `GET /about` — honesty page · `GET /api/status` — engine status.
- `WS /ws` — `DepthIndicatorFrame` stream + `price_tick` messages.

## Dependencies

- Python 3.x, FastAPI + Uvicorn, NumPy, Pydantic, PyYAML.
- Optional: a native `core/` C++ extension for acceleration — **not required**; the pure-Python
  path is the default and the only one used in this demo.
- No external services or data providers are used in the demo (no Databento, no broker).

## Known Fragile Areas

- The app uses package-relative imports and must be launched via `run_server.py` (which sets
  up the package alias) rather than imported directly.
- The pattern-correlation layer requires a matching mined library; the demo ships illustrative
  patterns that do not fire matches, so correlation badges stay empty by design.
