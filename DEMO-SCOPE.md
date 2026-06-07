# Demo Scope — Shadow Lab Microstructure Engine

This is a stripped, public-facing build of a private order-book microstructure
research stack. Demo URL: `<microstructure.example.com>`

## What's real
- The 6-layer detection pipeline (`pipeline.py` + `cup_flip/`, `spoof_detection/`,
  `staircase_analyzer/`, `spectral_force/`, `regime_context/`, `level_tracker`) —
  production code, unmodified.
- The frontend (`web/`): chart, DOM panel, overlays, replay scrubber, paper exec sim.
- The WebSocket frame contract (`DepthIndicatorFrame`).
- The synthetic L3 generator (`ingestion/synthetic.py`).

## What's stripped
- **Live Databento connector + API key** — no live feed; no credentials in this build.
- **Real licensed market data** (`*.dbn.zst`) — none shipped.
- **Mined pattern library + trained correlation models** (`.pkl`) — replaced with a
  3-pattern illustrative stub (`demo/library_demo.yaml`); the correlation engine runs
  *unloaded* and does not emit pattern matches in the demo.
- **Production-tuned detector thresholds** — the values shipped here are illustrative
  demo defaults; production-tuned weights are withheld (annotated in-code).

## What runs
The bundled synthetic L3 feed streams through the real pipeline at ~30 Hz in
`DEMO_MODE`. A deterministic 60-second loop scripts microstructure events (momentum
runs, institutional walls, spoofs, stop-runs) so every detector layer is visible.

## Credentials
**None.** The pipeline is CSV/synthetic-driven and fully offline — there were no
secrets to rotate. (No `.env`, no `.dbn.zst`, no `.pkl`/`.pt` in this repository.)

## Run it
```
DEMO_MODE=true python3 run_server.py        # http://localhost:8420/
```

## What you can do here
- Watch the live synthetic feed at `/`.
- See cup-flip / spoof / regime overlays fire on the scripted loop.
- Drag SL/TP on the paper exec sim.
- Read the architecture, accuracy, and roadmap at `/about`.

## Contact
[email] · [LinkedIn]
