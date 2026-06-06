# Architecture — P6 Staircase Terminal (v2)

> **Status:** Stub — fill in with actual architecture before first pipeline cycle.
> Pipeline agent will flag spec drift against this document each build.

---

## System Overview

_(Describe the high-level system: what it is, what it does, major components)_

## Component Map

_(List major modules and their responsibilities)_
- `ingestion/` —
- `core/` —
- `p6v2/` —
- `server/` —
- `depth_indicator/` —
- `spoof_detection/` —
- `regime_context/` —

## Data Flow

_(How data moves through the system from ingestion to output)_

## Key Interfaces

_(APIs, ports, protocols)_
- Port 8420: p6-server (run_server.py)
- Databento MBO feed (GLBX.MDP3)

## Dependencies

_(External services, libraries, data sources this system depends on)_
- Databento API (MBO L3 data)
- Redis
- Python 3.x

## Known Fragile Areas

_(Fill in as the pipeline discovers them)_
