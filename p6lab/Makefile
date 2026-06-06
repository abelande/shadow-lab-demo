# p6lab — common developer tasks.
#
# Targets:
#   make install       — editable install with [lab,notebook,dev] extras
#   make smoketest     — execute fast notebooks under pytest (≈90s)
#   make smoketest-all — execute every notebook including slow ones (~5min)
#   make test          — run the unit/integration test suite
#   make freeze        — refresh requirements-lock.txt from current env
#   make clean         — remove __pycache__ + .pytest_cache + ipynb checkpoints
#
# All targets assume Python ≥ 3.11.

PY ?= python3
PIP ?= $(PY) -m pip
PYTEST ?= $(PY) -m pytest

.PHONY: help install smoketest smoketest-all test freeze clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS=":.*?# "}; {printf "  %-18s %s\n", $$1, $$2}'

install: # editable install with all dev/lab extras
	$(PIP) install --break-system-packages -e ".[lab,notebook,dev]"

smoketest: # execute fast notebooks (NB03/NB05/NB06/NB07) end-to-end
	$(PYTEST) tests/test_smoke_notebooks.py -v

smoketest-all: # also run NB04 (~3min)
	$(PYTEST) tests/test_smoke_notebooks.py -v --slow

smoketest-instruments: # cross-instrument harness (ES/CL/GC/SI × portable NBs)
	$(PYTEST) tests/test_smoke_instruments.py -v

bench: # latency benchmark — asserts engine.match() p95 < 50ms
	$(PYTEST) tests/test_engine_latency.py -v -s

test-parity: # live ↔ replay feed parity contract tests (Wave 2 #1A)
	$(PYTEST) tests/test_feed_parity.py -v

test-divergence: # live ↔ replay score-divergence auditor (Wave 2 #1C)
	$(PYTEST) tests/test_divergence_audit.py -v

test-cpcv-multiday: # multi-day calendar-aware CPCV (Wave 2 #2)
	$(PYTEST) tests/test_cpcv_multiday.py -v

test-cascade-taxonomy: # Type A/B/C/D cascade classifier (Wave 2 #3)
	$(PYTEST) tests/test_cascade_taxonomy.py -v

test-logging: # structured logging with correlation IDs (Wave 2 #4)
	$(PYTEST) tests/test_logging.py -v

test-wave2: test-parity test-divergence test-cpcv-multiday test-cascade-taxonomy test-logging # all Wave 2 regression tests

test-wave3: test-live # Wave 3 live integration regression (via MockLiveFeed)

sample: # regenerate data/nq-mbo-sample-15min.dbn.zst (reproducible CI fixture)
	$(PY) scripts/generate_sample_data.py

ci: # run the exact check sequence GitHub Actions runs (unit + bench + renderers + wave2 + smoketest)
	$(PYTEST) tests/ -v --ignore=tests/test_smoke_notebooks.py --ignore=tests/test_smoke_instruments.py
	$(MAKE) bench
	$(MAKE) test-renderers
	$(MAKE) test-wave2
	$(MAKE) smoketest

test-live: # Wave 3 live runner integration test (uses committed sample, no network)
	$(PYTEST) tests/test_live_runner.py -v

run-live-mock: # rehearse live pipeline against the committed 15-min NQ sample
	$(PY) scripts/run_live.py \
	  --mock-source ../data/nq-mbo-sample-15min.dbn.zst \
	  --duration 30 --json-logs \
	  --audit-log artifacts/p6lab/live/matches.jsonl \
	  --metrics-port 8001

run-live: # fire the REAL live pipeline (requires DATABENTO_API_KEY in .env)
	$(PY) scripts/run_live.py \
	  --symbol NQ --json-logs \
	  --audit-log artifacts/p6lab/live/matches.jsonl \
	  --metrics-port 8001

test-discord: # send one synthetic tier-A match to your Discord webhook
	$(PY) scripts/test_discord_webhook.py

test-audit-log: # smoketest JSONL audit renderer — N matches × M threads
	$(PY) scripts/test_audit_log.py

test-metrics: # smoketest MetricsRenderer + scrape Prometheus /metrics
	$(PY) scripts/test_metrics.py

test-chart-overlay: # smoketest triple_view_panel correlation overlay (node)
	node scripts/test_chart_overlay.js

test-renderers: test-audit-log test-metrics test-chart-overlay # all 3 (minus Discord, which requires your live webhook URL)

test: # unit + integration test suite
	$(PYTEST) tests/ -v --ignore=tests/test_smoke_notebooks.py

freeze: # rewrite requirements-lock.txt from current env
	@$(PY) -c "import importlib.metadata as m; \
	  pkgs = ['numpy','pandas','scikit-learn','scipy','hdbscan','umap-learn',\
	          'pyarrow','pydantic','sortedcontainers','filelock','lightgbm',\
	          'tabulate','databento','jupyter','nbclient']; \
	  print('# Auto-generated — refresh with: make freeze'); \
	  [print(f'{p}=={m.version(p)}') for p in pkgs]" > requirements-lock.txt
	@echo 'wrote requirements-lock.txt'

clean: # remove pyc, caches, ipynb checkpoints
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ipynb_checkpoints' -prune -exec rm -rf {} +
