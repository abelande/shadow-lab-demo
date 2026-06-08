# Demo Scope — p6lab Research Notebooks

A public, read-only window into the **p6lab** research library: leakage-free
microstructure ML in notebook form. Companion to the microstructure demo (`../`).

## What's real
- The **methodology and code**, shown in full: L1/L2 feature engineering,
  triple-barrier / event labeling, **combinatorial purged cross-validation (CPCV)**
  with purge + embargo, cascade classification, and paper execution simulation.
- The `cv_visual.py` CPCV figure is generated live (matplotlib, no data).
- These are real research notebooks, rendered to static HTML from their committed runs.

## What's redacted / withheld
- **Result numbers are redacted** — every floating-point value in the notebook
  *outputs* (Sharpe, PnL, hit-rate, AUC, returns, drawdowns, etc.) is blanked to `▒.▒▒`.
  The code that computes them is fully visible; the realized numbers are not published.
- **Raw L3 data is not shipped** — the notebooks read licensed Databento MBO
  (`.dbn.zst`) data, which is withheld. The rendered HTML carries no data.
- **Absolute paths scrubbed** from both code and outputs.
- Trained models, mined pattern libraries, and run artifacts — not shipped.

## What you can read here
- `index.html` — sidebar of 7 rendered notebooks.
- `04_CPCV_FIGURE` — the one live visual (purge/embargo structure).
- `01`–`07` — the feature → label → CPCV → cascade → execution methodology.

## Credentials
- **None.** Static HTML; nothing executes at serve time.

## Contact
[email] · [LinkedIn]
