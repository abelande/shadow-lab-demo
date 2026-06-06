"""
p6lab.correlation.engine
========================
Runtime correlation engine — §7.1 of the P6 Lab Spec.

Core role: live L2 (+ optional L1) state → matched patterns with scores.
This is the pure-Python implementation of the L3↔L1/L2 correlation
architecture from OB-reference.md §L356-L549.

Architecture
------------
The engine operates in the server's ``engine_runner`` main loop, called
after each ``OrderBookMetaPipeline.run()`` invocation.  Flow:

    L2 window (30s–5m of L2 features)
      + optional L1 window
      + MatchContext (TOD, VIX regime, instrument)
      │
      ▼
    RegimeConditioner → filter template set for current regime (§7.3)
      │
      ▼
    TemplateMatcher.match() → raw MatchResults (§5.4)
      │
      ▼
    EnsembleScorer → PatternMatch list with tier assignments (§7.2)
      │
      ▼
    Filter: keep tier A/B/C (ensemble_score ≥ 0.60), discard D
      │
      ▼
    Return sorted list of PatternMatch objects

Latency target: <50ms per match call (OB-reference.md §L476).
Benchmarked in notebook 06 §10.

Model loading
-------------
The engine loads a trained model (from ``correlation_runs/models/{version}.pkl``)
exported by notebook 06 (§9.4 §12).  Model reload via:
    POST /api/correlation/reload (§11.4)

Two-stage architecture (notebook 06 §07)
-----------------------------------------
Stage 1: L1-only features → fast pre-screen (<5ms)
Stage 2: L1+L2 features → full scoring  (<50ms total)
If Stage 1 score < 0.40, skip Stage 2 entirely (optimization).

References
----------
- Spec §7.1 — engine API, latency target, consumer list
- OB-reference.md §L356-L549 — correlation engine design
- OB-reference.md §L476 — 50ms latency target
- Spec §9.4 notebook 06 — training, validation, model export
- Spec §11.4 — correlation_api.py server wiring
- Spec §10.4 — correlation_feed.js UI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

import logging
import pickle

from p6lab.patterns.library import PatternLibrary, PatternDefinition
from p6lab.patterns.template_matcher import (
    MatchContext, MatchResult, PatternTemplate, TemplateMatcher,
)
from p6lab.correlation.regime_conditioner import FIConditioner, RegimeConditioner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum ensemble score to keep a match (Tier C threshold).
MIN_MATCH_SCORE: float = 0.60

#: Stage 1 (L1-only) pre-screen threshold — below this, skip Stage 2.
STAGE1_PRESCREEN_THRESHOLD: float = 0.40

#: Maximum L2 window length in seconds for a single match call.
MAX_WINDOW_SECONDS: float = 300.0  # 5 minutes

#: Minimum L2 window length in seconds.
MIN_WINDOW_SECONDS: float = 30.0

#: Maximum number of matches returned per call.
MAX_MATCHES_PER_CALL: int = 20


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PatternMatch:
    """A matched pattern with score and metadata.

    Attributes
    ----------
    pattern_id:
        Library pattern slug.
    ensemble_score:
        Combined score ∈ [0, 1] from the template matcher ensemble.
    confidence_tier:
        'A' (≥0.85), 'B' (0.72–0.85), 'C' (0.60–0.72).
    expected_direction:
        'bull', 'bear', or 'neutral'.
    expected_move_atr:
        Expected ATR-normalized move at the 5m horizon (from library).
    template_similarity:
        Cosine similarity component of the ensemble.
    mahalanobis_score:
        Mahalanobis distance component of the ensemble.
    contextual_score:
        Context (TOD, VIX, volume) component of the ensemble.
    match_window_start_ms:
        Start of the L2 window used for matching.
    match_window_end_ms:
        End of the L2 window (== current time).
    regime:
        VIX regime at match time.
    instrument:
        Matched instrument.
    stage1_score:
        Stage 1 (L1-only) pre-screen score.
    """

    pattern_id: str
    ensemble_score: float
    confidence_tier: Literal["A", "B", "C"]
    expected_direction: Literal["bull", "bear", "neutral"]
    expected_move_atr: float
    template_similarity: float
    mahalanobis_score: float
    contextual_score: float
    match_window_start_ms: int
    match_window_end_ms: int
    regime: str
    instrument: str
    stage1_score: float = 0.0
    # Wave 9-A: calibrated LightGBM directional probability stamped per
    # snapshot when the engine has a primary model loaded. ``None`` when
    # no model is wired in (legacy / matcher-only mode).
    primary_proba: float | None = None


# ---------------------------------------------------------------------------
# CorrelationEngine
# ---------------------------------------------------------------------------


class CorrelationEngine:
    """Runtime L2 → pattern match engine.

    Parameters
    ----------
    library:
        Pattern library (active + mined_approved patterns).
    matcher:
        TemplateMatcher with fitted covariance matrices.
    scorer:
        EnsembleScorer for tier assignment.
    regime_classifier:
        Callable that maps (VIX value,) → regime string.
        Default: use VIX buckets from instrument_normalizer (§3.3).

    Usage (server integration, §11.4)
    ----------------------------------
    >>> engine = CorrelationEngine(library, matcher, scorer, regime_fn)
    >>> # In engine_runner main loop:
    >>> matches = engine.match(l2_window, l1_window, context)
    >>> for m in matches:
    ...     websocket.send('correlation_match', m)
    """

    def __init__(
        self,
        library: PatternLibrary,
        matcher: TemplateMatcher,
        scorer: Any,  # EnsembleScorer — imported at runtime to avoid circular
        regime_classifier: Callable[[float], str] | None = None,
        broker: Any = None,  # MatchBroker — optional pub/sub for live consumers
        *,
        base_rate: float = 0.5,
    ) -> None:
        self.library = library
        self.matcher = matcher
        self.scorer = scorer
        self.regime_classifier = regime_classifier or self._default_regime_classifier
        self.broker = broker

        # Cached active pattern IDs for the current regime (refreshed on regime change).
        self._active_pattern_ids: list[str] = []
        self._current_regime: str | None = None

        # Model version string (from the pkl filename).
        self.model_version: str = "unloaded"

        # Wave 9-A: optional primary LightGBM model (extracted by
        # ``reload_model`` from ``model_dict["lightgbm_model"]``). When None,
        # the engine runs matcher-only — the legacy behavior. When loaded,
        # ``match()`` accepts a ``feature_row`` and stamps a calibrated
        # directional proba onto each ``PatternMatch``.
        self._primary_model: Any = None
        self._primary_feature_names: list[str] = []

        # Wave 9 A5: uninformative prior emitted when the activity detector
        # marks a snapshot inactive (``is_active=False`` passed to match()).
        # The model is trained on activity-masked rows, so on inactive rows
        # we emit base rate rather than extrapolating outside the training
        # support. Default 0.5 — symmetric ignorance for binary
        # directional output. Adjust to the empirical class prior for
        # multi-class or skewed-base-rate downstream use.
        self._base_rate: float = float(base_rate)

    def match(
        self,
        l2_window: pd.DataFrame,
        l1_window: pd.DataFrame | None,
        context: MatchContext,
        feature_row: "np.ndarray | dict[str, float] | None" = None,
        is_active: bool | None = None,
    ) -> list[PatternMatch]:
        """All patterns matching above threshold, sorted by score.

        Parameters
        ----------
        l2_window:
            Last 30s–5m of L2 features.  Columns include
            ``book_shape_vector`` (40-dim array) and the 12 L2 features.
            Index = timestamp_ms.
        l1_window:
            Optional L1 features for Stage 1 pre-screen.  16-dim per row.
            Index = timestamp_ms.
        context:
            MatchContext with TOD, VIX regime, volume ratio, instrument.
        feature_row:
            Wave 9-A — the latest snapshot's flat feature vector, used by
            the primary LightGBM model to produce ``primary_proba``.
            ``None`` (the legacy default) means matcher-only mode; no
            ``primary_proba`` is stamped on returned matches.
        is_active:
            Wave 9 A5 — activity-mask state for the current snapshot.
            ``True`` → trust the model output. ``False`` → emit
            ``self._base_rate`` (the soft prior — model wasn't trained
            on inactive states). ``None`` (legacy default) → behave
            exactly like Wave 9-A: model output if feature_row provided,
            else None.

        Returns
        -------
        list[PatternMatch]
            Sorted descending by ensemble_score.  Only tier A/B/C included.
            Maximum MAX_MATCHES_PER_CALL entries.

        Latency
        -------
        Target <50ms.  Stage 1 pre-screen on L1 features runs first (<5ms);
        if score < STAGE1_PRESCREEN_THRESHOLD (0.40), skip the pattern
        entirely for Stage 2.
        """
        # Structured logging: stamp one correlation_id per match cycle so
        # every downstream log (broker dispatch, renderer logs, webhook
        # POST failure, etc.) can be joined back to this call.
        from p6lab._logging import new_correlation_id, with_context
        with with_context(
            correlation_id=new_correlation_id(),
            instrument=context.instrument,
            regime=str(self.regime_classifier(context.vix_level)),
        ):
            return self._match_body(
                l2_window, l1_window, context, feature_row, is_active,
            )

    def _match_body(
        self, l2_window, l1_window, context, feature_row=None, is_active=None,
    ):
        regime = self.regime_classifier(context.vix_level)
        # 1. Regime-conditioned pattern set
        if regime != self._current_regime:
            conditioner = RegimeConditioner()
            sel = conditioner.select_patterns(self.library, context.instrument, regime)  # type: ignore[arg-type]
            self._active_pattern_ids = sel.selected_pattern_ids
            self._current_regime = regime
            logger.debug("Regime change → %s; %d active patterns",
                         regime, len(self._active_pattern_ids))

        if not self._active_pattern_ids:
            return []

        # 1b. FI-conditioned filter (Phase 5B). Narrows the regime-filtered
        # set to patterns whose fi_bucket matches the live context. FI is a
        # gate, not a feature — patterns without an fi_bucket tag pass
        # through unchanged (backward compatible).
        fi_bucket = getattr(context, "fi_bucket", "calm")
        if fi_bucket in ("calm", "elevated", "fragile"):
            fi_cond = FIConditioner()
            fi_ok = set(fi_cond.select_patterns(
                self.library, context.instrument, fi_bucket,  # type: ignore[arg-type]
            ))
            active_ids = [pid for pid in self._active_pattern_ids if pid in fi_ok]
            if not active_ids:
                return []
        else:
            active_ids = self._active_pattern_ids

        # 1c. Cost gate (Wave 4 Phase 1E). For each candidate pattern,
        # estimate expected edge = pattern.outcome.mean_atr × atr_recent
        # converted to ticks; reject if edge doesn't clear slippage ×
        # cost_multiplier. Skipped when atr_recent or slippage_bps is 0
        # (live runner populates them from the accumulator; default 0
        # means "gate disabled" for backward compatibility).
        atr_recent = float(getattr(context, "atr_recent", 0.0) or 0.0)
        slippage_bps = float(getattr(context, "slippage_bps", 0.0) or 0.0)
        if atr_recent > 0 and slippage_bps > 0:
            active_ids = self._apply_cost_gate(
                active_ids, atr_recent, slippage_bps,
            )
            if not active_ids:
                return []

        # 2. Build candidate book-shape series from L2 window
        candidate_bsv = self._extract_bsv_series(l2_window)
        candidate_features = self._latest_l2_features(l2_window)
        l1_latest = (
            np.asarray(l1_window.iloc[-1].to_numpy()) if l1_window is not None
            and len(l1_window) > 0 else None
        )

        # Wave 9-A: compute the directional proba once per call (it depends
        # only on the snapshot, not on each candidate pattern). When the
        # primary model is unloaded or no feature_row was provided, this is
        # None — matches go out without primary_proba stamped.
        # Wave 9 A5: pass is_active so the engine can substitute the soft
        # prior when the activity mask flags this snapshot inactive.
        primary_proba = self._predict_primary_proba(
            feature_row, is_active=is_active,
        )
        if primary_proba is not None:
            logger.debug(
                "primary_proba=%.4f (is_active=%s)", primary_proba, is_active,
            )

        results: list[PatternMatch] = []
        end_ms = int(l2_window.index[-1]) if len(l2_window) > 0 else 0
        start_ms = int(l2_window.index[0]) if len(l2_window) > 0 else 0

        # 3. Per-pattern: Stage 1 → Stage 2
        for pid in active_ids:
            tmpl = self.matcher.templates.get(pid)
            if tmpl is None:
                continue

            stage1 = 1.0
            if l1_latest is not None:
                stage1 = self._stage1_prescreen(l1_latest, pid)
                if stage1 < STAGE1_PRESCREEN_THRESHOLD:
                    continue

            r = self.matcher.match(
                current_book_series=candidate_bsv,
                current_features=candidate_features,
                template_book_series=tmpl.book_series,
                template_features=tmpl.feature_centroid,
                pattern_id=pid,
                context=context,
                pattern_context=tmpl.pattern_context,
                primary_proba=primary_proba,
            )
            if r.ensemble_score < MIN_MATCH_SCORE:
                continue

            tier = "A" if r.ensemble_score >= 0.85 else (
                "B" if r.ensemble_score >= 0.72 else "C"
            )
            pat = self.library.get_active_patterns().get(pid)
            expected_dir = "neutral"
            expected_atr = 0.0
            if pat and pat.outcome_distribution:
                # Pick "5m" or first available
                key = "5m" if "5m" in pat.outcome_distribution else next(iter(pat.outcome_distribution))
                od = pat.outcome_distribution[key]
                expected_atr = od.mean_atr
                expected_dir = "bull" if expected_atr > 0.1 else (
                    "bear" if expected_atr < -0.1 else "neutral"
                )

            results.append(PatternMatch(
                pattern_id=pid,
                ensemble_score=r.ensemble_score,
                confidence_tier=tier,
                expected_direction=expected_dir,
                expected_move_atr=expected_atr,
                template_similarity=r.template_cosine_similarity,
                mahalanobis_score=r.mahalanobis_distance,
                contextual_score=r.contextual_score,
                match_window_start_ms=start_ms,
                match_window_end_ms=end_ms,
                regime=regime,
                instrument=context.instrument,
                stage1_score=stage1,
                primary_proba=r.primary_proba,
            ))

        results.sort(key=lambda m: m.ensemble_score, reverse=True)
        results = results[:MAX_MATCHES_PER_CALL]

        # Fan out to any pub/sub subscribers (WebSocket, dock, audit log, ...).
        # Broker is optional — legacy callers that construct the engine without
        # one keep working unchanged.
        if self.broker is not None and results:
            for m in results:
                self.broker.emit(m)

        return results

    @staticmethod
    def _extract_bsv_series(l2_window: pd.DataFrame) -> np.ndarray:
        """Stack the book_shape_vector column into a (T, 40) array."""
        if "book_shape_vector" not in l2_window.columns or len(l2_window) == 0:
            return np.zeros((0, 40))
        arr = np.stack([np.asarray(v, dtype=float) for v in l2_window["book_shape_vector"]])
        return arr

    # L2 columns excluded from centroid-matching: book_shape_vector is the
    # 40-dim pyramid (matched via DTW, not mahalanobis), and weighted_mid is
    # the label source (leaks forward return info into feature space).
    _L2_MATCH_EXCLUDE = ("book_shape_vector", "weighted_mid")

    # Wave 4 Phase 1E — cost gate multiplier: expected edge must clear
    # ``slippage × COST_MULTIPLIER`` for the pattern to fire.
    COST_GATE_MULTIPLIER = 2.0

    def _apply_cost_gate(
        self,
        active_ids: list[str],
        atr_recent: float,
        slippage_bps: float,
    ) -> list[str]:
        """Reject patterns whose expected move doesn't clear cost.

        Expected edge (ticks) = ``pattern.outcome.mean_atr × atr_recent``
        (mean_atr is in ATR multiples; atr_recent is the current ATR in
        price units so edge_ticks = mean_atr_multiple × atr_recent / tick_size).
        Slippage (ticks) ≈ ``slippage_bps / 10_000 × price / tick_size``.
        Without a price anchor we approximate via atr_recent as a proxy
        for notional. The gate is deliberately conservative: reject if
        edge_ticks < slippage_bps × COST_MULTIPLIER / 10000 × 1000 =
        simply compare edge_ticks ≥ slippage_ticks × COST_MULTIPLIER.
        """
        kept: list[str] = []
        # Approximate tick_size via the smallest first-differnece of the
        # BSV anchor. Simpler fallback: use 0.25 as default (NQ) when
        # library lacks the info. Production wiring passes via
        # MatchContext in a future revision.
        tick_size = 0.25
        # Rough slippage-in-ticks from bps (assume price ~= 20000 for NQ).
        # TODO(wave5): pass price into MatchContext to avoid the constant.
        ref_price = max(atr_recent * 100.0, 100.0)
        slippage_ticks = (slippage_bps / 10_000.0) * ref_price / tick_size
        min_edge_ticks = slippage_ticks * self.COST_GATE_MULTIPLIER

        active = self.library.get_active_patterns()
        for pid in active_ids:
            pat = active.get(pid)
            if pat is None:
                kept.append(pid)
                continue
            od = pat.outcome_distribution or {}
            if not od:
                kept.append(pid)
                continue
            # Use the largest-horizon entry as the expected-move reference
            best_mean_atr = max((x.mean_atr for x in od.values()), default=0.0)
            edge_ticks = abs(best_mean_atr) * atr_recent / tick_size
            if edge_ticks >= min_edge_ticks:
                kept.append(pid)
            else:
                logger.debug(
                    "cost_gate REJECT %s edge_ticks=%.2f < min_edge_ticks=%.2f",
                    pid, edge_ticks, min_edge_ticks,
                )
        return kept

    @classmethod
    def _latest_l2_features(cls, l2_window: pd.DataFrame) -> np.ndarray:
        """Return the most recent L2 feature row (without book_shape_vector / weighted_mid)."""
        if len(l2_window) == 0:
            return np.zeros(10)
        cols = [c for c in l2_window.columns if c not in cls._L2_MATCH_EXCLUDE]
        if not cols:
            return np.zeros(10)
        return np.asarray(l2_window[cols].iloc[-1].to_numpy(), dtype=float)

    def reload_library(self, library: PatternLibrary) -> None:
        """Hot-reload the pattern library (triggered by POST /api/correlation/reload).

        Clears the regime-conditioned cache to force a re-filter on next match().
        """
        self.library = library
        self._current_regime = None
        self._active_pattern_ids = []

    @staticmethod
    def resolve_current_model(registry_path):
        """Read ``CURRENT.json`` and return the active model pickle path.

        The registry file is a JSON object with at minimum:
            {
              "version": "v1_nq_fwd1m",
              "model_path": "v1_nq_fwd1m_20260415_1840/v1_nq_fwd1m.pkl",
              "promoted_at": "2026-04-15T18:40:00Z",
              "promoted_by": "nb06-baseline-gate"
            }

        ``model_path`` is resolved relative to ``registry_path.parent`` so the
        registry can live next to the versioned model directories.
        """
        import json
        from pathlib import Path as _P
        rp = _P(registry_path)
        if not rp.is_file():
            raise FileNotFoundError(f"model registry missing: {rp}")
        with open(rp, encoding="utf-8") as fh:
            entry = json.load(fh)
        mp = entry.get("model_path")
        if not mp:
            raise ValueError(f"{rp} missing 'model_path' key")
        full = (rp.parent / mp).resolve()
        if not full.is_file():
            raise FileNotFoundError(f"model file missing: {full}")
        return full

    def load_current_model(self, registry_path) -> None:
        """Load the model pointed to by ``CURRENT.json`` — the production pointer.

        Thin wrapper around :meth:`reload_model`. Single-line change at call
        site: engine authors don't touch filenames; promotion is a deliberate
        write to ``CURRENT.json``, not "whichever file happens to have the
        newest mtime".
        """
        model_path = self.resolve_current_model(registry_path)
        self.reload_model(str(model_path))

    def _predict_primary_proba(
        self,
        feature_row: "np.ndarray | dict[str, float] | None",
        is_active: bool | None = None,
    ) -> float | None:
        """Run the primary LightGBM model on one snapshot's features.

        Accepts either a 1-D ``np.ndarray`` (caller already aligned to
        ``self._primary_feature_names``) or a ``dict[str, float]`` mapping
        feature name → value (the helper aligns to the model's expected
        column order using ``_primary_feature_names``).

        Wave 9 A5 — activity gate (``is_active``):
          - ``True`` → trust the model; run ``predict_proba`` as Wave 9-A.
          - ``False`` → the snapshot is outside the model's training
            support (activity mask was False at training time). Emit
            the configured ``self._base_rate`` as an uninformative prior
            instead of extrapolating.
          - ``None`` (legacy) → ignore the gate; run model when
            ``feature_row`` provided. Preserves backward-compat for
            callers that don't supply activity state.

        Returns ``None`` when:
          - the primary model is not loaded,
          - ``feature_row`` is ``None`` AND ``is_active`` is not False,
          - the feature row has the wrong width,
          - the dict is missing a required column,
          - or prediction otherwise fails.

        Failures log a warning and degrade to ``None`` rather than
        raising, so a single bad snapshot can't take down the engine
        loop.
        """
        if self._primary_model is None:
            return None
        # Wave 9 A5: short-circuit to soft prior when activity gate fails.
        # We require the model to be loaded (above) so the absence of a
        # model still yields None — A5 is layered on top of 9-A wiring.
        if is_active is False:
            return self._base_rate
        if feature_row is None:
            return None
        try:
            if isinstance(feature_row, dict):
                if not self._primary_feature_names:
                    return None
                try:
                    row = np.asarray(
                        [float(feature_row[n]) for n in self._primary_feature_names],
                        dtype=float,
                    )
                except KeyError as missing:
                    logger.warning(
                        "primary_proba: feature dict missing column %s; "
                        "skipping inference",
                        missing,
                    )
                    return None
            else:
                row = np.asarray(feature_row, dtype=float).ravel()
            if (
                self._primary_feature_names
                and row.shape[0] != len(self._primary_feature_names)
            ):
                logger.warning(
                    "primary_proba: feature row width %d != expected %d; "
                    "skipping inference",
                    row.shape[0],
                    len(self._primary_feature_names),
                )
                return None
            proba_arr = np.asarray(
                self._primary_model.predict_proba(row.reshape(1, -1)),
            )
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                # Binary classifier convention: column 1 = positive class.
                return float(proba_arr[0, 1])
            return float(proba_arr.ravel()[0])
        except Exception as exc:
            logger.warning("primary_proba: prediction failed (%s)", exc)
            return None

    def reload_model(self, model_path: str) -> None:
        """Load a new trained model from disk.

        Model format: pickle of a dict with keys:
            'matcher_templates': dict[str, np.ndarray]
            'matcher_centroids': dict[str, np.ndarray]
            'matcher_covariances': dict[str, np.ndarray]
            'pattern_contexts': dict[str, dict]
            'version': str
            'lightgbm_model': bytes (pickle.dumps of trained LightGBM)  -- Wave 9-A
            'feature_names': list[str]                                  -- Wave 9-A

        Exported by notebook 06 §12. ``lightgbm_model`` and ``feature_names``
        are optional — pickles produced before Wave 9-A omit them and the
        engine continues to run matcher-only with a logged warning.
        """
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)  # noqa: S301 — trusted local artifact
        templates = model_dict.get("matcher_templates", {})
        centroids = model_dict.get("matcher_centroids", {})
        contexts = model_dict.get("pattern_contexts", {})
        covariance = model_dict.get("global_covariance", None)
        self.matcher.templates = {
            pid: PatternTemplate(
                pattern_id=pid,
                book_series=np.asarray(templates[pid]),
                feature_centroid=np.asarray(centroids.get(pid, np.zeros(12))),
                pattern_context=contexts.get(pid, {}),
            )
            for pid in templates
        }
        if covariance is not None:
            self.matcher.fit_covariance(np.asarray(covariance))
        self.model_version = model_dict.get("version", "unknown")

        # Wave 9-A: extract the primary LightGBM model + feature schema.
        # Pickles exported before 9-A omit these; that is not a hard error,
        # the engine simply runs matcher-only.
        raw_model = model_dict.get("lightgbm_model")
        if raw_model is None:
            self._primary_model = None
            self._primary_feature_names = []
            logger.warning(
                "reload_model: no 'lightgbm_model' in pickle — "
                "engine runs matcher-only (Wave 9-A wiring inactive)",
            )
        else:
            self._primary_model = pickle.loads(raw_model)  # noqa: S301
            self._primary_feature_names = list(
                model_dict.get("feature_names", []),
            )
            logger.info(
                "reload_model: primary model loaded with %d features",
                len(self._primary_feature_names),
            )

        # Reset regime cache so next match() re-filters
        self._current_regime = None
        self._active_pattern_ids = []
        logger.info("Loaded model %s with %d templates",
                    self.model_version, len(self.matcher.templates))

    def _stage1_prescreen(
        self,
        l1_features: np.ndarray,
        pattern_id: str,
    ) -> float:
        """L1-only quick pre-screen.

        Uses cosine similarity between the live L1 vector and the pattern's
        stored feature centroid. Scale-invariant — works regardless of
        whether features are z-scored, raw magnitudes, or normalized. Maps
        [-1, 1] cosine to [0, 1] for consistency with the threshold.

        Returns score ∈ [0, 1]. Must run in <5ms.
        """
        tmpl = self.matcher.templates.get(pattern_id)
        if tmpl is None or tmpl.feature_centroid.size == 0:
            return 1.0  # no centroid to compare against — pass through
        n = min(len(l1_features), len(tmpl.feature_centroid))
        if n == 0:
            return 1.0
        a = np.asarray(l1_features[:n], dtype=float)
        b = np.asarray(tmpl.feature_centroid[:n], dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 1.0
        cos = float(np.dot(a, b) / (na * nb))        # [-1, 1]
        return max(0.0, min(1.0, (cos + 1.0) / 2.0)) # → [0, 1]

    @staticmethod
    def _default_regime_classifier(vix: float) -> str:
        """Default VIX regime buckets (§3.3 instrument_normalizer).

        Returns
        -------
        'low'      if vix < 15
        'normal'   if 15 ≤ vix < 25
        'elevated' if 25 ≤ vix < 35
        'high'     if vix ≥ 35
        """
        if vix < 15:
            return "low"
        if vix < 25:
            return "normal"
        if vix < 35:
            return "elevated"
        return "high"
