"""
Template Matcher — Cosine Similarity Ensemble Scoring
Spec §5.4 | OB-reference.md:470-487

Ensemble score:
    ensemble = 0.40 × cosine_similarity
             + 0.35 × mahalanobis_score   (1 - normalized distance)
             + 0.25 × contextual_score

Mahalanobis covariance fit via Ledoit-Wolf shrinkage (sklearn).
Falls back to Euclidean when condition number κ(Σ) > 1000.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

WEIGHT_TEMPLATE_MATCH = 0.40
WEIGHT_MAHALANOBIS = 0.35
WEIGHT_CONTEXTUAL = 0.25

CONDITION_NUMBER_THRESHOLD = 1000.0
BOOK_SHAPE_DIM = 40

# Mahalanobis distance is open-ended; normalize via 1 / (1 + d) so the
# "score" component is in (0, 1].
def _distance_to_score(d: float) -> float:
    if d < 0 or not np.isfinite(d):
        return 0.0
    return 1.0 / (1.0 + d)


@dataclass(frozen=True)
class MatchContext:
    """Contextual information for ensemble scoring."""
    time_of_day_minutes: int
    vix_level: float
    vix_regime: str
    relative_volume: float
    instrument: str
    # Fragility Index fields (Phase 5B). FI is a gate, not a feature —
    # the engine filters patterns by ``fi_bucket`` before scoring.
    # Default 0.0/'calm' keeps existing call sites valid.
    fi_fast: float = 0.0
    fi_bucket: str = "calm"
    # Wave 4 Phase 1E — cost gate inputs. atr_recent is the rolling-20
    # ATR (in price units); slippage_bps is the estimated per-fill
    # slippage in basis points. Engine rejects patterns whose
    # expected_move_atr (from library outcome_distribution) doesn't
    # clear slippage × cost_multiplier. Defaults = no gate applied.
    atr_recent: float = 0.0
    slippage_bps: float = 0.0


@dataclass(frozen=True)
class MatchResult:
    """Result of matching against one pattern template (a.k.a. TemplateMatchResult).

    ``primary_proba`` carries the calibrated LightGBM directional probability
    when the engine has a primary model loaded (Wave 9-A wiring). It is
    ``None`` for legacy callers that don't supply it; Wave 9-C uses it as the
    fourth slot in the configurable fusion combiner.
    """
    pattern_id: str
    template_cosine_similarity: float
    mahalanobis_distance: float
    contextual_score: float
    ensemble_score: float
    used_euclidean_fallback: bool
    primary_proba: float | None = None


# Backwards-compat alias for code that imported the old name.
TemplateMatchResult = MatchResult


@dataclass
class PatternTemplate:
    """One pattern's templates + reference vectors used by the matcher."""
    pattern_id: str
    book_series: np.ndarray         # (T, 40)
    feature_centroid: np.ndarray    # (D,)
    pattern_context: dict[str, Any] = field(default_factory=dict)


class TemplateMatcher:
    """Matches current book state against pattern templates.

    Latency target: <50ms per match call (§7.1).
    """

    def __init__(self) -> None:
        self._covariance: np.ndarray | None = None
        self._covariance_inv: np.ndarray | None = None
        self._use_euclidean: bool = False
        self.templates: dict[str, PatternTemplate] = {}

    # ──────────────────────────────────────────────────────────────
    # Covariance fitting
    # ──────────────────────────────────────────────────────────────

    def fit_covariance(self, training_vectors: np.ndarray) -> None:
        """Fit Ledoit-Wolf shrinkage covariance. Sets fallback flag if ill-conditioned."""
        from sklearn.covariance import LedoitWolf
        if training_vectors.ndim != 2 or training_vectors.shape[0] < 2:
            self._use_euclidean = True
            self._covariance = None
            self._covariance_inv = None
            return
        lw = LedoitWolf().fit(training_vectors)
        cov = np.asarray(lw.covariance_)
        cond = float(np.linalg.cond(cov)) if cov.size > 0 else float("inf")
        self._covariance = cov
        if cond > CONDITION_NUMBER_THRESHOLD or not np.isfinite(cond):
            logger.warning(
                "TemplateMatcher: covariance ill-conditioned (κ=%.1f) — "
                "falling back to Euclidean", cond,
            )
            self._use_euclidean = True
            self._covariance_inv = None
        else:
            self._use_euclidean = False
            self._covariance_inv = np.linalg.pinv(cov)

    # ──────────────────────────────────────────────────────────────
    # Component scoring
    # ──────────────────────────────────────────────────────────────

    def cosine_similarity(
        self,
        current_series: np.ndarray,
        template_series: np.ndarray,
    ) -> float:
        """Cosine similarity on flattened time series. Result in [-1, 1]."""
        if current_series.size == 0 or template_series.size == 0:
            return 0.0
        # Align lengths by truncating to the shorter series
        n = min(current_series.shape[0], template_series.shape[0])
        if n == 0:
            return 0.0
        a = current_series[-n:].ravel().astype(np.float64)
        b = template_series[-n:].ravel().astype(np.float64)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def mahalanobis_distance(
        self,
        current_features: np.ndarray,
        template_features: np.ndarray,
    ) -> float:
        """Mahalanobis distance using fitted covariance, or Euclidean fallback."""
        diff = np.asarray(current_features, dtype=np.float64) - np.asarray(
            template_features, dtype=np.float64
        )
        if self._use_euclidean or self._covariance_inv is None:
            return float(np.linalg.norm(diff))
        try:
            d2 = float(diff @ self._covariance_inv @ diff)
            if d2 < 0:
                d2 = 0.0
            return float(np.sqrt(d2))
        except Exception:
            return float(np.linalg.norm(diff))

    def contextual_score(self, context: MatchContext, pattern_context: dict | None = None) -> float:
        """Contextual relevance ∈ [0, 1].

        Components (each in [0, 1]):
          - VIX regime match: 1.0 if regimes match, 0.5 otherwise
          - Time-of-day proximity: gaussian on minute distance to pattern TOD
          - Relative volume: clamp(volume, 0.5, 2.0) → [0, 1] linear
        """
        pc = pattern_context or {}
        # VIX regime
        target_regime = pc.get("vix_regime")
        regime_match = 1.0 if (target_regime is None or target_regime == context.vix_regime) else 0.5
        # Time of day
        target_tod = pc.get("time_of_day_minutes")
        if target_tod is None:
            tod_score = 1.0
        else:
            delta = abs(int(target_tod) - int(context.time_of_day_minutes))
            tod_score = float(np.exp(-(delta / 90.0) ** 2))  # 1.5h scale
        # Relative volume: 1.0 at "normal", taper either side
        vol = max(0.0, min(3.0, context.relative_volume))
        vol_score = max(0.0, 1.0 - abs(vol - 1.0) / 2.0)
        return float((regime_match + tod_score + vol_score) / 3.0)

    # ──────────────────────────────────────────────────────────────
    # Match
    # ──────────────────────────────────────────────────────────────

    def match(
        self,
        current_book_series: np.ndarray,
        current_features: np.ndarray,
        template_book_series: np.ndarray,
        template_features: np.ndarray,
        pattern_id: str,
        context: MatchContext,
        pattern_context: dict | None = None,
        primary_proba: float | None = None,
    ) -> MatchResult:
        """Full ensemble match against one pattern template."""
        cosine = self.cosine_similarity(current_book_series, template_book_series)
        # Map cosine ∈ [-1, 1] → score ∈ [0, 1]
        cos_score = (cosine + 1.0) / 2.0
        mahal_d = self.mahalanobis_distance(current_features, template_features)
        mahal_score = _distance_to_score(mahal_d)
        ctx_score = self.contextual_score(context, pattern_context)
        ensemble = (
            WEIGHT_TEMPLATE_MATCH * cos_score
            + WEIGHT_MAHALANOBIS * mahal_score
            + WEIGHT_CONTEXTUAL * ctx_score
        )
        return MatchResult(
            pattern_id=pattern_id,
            template_cosine_similarity=cosine,
            mahalanobis_distance=mahal_d,
            contextual_score=ctx_score,
            ensemble_score=float(ensemble),
            used_euclidean_fallback=self._use_euclidean,
            primary_proba=primary_proba,
        )

    def match_all(
        self,
        current_book_series: np.ndarray,
        current_features: np.ndarray,
        templates: dict[str, PatternTemplate] | None,
        context: MatchContext,
        min_score: float = 0.60,
    ) -> list[MatchResult]:
        """Match against all templates, return those above min_score, sorted desc."""
        templates = templates if templates is not None else self.templates
        out: list[MatchResult] = []
        for pid, tmpl in templates.items():
            r = self.match(
                current_book_series=current_book_series,
                current_features=current_features,
                template_book_series=tmpl.book_series,
                template_features=tmpl.feature_centroid,
                pattern_id=pid,
                context=context,
                pattern_context=tmpl.pattern_context,
            )
            if r.ensemble_score >= min_score:
                out.append(r)
        out.sort(key=lambda r: r.ensemble_score, reverse=True)
        return out
