p6lab/validation/precision_grid.py
─────────────────────────────────
@dataclass(frozen=True)
class GridConfig:
    horizons_ms: tuple[int, ...]
    barrier_ticks: tuple[float, ...]
    percentiles: tuple[float, ...]
    min_n_signals: int = 10              # don't report cells below this
    cv_folds: int = 10
    cv_purge_rows: int = 660             # horizon + safety
    
@dataclass(frozen=True)
class GridCellResult:
    horizon_ms: int
    barrier_ticks: float
    percentile: float
    threshold: float
    n_signals: int
    precision: float
    positive_rate_baseline: float
    lift: float
    auc: float                           # within-cell AUC for sanity
    folds_run: int

@dataclass(frozen=True)
class PrecisionGridReport:
    cells: list[GridCellResult]
    config: GridConfig
    best_for_scalp: GridCellResult | None     # short-horizon ultra-selective
    best_for_swing: GridCellResult | None     # long-horizon moderate
    best_for_market_making: GridCellResult | None  # short-horizon broad
    notes: str

def evaluate_precision_grid(
    X: pd.DataFrame,
    mid_series: pd.Series,
    timestamps_ms: np.ndarray,
    config: GridConfig,
    model_factory: Callable[[], Any] = make_default_lgbm,
) -> PrecisionGridReport:
    """For each (horizon, barrier) pair: train a fresh model on that label,
    do CPCV, aggregate OOF predictions, evaluate at each percentile."""
    ...
