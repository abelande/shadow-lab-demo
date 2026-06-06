"""
P6 Order Book Meta Model — Core Types
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import time


# ── Order Book Primitives ──────────────────────────────────────────

class Side(Enum):
    BID = "BID"
    ASK = "ASK"


class OrderAction(Enum):
    ADD = "ADD"
    CANCEL = "CANCEL"
    MODIFY = "MODIFY"
    FILL = "FILL"


@dataclass
class Order:
    order_id: str
    side: Side
    price: float
    size: float
    timestamp_ms: int
    action: OrderAction = OrderAction.ADD
    is_aggressive: bool = False  # crossed spread


@dataclass
class OrderBookLevel:
    price: float
    side: Side
    volume: float
    order_count: int
    orders: List[Order] = field(default_factory=list)

    @property
    def avg_order_size(self) -> float:
        return self.volume / self.order_count if self.order_count > 0 else 0.0


@dataclass
class OrderBookSnapshot:
    """Full L3 order book snapshot at a point in time."""
    timestamp_ms: int
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)  # sorted desc by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # sorted asc by price
    recent_trades: List[Order] = field(default_factory=list)
    recent_events: List[Order] = field(default_factory=list)  # ADD/CXL/MOD/FIL stream

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


# ── Layer 1: Staircase Profile ─────────────────────────────────────

class FragilityState(Enum):
    FRAGILE = "FRAGILE"  # few orders, large avg → single cancel = level gone
    SOLID = "SOLID"      # many orders, small avg → resilient
    MODERATE = "MODERATE"


@dataclass
class LevelProfile:
    price: float
    side: Side
    volume: float
    order_count: int
    avg_order_size: float
    aggressive_ratio: float  # fraction of volume placed aggressively
    fragility: FragilityState
    fragility_score: float   # 0 (solid) to 1 (fragile)


@dataclass
class StaircaseProfile:
    """Layer 1 output: 5-level staircase volume/count/fragility profile."""
    timestamp_ms: int
    bid_levels: List[LevelProfile] = field(default_factory=list)
    ask_levels: List[LevelProfile] = field(default_factory=list)
    bid_total_volume: float = 0.0
    ask_total_volume: float = 0.0
    imbalance_ratio: float = 0.0  # (bid - ask) / (bid + ask), range [-1, 1]


# ── Layer 2: Cup Flip / Game State ─────────────────────────────────

class CupFlipState(Enum):
    BALANCED = "BALANCED"
    BULL_STREAK = "BULL_STREAK"
    BEAR_STREAK = "BEAR_STREAK"
    BULL_STALL = "BULL_STALL"
    BEAR_STALL = "BEAR_STALL"
    STOP_RUN = "STOP_RUN"


@dataclass
class GameState:
    """Layer 2 output: cup flip state machine snapshot."""
    state: CupFlipState
    streak_length: int = 0
    streak_velocity: float = 0.0   # levels/sec cleared
    streak_depth: int = 0          # how many price levels consumed
    pressure: float = 0.0          # net directional pressure [-1, +1]
    stall_count: int = 0           # consecutive failed fills
    stop_run_side: Optional[Side] = None
    timestamp_ms: int = 0
    # Enriched Cup Flip signals (default 0 for backward compatibility)
    pressure_acceleration: float = 0.0   # energy ratio: >1 accelerating, <1 exhausting
    streak_exhaustion: float = 0.0       # 0 = fresh streak, 1 = fully exhausted
    state_confidence: float = 0.0        # blended threshold + markov confidence


# ── Layer 3: Spectral Force ────────────────────────────────────────

class FrequencyBand(Enum):
    INSTITUTIONAL = "INSTITUTIONAL"  # lowest freq
    FUND = "FUND"
    DAYTRADING = "DAYTRADING"
    HFT = "HFT"                      # highest freq


@dataclass
class BandEnergy:
    band: FrequencyBand
    energy: float
    sign: int  # +1 buy, -1 sell, 0 neutral
    weighted_force: float  # (1/f_k) * E_k * sign


@dataclass
class ForceVector:
    """Layer 3 output: spectral decomposition of volume delta."""
    total_force: float  # Σ[(1/f_k) × E_k × sign(Δv_k)]
    bands: List[BandEnergy] = field(default_factory=list)
    institutional_score: float = 0.0  # 0-1, dominance of institutional band
    dominant_band: Optional[FrequencyBand] = None
    timestamp_ms: int = 0


# ── Layer 4: Spoof / Authenticity Detection ────────────────────────

class SpoofType(Enum):
    PULL_BEFORE_TOUCH = "PULL_BEFORE_TOUCH"
    LAYERING = "LAYERING"
    ICEBERG = "ICEBERG"
    PHANTOM_WALL = "PHANTOM_WALL"
    STUFFING = "STUFFING"


@dataclass
class SpoofEvent:
    spoof_type: SpoofType
    price: float
    side: Side
    confidence: float  # 0-1
    timestamp_ms: int
    details: str = ""


@dataclass
class AuthenticityProfile:
    """Layer 4 output: how real is this order book?"""
    authenticity_score: float  # 0 (all fake) to 1 (all real)
    spoof_events: List[SpoofEvent] = field(default_factory=list)
    pull_score: float = 0.0      # 0-1, pull-before-touch severity
    layering_score: float = 0.0  # 0-1
    phantom_score: float = 0.0   # 0-1
    stuffing_score: float = 0.0  # 0-1
    timestamp_ms: int = 0


# ── Layer 5: Regime Context ────────────────────────────────────────

class RegimeType(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeWeights:
    """Per-regime layer weights for aggregation."""
    regime: RegimeType
    l1_weight: float  # staircase
    l2_weight: float  # cup flip
    l3_weight: float  # spectral force
    l4_weight: float  # spoof detection
    abstain: bool = False


# ── Depth Indicator (Visual Output) ───────────────────────────────

@dataclass
class DepthBar:
    """Single bar in the depth indicator visualization."""
    price: float
    side: Side
    volume: float
    order_count: int
    cumulative_volume: float
    bar_length: float  # normalized 0-1
    is_round_number: bool = False
    authenticity: float = 1.0  # 0 = likely spoofed, 1 = authentic
    lifecycle: Optional["LevelLifecycle"] = None
    significance: float = 0.0
    spoof_type: Optional["SpoofType"] = None
    iceberg_suspected: bool = False


@dataclass
class DOMRow:
    """Single row in the DOM panel."""
    price: float
    volume: float
    order_count: int
    cumulative_volume: float
    side: Side


@dataclass
class TapeEntry:
    """Single trade on the tape feed."""
    timestamp_ms: int
    side: Side
    price: float
    size: float


@dataclass
class StatsSnapshot:
    """Header stats for the depth indicator."""
    cvd: float              # cumulative volume delta
    trades_per_sec: float
    add_count: int
    cancel_count: int
    modify_count: int
    fill_count: int
    live_orders: int


@dataclass
class DepthIndicatorFrame:
    """Complete depth indicator output — everything needed to render one frame."""
    timestamp_ms: int
    symbol: str
    # Bars
    bid_bars: List[DepthBar] = field(default_factory=list)
    ask_bars: List[DepthBar] = field(default_factory=list)
    # DOM
    dom_rows: List[DOMRow] = field(default_factory=list)
    # Tape
    tape: List[TapeEntry] = field(default_factory=list)
    # Stats
    stats: Optional[StatsSnapshot] = None
    # Layer outputs
    staircase: Optional[StaircaseProfile] = None
    game_state: Optional[GameState] = None
    force_vector: Optional[ForceVector] = None
    authenticity: Optional[AuthenticityProfile] = None
    regime_weights: Optional[RegimeWeights] = None
    # Aggregated signal
    # Level states from LevelTracker
    level_states: List["LevelState"] = field(default_factory=list)
    # Aggregated signal
    direction: float = 0.0        # -1 (strong sell) to +1 (strong buy)
    confidence: float = 0.0       # 0-1
    urgency: float = 0.0          # 0-1
    size_multiplier: float = 1.0  # position sizing hint
    # Microstructure metrics (from OFI tracker)
    ofi: float = 0.0              # Order Flow Imbalance (EMA-smoothed)
    vpin: float = 0.0             # Volume-sync'd P(Informed Trading)
    # Wave 5 Phase 5A — thesis-chain wire: L6 pattern correlation matches
    # emitted by p6lab's CorrelationEngine at ``match_interval_ms`` cadence.
    # Each dict carries pattern_id / tier / direction / expected_move_atr /
    # ensemble_score / regime / match_window timestamps. Empty when the
    # engine is not wired (default) or mid-cadence.
    correlation_matches: List[Dict] = field(default_factory=list)


# ── Level Tracker (Renovation Part 1) ────────────────────────────

class LevelLifecycle(Enum):
    FORMING = "FORMING"
    RESTING = "RESTING"
    TESTED = "TESTED"
    DEFENDED = "DEFENDED"
    BROKEN = "BROKEN"
    PULLED = "PULLED"


@dataclass
class LevelState:
    """State of a single significant price level."""
    price: float
    side: Side
    volume: float
    peak_volume: float
    order_count: int
    lifecycle: LevelLifecycle
    first_seen_ms: int
    last_seen_ms: int
    age_ms: int
    significance: float
    authenticity: float
    spoof_type: Optional[SpoofType]
    iceberg_suspected: bool
    volume_history: List[float] = field(default_factory=list)
    fill_count: int = 0
    refill_count: int = 0


@dataclass
class InstrumentVisualConfig:
    """Per-instrument thresholds for level significance and display."""
    symbol: str
    tick_size: float
    significant_volume: float
    significant_age_ms: int
    significant_order_count: int
    round_number_step: float
    zone_merge_ticks: int
    level_fade_candles: int = 3

    @classmethod
    def for_symbol(cls, symbol: str) -> "InstrumentVisualConfig":
        """Return a config preset for a known symbol."""
        presets: Dict[str, "InstrumentVisualConfig"] = {
            "NQ": cls("NQ", 0.25, 50.0, 5000, 5, 25.0, 4),
            "ES": cls("ES", 0.25, 100.0, 8000, 8, 10.0, 4),
            "CL": cls("CL", 0.01, 30.0, 5000, 5, 0.50, 4),
            "GC": cls("GC", 0.10, 50.0, 5000, 5, 10.0, 4),
            "SI": cls("SI", 0.005, 40.0, 5000, 5, 0.50, 4),
        }
        for key, cfg in presets.items():
            if key in symbol.upper():
                return cfg
        return cls("DEFAULT", 0.25, 50.0, 5000, 5, 25.0, 4)


@dataclass
class ReplayCandle:
    """OHLCV candle for replay."""
    time: int       # Unix timestamp (seconds)
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TapeSummary:
    """Per-candle tape digest."""
    buy_volume: float
    sell_volume: float
    delta: float
    largest_fill_price: Optional[float]
    largest_fill_size: float
    largest_fill_side: Optional[str]
    iceberg_hits: int
    cancel_count_bid: int
    cancel_count_ask: int


@dataclass
class ReplayFrame:
    """Complete replay frame emitted once per candle."""
    candle: ReplayCandle
    levels: List[LevelState]
    game_state: Optional["GameState"]
    force_vector: Optional["ForceVector"]
    spoof_events: List[SpoofEvent]
    tape_summary: Optional[TapeSummary]
    timestamp_ms: int


# ── Aggregated Output ─────────────────────────────────────────────

@dataclass
class AggregatedSignal:
    """Final cross-layer aggregated signal."""
    direction: float        # -1 to +1
    confidence: float       # 0-1
    urgency: float          # 0-1
    size_multiplier: float  # >= 0
    regime: RegimeType = RegimeType.UNKNOWN
    abstain: bool = False
    components: Dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = 0
