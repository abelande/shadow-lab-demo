#pragma once
/**
 * P6 Core Types — zero-allocation-friendly structs for the hot path.
 * 
 * Design principles:
 * - POD-like structs for cache locality
 * - Fixed-size arrays where possible (no heap allocs in hot path)
 * - Enum classes for type safety
 */

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace p6 {

// ── Constants ────────────────────────────────────────────────────

constexpr int MAX_LEVELS = 20;
constexpr int MAX_ORDERS_PER_LEVEL = 256;
constexpr int MAX_EVENTS = 512;
constexpr int MAX_TRADES = 256;
constexpr int MAX_SPOOF_EVENTS = 64;
constexpr int FFT_WINDOW = 256;

// ── Enums ────────────────────────────────────────────────────────

enum class Side : uint8_t { BID = 0, ASK = 1 };

enum class OrderAction : uint8_t {
    ADD = 0,
    CANCEL = 1,
    MODIFY = 2,
    FILL = 3,
};

enum class FragilityState : uint8_t {
    SOLID = 0,
    MODERATE = 1,
    FRAGILE = 2,
};

enum class CupFlipState : uint8_t {
    BALANCED = 0,
    BULL_STREAK = 1,
    BEAR_STREAK = 2,
    BULL_STALL = 3,
    BEAR_STALL = 4,
    STOP_RUN = 5,
};

enum class FrequencyBand : uint8_t {
    INSTITUTIONAL = 0,
    FUND = 1,
    DAYTRADING = 2,
    HFT = 3,
};

enum class SpoofType : uint8_t {
    PULL_BEFORE_TOUCH = 0,
    LAYERING = 1,
    ICEBERG = 2,
    PHANTOM_WALL = 3,
    STUFFING = 4,
};

enum class RegimeType : uint8_t {
    TRENDING = 0,
    RANGING = 1,
    VOLATILE = 2,
    UNKNOWN = 3,
};

// ── Core Structs ─────────────────────────────────────────────────

struct Order {
    uint64_t order_id = 0;
    Side side = Side::BID;
    double price = 0.0;
    double size = 0.0;
    int64_t timestamp_ms = 0;
    OrderAction action = OrderAction::ADD;
    bool is_aggressive = false;
};

struct BookLevel {
    double price = 0.0;
    Side side = Side::BID;
    double volume = 0.0;
    int32_t order_count = 0;
    double avg_order_size = 0.0;
};

struct Snapshot {
    int64_t timestamp_ms = 0;
    std::array<BookLevel, MAX_LEVELS> bids{};
    std::array<BookLevel, MAX_LEVELS> asks{};
    int32_t num_bids = 0;
    int32_t num_asks = 0;
    double best_bid = 0.0;
    double best_ask = 0.0;
    double mid_price = 0.0;
    double spread = 0.0;
    // Recent events/trades (ring buffer indices)
    std::vector<Order> recent_events;
    std::vector<Order> recent_trades;
};

// ── Layer 1: Staircase Profile ──────────────────────────────────

struct LevelProfile {
    double price = 0.0;
    Side side = Side::BID;
    double volume = 0.0;
    int32_t order_count = 0;
    double avg_order_size = 0.0;
    double aggressive_ratio = 0.0;
    FragilityState fragility = FragilityState::MODERATE;
    double fragility_score = 0.0;
};

struct StaircaseProfile {
    int64_t timestamp_ms = 0;
    std::array<LevelProfile, MAX_LEVELS> bid_levels{};
    std::array<LevelProfile, MAX_LEVELS> ask_levels{};
    int32_t num_bid_levels = 0;
    int32_t num_ask_levels = 0;
    double bid_total_volume = 0.0;
    double ask_total_volume = 0.0;
    double imbalance_ratio = 0.0;
};

// ── Layer 2: Cup Flip / Game State ──────────────────────────────

struct GameState {
    CupFlipState state = CupFlipState::BALANCED;
    int32_t streak_length = 0;
    double streak_velocity = 0.0;
    int32_t streak_depth = 0;
    double pressure = 0.0;
    int32_t stall_count = 0;
    Side stop_run_side = Side::BID;
    bool has_stop_run = false;
    int64_t timestamp_ms = 0;
};

// ── Layer 3: Spectral Force ─────────────────────────────────────

struct BandEnergy {
    FrequencyBand band = FrequencyBand::HFT;
    double energy = 0.0;
    int8_t sign = 0;  // +1 buy, -1 sell, 0 neutral
    double weighted_force = 0.0;
};

struct ForceVector {
    double total_force = 0.0;
    std::array<BandEnergy, 4> bands{};
    double institutional_score = 0.0;
    FrequencyBand dominant_band = FrequencyBand::HFT;
    int64_t timestamp_ms = 0;
};

// ── Layer 4: Spoof Detection ────────────────────────────────────

struct SpoofEvent {
    SpoofType type = SpoofType::PULL_BEFORE_TOUCH;
    double price = 0.0;
    Side side = Side::BID;
    double confidence = 0.0;
    int64_t timestamp_ms = 0;
};

struct AuthenticityProfile {
    double authenticity_score = 1.0;
    double pull_score = 0.0;
    double layering_score = 0.0;
    double phantom_score = 0.0;
    double iceberg_score = 0.0;
    int32_t num_spoof_events = 0;
    std::array<SpoofEvent, MAX_SPOOF_EVENTS> spoof_events{};
    int64_t timestamp_ms = 0;
};

// ── Layer 5: Regime Context ─────────────────────────────────────

struct RegimeWeights {
    RegimeType regime = RegimeType::UNKNOWN;
    double l1_weight = 0.25;
    double l2_weight = 0.25;
    double l3_weight = 0.25;
    double l4_weight = 0.25;
    bool abstain = false;
};

// ── Aggregated Output ───────────────────────────────────────────

struct Signal {
    double direction = 0.0;     // -1 to +1
    double confidence = 0.0;    // 0-1
    double urgency = 0.0;       // 0-1
    double size_multiplier = 0.0;
    RegimeType regime = RegimeType::UNKNOWN;
    bool abstain = false;
    double l1_component = 0.0;
    double l2_component = 0.0;
    double l3_component = 0.0;
    double l4_component = 0.0;
    int64_t timestamp_ms = 0;
};

// ── Complete Frame ──────────────────────────────────────────────

struct Frame {
    int64_t timestamp_ms = 0;
    // Layer outputs
    StaircaseProfile staircase;
    GameState game_state;
    ForceVector force_vector;
    AuthenticityProfile authenticity;
    RegimeWeights regime_weights;
    Signal signal;
    // Book state for rendering
    Snapshot snapshot;
};

} // namespace p6
