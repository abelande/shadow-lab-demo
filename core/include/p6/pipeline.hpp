#pragma once
/**
 * P6 Pipeline — runs all 5 layers and produces a Frame.
 */

#include "p6/types.hpp"
#include "p6/orderbook.hpp"

namespace p6 {

// ── Layer interfaces ─────────────────────────────────────────────

class StaircaseAnalyzer {
public:
    StaircaseProfile analyze(const Snapshot& snap) const;
};

class CupFlipTracker {
public:
    CupFlipTracker() = default;
    explicit CupFlipTracker(int levels_threshold, int gap_tolerance = 1)
        : levels_threshold_(levels_threshold), gap_tolerance_(gap_tolerance) {}

    GameState update(const std::vector<Order>& events, int64_t timestamp_ms,
                     double best_bid = 0.0, double best_ask = 0.0);
private:
    GameState state_{};
    int last_fill_side_ = 0;  // +1 bid, -1 ask
    int consecutive_ = 0;
    int gap_count_ = 0;       // opposing fills absorbed (gap tolerance)
    int gap_tolerance_ = 1;   // max opposing fills before streak reset
    int levels_threshold_ = 5; // stop-run levels (5 for NQ, 3 for ES)
    double velocity_sum_ = 0.0;
    int64_t last_event_ts_ = 0;
};

class SpectralForce {
public:
    ForceVector compute(const std::vector<Order>& trades, int64_t timestamp_ms);
private:
    std::array<double, FFT_WINDOW> volume_delta_buf_{};
    int buf_pos_ = 0;
};

class SpoofDetector {
public:
    struct Config {
        // Pull before touch
        int pull_threshold_ms = 150;
        double pull_min_size = 3.0;
        int pull_min_repeats = 2;
        // Layering
        int layer_min_levels = 3;
        double layer_size_tolerance = 0.03;
        int layer_max_time_spread_ms = 300;
        // Phantom wall
        double phantom_min_size = 50.0;
        double phantom_approach_ticks = 2.0;
        int phantom_cancel_ms = 500;
        // Iceberg
        int iceberg_min_fills = 4;
        double iceberg_max_visible = 8.0;
        double iceberg_min_total = 30.0;
        // Authenticity
        double auth_risk_exponent = 1.5;
        double auth_floor = 0.15;
    };

    SpoofDetector();
    explicit SpoofDetector(Config config);
    AuthenticityProfile detect(const std::vector<Order>& events,
                               double best_bid, double best_ask,
                               double mid_price, int64_t timestamp_ms);
private:
    Config config_;
};

class RegimeClassifier {
public:
    RegimeWeights classify(RegimeType hint = RegimeType::UNKNOWN);
};

class SignalAggregator {
public:
    struct Config {
        double streak_floor = 0.4;
        double force_squash_denom = 1.0;
        double auth_center = 0.5;
        double auth_scale = 2.0;
        double confidence_base = 0.6;
        double confidence_auth_weight = 0.4;
        double urgency_pressure_weight = 0.5;
        double urgency_force_weight = 0.5;
        double urgency_force_cap = 10.0;
        double size_base = 0.5;
        double size_scale = 1.5;
        double min_confidence = 0.2;
        double min_authenticity = 0.3;
    };

    SignalAggregator();
    explicit SignalAggregator(Config config);
    Signal aggregate(const StaircaseProfile& staircase,
                     const GameState& game_state,
                     const ForceVector& force,
                     const AuthenticityProfile& auth,
                     const RegimeWeights& regime,
                     int64_t timestamp_ms);
private:
    Config config_;
};

// ── Main Pipeline ────────────────────────────────────────────────

class Pipeline {
public:
    Pipeline();
    
    /// Process a snapshot through all 5 layers
    Frame process(const Snapshot& snapshot, RegimeType regime_hint = RegimeType::UNKNOWN);
    
private:
    StaircaseAnalyzer staircase_;
    CupFlipTracker cup_flip_;
    SpectralForce spectral_;
    SpoofDetector spoof_;
    RegimeClassifier regime_;
    SignalAggregator aggregator_;
};

} // namespace p6
