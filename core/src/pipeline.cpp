#include "p6/pipeline.hpp"

namespace p6 {

Pipeline::Pipeline()
    : staircase_(), cup_flip_(), spectral_(), spoof_(), regime_(), aggregator_() {}

Frame Pipeline::process(const Snapshot& snapshot, RegimeType regime_hint) {
    Frame frame;
    frame.timestamp_ms = snapshot.timestamp_ms;
    frame.snapshot     = snapshot;

    // L1: Staircase — volume/fragility profile per price level
    frame.staircase = staircase_.analyze(snapshot);

    // L2: Cup Flip — directional game-state machine
    frame.game_state = cup_flip_.update(snapshot.recent_events, snapshot.timestamp_ms,
                                         snapshot.best_bid, snapshot.best_ask);

    // L3: Spectral Force — FFT-based frequency band decomposition
    frame.force_vector = spectral_.compute(snapshot.recent_trades, snapshot.timestamp_ms);

    // L4: Spoof Detection — authenticity scoring
    frame.authenticity = spoof_.detect(
        snapshot.recent_events,
        snapshot.best_bid, snapshot.best_ask,
        snapshot.mid_price,
        snapshot.timestamp_ms);

    // L5: Regime Context
    frame.regime_weights = regime_.classify(regime_hint);

    // Aggregation
    frame.signal = aggregator_.aggregate(
        frame.staircase,
        frame.game_state,
        frame.force_vector,
        frame.authenticity,
        frame.regime_weights,
        snapshot.timestamp_ms);

    return frame;
}

} // namespace p6
