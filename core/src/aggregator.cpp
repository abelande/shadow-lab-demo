#include "p6/pipeline.hpp"
#include <algorithm>
#include <cmath>

namespace p6 {

// ── RegimeClassifier ──────────────────────────────────────────────────────────
// Assigns per-regime layer weights.  Weights always sum to 1.0.
//   TRENDING  : L2 (cup flip) and L3 (spectral force) dominate — momentum play
//   RANGING   : L1 (staircase) and L4 (spoof) dominate — mean-reversion / fake-out
//   VOLATILE  : L3 and L4 dominate — risk / deception aware
//   UNKNOWN   : equal 0.25 each

RegimeWeights RegimeClassifier::classify(RegimeType hint) {
    RegimeWeights rw;
    rw.regime = hint;

    switch (hint) {
        case RegimeType::TRENDING:
            rw.l1_weight = 0.15;
            rw.l2_weight = 0.35;
            rw.l3_weight = 0.35;
            rw.l4_weight = 0.15;
            rw.abstain   = false;
            break;
        case RegimeType::RANGING:
            rw.l1_weight = 0.35;
            rw.l2_weight = 0.15;
            rw.l3_weight = 0.15;
            rw.l4_weight = 0.35;
            rw.abstain   = false;
            break;
        case RegimeType::VOLATILE:
            rw.l1_weight = 0.15;
            rw.l2_weight = 0.20;
            rw.l3_weight = 0.30;
            rw.l4_weight = 0.35;
            rw.abstain   = false;
            break;
        case RegimeType::UNKNOWN:
        default:
            rw.l1_weight = 0.25;
            rw.l2_weight = 0.25;
            rw.l3_weight = 0.25;
            rw.l4_weight = 0.25;
            rw.abstain   = false;
            break;
    }
    return rw;
}

// ── SignalAggregator ──────────────────────────────────────────────────────────
// Matches Python aggregator.py formulas exactly:
//
//   L1 direction = imbalance_ratio
//   L2 direction = pressure, floored to ±streak_floor when in BULL/BEAR_STREAK
//   L3 direction = total_force / (|total_force| + force_squash_denom)
//   L4 direction = (auth_score - auth_center) * auth_scale   clamped [-1,1]
//
//   direction = Σ weight_i * comp_i   clamped [-1, 1]
//
//   confidence = |raw_direction| * (confidence_base + confidence_auth_weight * auth_score)
//   urgency    = pressure_weight * |pressure| + force_weight * min(1, |force|/force_cap)
//   size_mult  = size_base + size_scale * confidence * auth_score  (or 0 if abstain)
//   abstain    = regime.abstain OR confidence < min_confidence OR auth < min_authenticity

SignalAggregator::SignalAggregator() : config_(Config{}) {}
SignalAggregator::SignalAggregator(Config config) : config_(config) {}

Signal SignalAggregator::aggregate(const StaircaseProfile& staircase,
                                   const GameState& game_state,
                                   const ForceVector& force,
                                   const AuthenticityProfile& auth,
                                   const RegimeWeights& regime,
                                   int64_t timestamp_ms) {
    Signal sig;
    sig.timestamp_ms = timestamp_ms;
    sig.regime       = regime.regime;

    // L1: staircase imbalance ratio is already in [-1, 1]
    double l1 = staircase.imbalance_ratio;

    // L2: pressure with streak floor
    double l2 = game_state.pressure;
    if (game_state.state == CupFlipState::BULL_STREAK)
        l2 = std::max(l2, config_.streak_floor);
    else if (game_state.state == CupFlipState::BEAR_STREAK)
        l2 = std::min(l2, -config_.streak_floor);

    // L3: squashed force (soft-sign via x/(|x|+d))
    double fv   = force.total_force;
    double l3   = fv / (std::abs(fv) + config_.force_squash_denom);

    // L4: authenticity mapped to direction
    double l4 = std::clamp(
        (auth.authenticity_score - config_.auth_center) * config_.auth_scale,
        -1.0, 1.0);

    sig.l1_component = l1;
    sig.l2_component = l2;
    sig.l3_component = l3;
    sig.l4_component = l4;

    // Weighted direction
    double raw = regime.l1_weight * l1
               + regime.l2_weight * l2
               + regime.l3_weight * l3
               + regime.l4_weight * l4;

    sig.direction = std::clamp(raw, -1.0, 1.0);

    // Confidence: conviction (|raw|) × (base + auth_weight × auth_score)
    double conviction = std::abs(raw);
    sig.confidence = std::clamp(
        conviction * (config_.confidence_base +
                      config_.confidence_auth_weight * auth.authenticity_score),
        0.0, 1.0);

    // Urgency: pressure term + spectral force term
    sig.urgency = std::clamp(
        config_.urgency_pressure_weight * std::abs(game_state.pressure) +
        config_.urgency_force_weight    * std::min(1.0, std::abs(fv) / config_.urgency_force_cap),
        0.0, 1.0);

    // Abstain logic
    bool should_abstain = regime.abstain
                       || sig.confidence < config_.min_confidence
                       || auth.authenticity_score < config_.min_authenticity;
    sig.abstain = should_abstain;
    sig.size_multiplier = should_abstain ? 0.0
        : config_.size_base + config_.size_scale * sig.confidence * auth.authenticity_score;

    return sig;
}

} // namespace p6
