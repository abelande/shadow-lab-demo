#include "p6/pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace p6 {

// Matches Python pressure_scorer.py weights:
//   FILL ASK (buy aggressor)   = +1.4
//   FILL BID (sell aggressor)  = -1.4
//   CANCEL BID (opens downside)= sell pressure (0.6)
//   CANCEL ASK (opens upside)  = buy pressure (0.6)
//   Aggressive ADD BID         = +0.8
//   Aggressive ADD ASK         = -0.8
//
// Parity patches (March 2026):
//   - Time decay: events weighted by recency (exp decay, half-life ~500ms)
//   - Gap tolerance: 1 opposing fill absorbed into streak before reset
//   - Stall detection: aggressive-add via price-crossing (BID >= best_ask)
//   - pressure_enter raised 0.25 → 0.40
//   - stop_run levels_threshold configurable (default 5 for NQ)
//
// State machine (priority order):
//   1. Stop-run  : levels_threshold+ unique prices consumed in one direction
//   2. Stall     : aggressive adds failing to convert (price-crossing detection)
//   3. Streak    : 3+ consecutive same-side fills (with gap_tolerance=1)
//   4. Pressure  : |pressure| >= 0.40 -> bull/bear streak
//   5. Neutral   : |pressure| < 0.10 -> BALANCED

GameState CupFlipTracker::update(const std::vector<Order>& events,
                                  int64_t timestamp_ms,
                                  double best_bid,
                                  double best_ask) {
    state_.timestamp_ms = timestamp_ms;

    if (events.empty()) return state_;

    double net_weight = 0.0;
    double total_weight = 0.0;

    // Collect fills for streak tracking
    std::vector<const Order*> fills;
    // Track unique prices per direction for stop-run detection
    std::unordered_set<double> bull_prices, bear_prices;
    // Stall detection: aggressive adds that don't convert
    int aggressive_attempts = 0;
    int fill_count = 0;

    // Use the last event timestamp as reference for time decay
    int64_t t_ref = events.back().timestamp_ms;

    for (const auto& ev : events) {
        double w = 0.0, sign = 0.0;

        // Time decay: half-life ~500ms, so events 1s ago get ~0.25 weight
        double age_ms = static_cast<double>(t_ref - ev.timestamp_ms);
        double decay = std::exp(-0.693 * age_ms / 500.0);  // ln(2) ≈ 0.693

        if (ev.action == OrderAction::FILL) {
            fills.push_back(&ev);
            fill_count++;
            if (ev.side == Side::ASK) {  // ASK fill = buy aggressor
                w = 1.4; sign = +1.0;
                bull_prices.insert(ev.price);
            } else {                     // BID fill = sell aggressor
                w = 1.4; sign = -1.0;
                bear_prices.insert(ev.price);
            }
        } else if (ev.action == OrderAction::CANCEL) {
            if (ev.side == Side::ASK) { w = 0.6; sign = +1.0; }  // ask removed = bullish
            else                      { w = 0.6; sign = -1.0; }  // bid removed = bearish
        } else if (ev.action == OrderAction::ADD || ev.action == OrderAction::MODIFY) {
            // Aggressive-add detection via price-crossing (not is_aggressive flag)
            bool is_agg = false;
            if (best_bid > 0 && best_ask > 0) {
                if (ev.side == Side::BID && ev.price >= best_ask) is_agg = true;
                if (ev.side == Side::ASK && ev.price <= best_bid) is_agg = true;
            } else {
                is_agg = ev.is_aggressive;  // fallback
            }
            if (is_agg) {
                aggressive_attempts++;
                if (ev.side == Side::BID) { w = 0.8; sign = +1.0; }
                else                      { w = 0.8; sign = -1.0; }
            }
        }

        net_weight   += sign * w * decay;
        total_weight += w * decay;
    }

    double pressure = (total_weight > 0.0) ? net_weight / total_weight : 0.0;
    pressure = std::clamp(pressure, -1.0, 1.0);
    state_.pressure = pressure;

    // Process fills with gap tolerance (1 opposing fill absorbed)
    for (const Order* fill : fills) {
        int fill_side = (fill->side == Side::ASK) ? +1 : -1;
        if (fill_side == last_fill_side_) {
            consecutive_++;
            gap_count_ = 0;  // reset gap counter on same-side fill
        } else {
            gap_count_++;
            if (gap_count_ > gap_tolerance_) {
                // Gap exceeded — reset streak
                last_fill_side_ = fill_side;
                consecutive_ = 1;
                gap_count_ = 0;
            }
            // else: absorb the opposing fill, don't break streak
        }
    }

    // Streak velocity: approximate levels/sec using consecutive count over time window
    double streak_vel = 0.0;
    if (last_event_ts_ > 0 && consecutive_ >= 3) {
        int64_t dt_ms = timestamp_ms - last_event_ts_;
        if (dt_ms > 0 && dt_ms < 10000) {
            streak_vel = static_cast<double>(consecutive_) / (dt_ms / 1000.0);
        }
    }

    // Stop-run: configurable levels_threshold (default 5 for NQ, 3 for ES)
    bool stop_run = (bull_prices.size() >= static_cast<size_t>(levels_threshold_) ||
                     bear_prices.size() >= static_cast<size_t>(levels_threshold_));
    Side stop_side = (bull_prices.size() >= bear_prices.size()) ? Side::ASK : Side::BID;

    // Stall detection: aggressive adds with low fill conversion
    bool stall_detected = false;
    if (aggressive_attempts >= 3) {
        int failed = aggressive_attempts - fill_count;
        if (failed >= 3) stall_detected = true;
    }

    // State machine transitions
    CupFlipState new_state = state_.state;

    if (stop_run) {
        new_state = CupFlipState::STOP_RUN;
        state_.has_stop_run = true;
        state_.stop_run_side = stop_side;
    } else if (stall_detected) {
        // Stall via aggressive-add failure detection
        if (pressure >= 0) {
            new_state = CupFlipState::BULL_STALL;
        } else {
            new_state = CupFlipState::BEAR_STALL;
        }
        ++state_.stall_count;
    } else if (state_.state == CupFlipState::BULL_STREAK && pressure < 0.10) {
        new_state = CupFlipState::BULL_STALL;
        ++state_.stall_count;
    } else if (state_.state == CupFlipState::BEAR_STREAK && pressure > -0.10) {
        new_state = CupFlipState::BEAR_STALL;
        ++state_.stall_count;
    } else if (consecutive_ >= 3 && last_fill_side_ == +1) {
        new_state = CupFlipState::BULL_STREAK;
    } else if (consecutive_ >= 3 && last_fill_side_ == -1) {
        new_state = CupFlipState::BEAR_STREAK;
    } else if (pressure >= 0.40) {   // was 0.25 — raised to reduce false streaks
        new_state = CupFlipState::BULL_STREAK;
    } else if (pressure <= -0.40) {
        new_state = CupFlipState::BEAR_STREAK;
    } else if (std::abs(pressure) < 0.10 && state_.state != CupFlipState::STOP_RUN) {
        new_state = CupFlipState::BALANCED;
    }

    state_.state          = new_state;
    state_.streak_length  = consecutive_;
    state_.streak_velocity = streak_vel;
    state_.streak_depth   = static_cast<int32_t>(
        std::max(bull_prices.size(), bear_prices.size()));

    if (!fills.empty()) {
        last_event_ts_ = timestamp_ms;
    }

    return state_;
}

} // namespace p6
