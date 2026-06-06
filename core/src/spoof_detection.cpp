#include "p6/pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace p6 {

SpoofDetector::SpoofDetector() : config_(Config{}) {}
SpoofDetector::SpoofDetector(Config config) : config_(config) {}

// Helper to push a SpoofEvent if capacity allows
static void push_event(std::vector<SpoofEvent>& vec, SpoofType type,
                       double price, Side side, double confidence,
                       int64_t ts) {
    if (static_cast<int>(vec.size()) >= MAX_SPOOF_EVENTS) return;
    SpoofEvent se;
    se.type        = type;
    se.price       = price;
    se.side        = side;
    se.confidence  = std::clamp(confidence, 0.0, 1.0);
    se.timestamp_ms = ts;
    vec.push_back(se);
}

AuthenticityProfile SpoofDetector::detect(const std::vector<Order>& events,
                                           double best_bid, double best_ask,
                                           double /*mid_price*/,
                                           int64_t timestamp_ms) {
    AuthenticityProfile auth;
    auth.timestamp_ms = timestamp_ms;

    if (events.empty()) return auth;

    std::vector<SpoofEvent> spoof_events;

    // ── PULL BEFORE TOUCH ─────────────────────────────────────────────────────
    // Detect: ADD at best bid/ask, then CANCEL within threshold_ms.
    // Requires min_repeats per side.
    // Confidence: 0.5*speed_conf + 0.5*repeat_conf
    {
        std::unordered_map<uint64_t, const Order*> pending;
        for (const auto& ev : events) {
            if (ev.action == OrderAction::ADD && ev.size >= config_.pull_min_size) {
                bool at_best = (ev.side == Side::BID && std::abs(ev.price - best_bid) < 1e-9) ||
                               (ev.side == Side::ASK && std::abs(ev.price - best_ask) < 1e-9);
                if (at_best) pending[ev.order_id] = &ev;
            }
        }

        std::unordered_map<int, int> pull_counts;  // side (0=BID,1=ASK) -> count
        for (const auto& ev : events) {
            if (ev.action != OrderAction::CANCEL) continue;
            auto it = pending.find(ev.order_id);
            if (it == pending.end()) continue;
            const Order* add = it->second;
            int64_t dt = ev.timestamp_ms - add->timestamp_ms;
            if (dt < 0 || dt > config_.pull_threshold_ms) continue;

            int key = static_cast<int>(add->side);
            ++pull_counts[key];
            int repeats = pull_counts[key];
            if (repeats < config_.pull_min_repeats) continue;

            double speed_conf  = 1.0 - static_cast<double>(dt) / config_.pull_threshold_ms;
            double repeat_conf = std::min(1.0, static_cast<double>(repeats - config_.pull_min_repeats + 1) / 3.0);
            double conf = 0.5 * speed_conf + 0.5 * repeat_conf;
            push_event(spoof_events, SpoofType::PULL_BEFORE_TOUCH,
                       add->price, add->side, conf, ev.timestamp_ms);
        }
    }

    // ── LAYERING ──────────────────────────────────────────────────────────────
    // Detect: same-size ADD orders across min_levels consecutive price levels,
    // all placed within max_time_spread_ms.
    // Confidence: 0.6*level_conf + 0.4*time_conf
    auto detect_layering = [&](Side side) {
        std::vector<const Order*> adds;
        for (const auto& ev : events) {
            if (ev.action == OrderAction::ADD && ev.side == side)
                adds.push_back(&ev);
        }
        if (static_cast<int>(adds.size()) < config_.layer_min_levels) return;

        // Sort by price
        std::sort(adds.begin(), adds.end(),
                  [](const Order* a, const Order* b) { return a->price < b->price; });

        for (size_t i = 0; i + config_.layer_min_levels <= adds.size(); ++i) {
            double ref_size = adds[i]->size;
            int64_t min_ts  = adds[i]->timestamp_ms;
            int64_t max_ts  = adds[i]->timestamp_ms;
            int count = 1;

            for (size_t j = i + 1; j < adds.size(); ++j) {
                double rel_diff = std::abs(adds[j]->size - ref_size) /
                                  std::max(ref_size, 1e-9);
                if (rel_diff > config_.layer_size_tolerance) break;
                ++count;
                min_ts = std::min(min_ts, adds[j]->timestamp_ms);
                max_ts = std::max(max_ts, adds[j]->timestamp_ms);

                if (count >= config_.layer_min_levels) {
                    int64_t spread = max_ts - min_ts;
                    if (spread <= config_.layer_max_time_spread_ms) {
                        double level_conf = std::min(1.0, static_cast<double>(
                            count - config_.layer_min_levels + 1) / 3.0);
                        double time_conf  = 1.0 - static_cast<double>(spread) /
                                            config_.layer_max_time_spread_ms;
                        double conf = 0.6 * level_conf + 0.4 * time_conf;
                        push_event(spoof_events, SpoofType::LAYERING,
                                   adds[i]->price, side, conf, max_ts);
                    }
                    break;
                }
            }
        }
    };
    detect_layering(Side::BID);
    detect_layering(Side::ASK);

    // ── PHANTOM WALL ──────────────────────────────────────────────────────────
    // Detect: large order (>= phantom_min_size) that gets CANCEL'd within
    // cancel_ms, while price has approached within approach_ticks.
    // Confidence: 0.4*size_conf + 0.3*speed_conf + 0.3*prox_conf
    {
        std::unordered_map<uint64_t, const Order*> large_adds;
        for (const auto& ev : events) {
            if (ev.action == OrderAction::ADD && ev.size >= config_.phantom_min_size)
                large_adds[ev.order_id] = &ev;
        }

        for (const auto& ev : events) {
            if (ev.action != OrderAction::CANCEL) continue;
            auto it = large_adds.find(ev.order_id);
            if (it == large_adds.end()) continue;
            const Order* add = it->second;

            int64_t duration_ms = ev.timestamp_ms - add->timestamp_ms;
            if (duration_ms < 100 || duration_ms > config_.phantom_cancel_ms) continue;

            // How close is the wall to current best price?
            double price_dist = (add->side == Side::BID)
                ? (best_bid - add->price)      // BID wall below best_bid
                : (add->price - best_ask);     // ASK wall above best_ask

            if (price_dist < 0.0 || price_dist > config_.phantom_approach_ticks) continue;

            double size_conf  = std::min(1.0,
                (add->size - config_.phantom_min_size) / (config_.phantom_min_size * 2.0));
            double speed_conf = 1.0 - static_cast<double>(duration_ms) / config_.phantom_cancel_ms;
            double prox_conf  = 1.0 - price_dist / config_.phantom_approach_ticks;
            double conf = 0.4 * size_conf + 0.3 * speed_conf + 0.3 * prox_conf;

            push_event(spoof_events, SpoofType::PHANTOM_WALL,
                       add->price, add->side, conf, ev.timestamp_ms);
        }
    }

    // ── ICEBERG ───────────────────────────────────────────────────────────────
    // Detect: repeated small FILL events at the same price suggesting hidden qty.
    // Config: min_fills=4, max_visible=8.0, min_total=30.0
    // Confidence: 0.5*fill_conf + 0.5*hidden_conf
    {
        std::unordered_map<double, std::vector<const Order*>> fills_by_price;
        for (const auto& ev : events) {
            if (ev.action == OrderAction::FILL && ev.size <= config_.iceberg_max_visible)
                fills_by_price[ev.price].push_back(&ev);
        }

        for (auto& [price, fills] : fills_by_price) {
            if (static_cast<int>(fills.size()) < config_.iceberg_min_fills) continue;
            double total_vol = 0.0;
            for (const auto* f : fills) total_vol += f->size;
            if (total_vol < config_.iceberg_min_total) continue;

            double fill_conf   = std::min(1.0, static_cast<double>(
                fills.size() - config_.iceberg_min_fills + 1) / 4.0);
            double hidden_conf = std::min(1.0,
                (total_vol - config_.iceberg_min_total) / config_.iceberg_min_total);
            double conf = 0.5 * fill_conf + 0.5 * hidden_conf;

            push_event(spoof_events, SpoofType::ICEBERG,
                       price, fills.back()->side, conf,
                       fills.back()->timestamp_ms);
        }
    }

    // ── AUTHENTICITY SCORING ──────────────────────────────────────────────────
    // Max confidence per type, then weighted spoof_risk.
    // Weights (matches authenticity_scorer.py): pull=0.35, layer=0.30, phantom=0.25, iceberg=0.10
    // Decay curve: authenticity = 1 - spoof_risk^(1/exponent)
    // Floor: authenticity >= auth_floor (0.15)
    double max_conf[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (const auto& se : spoof_events) {
        int t = static_cast<int>(se.type);
        if (t < 5) max_conf[t] = std::max(max_conf[t], se.confidence);
    }

    auth.pull_score      = max_conf[static_cast<int>(SpoofType::PULL_BEFORE_TOUCH)];
    auth.layering_score  = max_conf[static_cast<int>(SpoofType::LAYERING)];
    auth.phantom_score   = max_conf[static_cast<int>(SpoofType::PHANTOM_WALL)];
    auth.iceberg_score   = max_conf[static_cast<int>(SpoofType::ICEBERG)];

    double spoof_risk = 0.35 * auth.pull_score
                      + 0.30 * auth.layering_score
                      + 0.25 * auth.phantom_score
                      + 0.10 * auth.iceberg_score;

    double raw_auth = (spoof_risk > 0.0)
        ? 1.0 - std::pow(spoof_risk, 1.0 / config_.auth_risk_exponent)
        : 1.0;
    auth.authenticity_score = std::max(config_.auth_floor, raw_auth);

    // Store spoof events
    auth.num_spoof_events = static_cast<int32_t>(
        std::min(spoof_events.size(), static_cast<size_t>(MAX_SPOOF_EVENTS)));
    for (int i = 0; i < auth.num_spoof_events; ++i) {
        auth.spoof_events[i] = spoof_events[i];
    }

    return auth;
}

} // namespace p6
