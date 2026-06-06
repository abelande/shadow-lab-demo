#include "p6/pipeline.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace p6 {

// Match Python fragility_scorer.py logic:
//   count_component = 1.0 / (log(order_count) + 1)   -- high for few orders
//   concentration   = 1.0 / order_count               -- approximation: avg_size / vol = 1/count
//   below_median    = 1.0 if order_count < median_count else 0.0
//   score = 0.5 * count_component + 0.2 * concentration + 0.3 * below_median
//   FRAGILE  if score >= 0.65
//   SOLID    if score <= 0.35
//   MODERATE otherwise
static double score_level(int order_count, double median_count) {
    int n = std::max(1, order_count);
    double count_component = 1.0 / (std::log(static_cast<double>(n)) + 1.0);
    double concentration   = 1.0 / static_cast<double>(n);
    double below_median    = (order_count < median_count) ? 1.0 : 0.0;
    double score = 0.5 * count_component + 0.2 * concentration + 0.3 * below_median;
    return std::min(1.0, score);
}

static FragilityState classify(double score) {
    if (score >= 0.65) return FragilityState::FRAGILE;
    if (score <= 0.35) return FragilityState::SOLID;
    return FragilityState::MODERATE;
}

StaircaseProfile StaircaseAnalyzer::analyze(const Snapshot& snap) const {
    StaircaseProfile profile;
    profile.timestamp_ms = snap.timestamp_ms;

    // Compute median order count across all levels for below-median penalty
    std::vector<int32_t> counts;
    counts.reserve(snap.num_bids + snap.num_asks);
    for (int i = 0; i < snap.num_bids; ++i) counts.push_back(snap.bids[i].order_count);
    for (int i = 0; i < snap.num_asks; ++i) counts.push_back(snap.asks[i].order_count);

    double median_count = 1.0;
    if (!counts.empty()) {
        std::sort(counts.begin(), counts.end());
        size_t n = counts.size();
        median_count = (n % 2 == 0)
            ? (counts[n / 2 - 1] + counts[n / 2]) / 2.0
            : static_cast<double>(counts[n / 2]);
    }

    // Process bid levels
    profile.num_bid_levels = snap.num_bids;
    for (int i = 0; i < snap.num_bids; ++i) {
        const BookLevel& bl = snap.bids[i];
        LevelProfile& lp = profile.bid_levels[i];
        lp.price         = bl.price;
        lp.side          = Side::BID;
        lp.volume        = bl.volume;
        lp.order_count   = bl.order_count;
        lp.avg_order_size = bl.avg_order_size;

        double fs        = score_level(bl.order_count, median_count);
        lp.fragility_score = fs;
        lp.fragility     = classify(fs);

        // Aggressive ratio: fraction of recent events at this price that are aggressive
        int agg_cnt = 0, total_cnt = 0;
        for (const auto& ev : snap.recent_events) {
            if (ev.side == Side::BID && std::abs(ev.price - bl.price) < 1e-9) {
                ++total_cnt;
                if (ev.is_aggressive) ++agg_cnt;
            }
        }
        lp.aggressive_ratio = (total_cnt > 0) ? static_cast<double>(agg_cnt) / total_cnt : 0.0;

        profile.bid_total_volume += bl.volume;
    }

    // Process ask levels
    profile.num_ask_levels = snap.num_asks;
    for (int i = 0; i < snap.num_asks; ++i) {
        const BookLevel& bl = snap.asks[i];
        LevelProfile& lp = profile.ask_levels[i];
        lp.price         = bl.price;
        lp.side          = Side::ASK;
        lp.volume        = bl.volume;
        lp.order_count   = bl.order_count;
        lp.avg_order_size = bl.avg_order_size;

        double fs        = score_level(bl.order_count, median_count);
        lp.fragility_score = fs;
        lp.fragility     = classify(fs);

        int agg_cnt = 0, total_cnt = 0;
        for (const auto& ev : snap.recent_events) {
            if (ev.side == Side::ASK && std::abs(ev.price - bl.price) < 1e-9) {
                ++total_cnt;
                if (ev.is_aggressive) ++agg_cnt;
            }
        }
        lp.aggressive_ratio = (total_cnt > 0) ? static_cast<double>(agg_cnt) / total_cnt : 0.0;

        profile.ask_total_volume += bl.volume;
    }

    // imbalance_ratio = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    double total_vol = profile.bid_total_volume + profile.ask_total_volume;
    if (total_vol > 0.0) {
        profile.imbalance_ratio =
            (profile.bid_total_volume - profile.ask_total_volume) / total_vol;
    }

    return profile;
}

} // namespace p6
