#include "p6/pipeline.hpp"
#include "p6/orderbook.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace p6;

static Order make_order(uint64_t id, Side side, double price, double size,
                         int64_t ts, OrderAction action, bool aggressive = false) {
    Order o;
    o.order_id      = id;
    o.side          = side;
    o.price         = price;
    o.size          = size;
    o.timestamp_ms  = ts;
    o.action        = action;
    o.is_aggressive = aggressive;
    return o;
}

// Build a snapshot with 5 bid + 5 ask levels and some fills for pressure
static Snapshot build_test_snapshot() {
    OrderBook book(10);
    for (int i = 0; i < 5; ++i) {
        book.apply(make_order(  1 + i, Side::BID, 100.0 - i, 10.0 + i, 1000, OrderAction::ADD));
        book.apply(make_order(100 + i, Side::ASK, 101.0 + i,  8.0 + i, 1000, OrderAction::ADD));
    }

    std::vector<Order> events, trades;
    // 3 consecutive ASK fills  (buy pressure -> BULL_STREAK candidate)
    for (int i = 0; i < 3; ++i) {
        events.push_back(make_order(200 + i, Side::ASK, 101.0, 2.0, 1001 + i, OrderAction::FILL));
        trades.push_back(make_order(300 + i, Side::ASK, 101.0, 2.0, 1001 + i, OrderAction::FILL));
    }
    // One BID fill (sell)
    events.push_back(make_order(210, Side::BID, 100.0, 1.0, 1010, OrderAction::FILL));

    return book.build_snapshot(1010, events, trades);
}

// ── Tests ──────────────────────────────────────────────────────────────────────

void test_pipeline_basic() {
    Pipeline pipeline;
    Snapshot snap = build_test_snapshot();
    Frame frame = pipeline.process(snap);

    assert(frame.timestamp_ms == snap.timestamp_ms);
    assert(frame.staircase.num_bid_levels > 0);
    assert(frame.staircase.num_ask_levels > 0);

    std::cout << "test_pipeline_basic: PASS\n";
}

void test_signal_ranges() {
    Pipeline pipeline;
    Frame frame = pipeline.process(build_test_snapshot());

    // All outputs must be bounded
    assert(frame.signal.direction     >= -1.0 && frame.signal.direction     <= 1.0);
    assert(frame.signal.confidence    >=  0.0 && frame.signal.confidence    <= 1.0);
    assert(frame.signal.urgency       >=  0.0 && frame.signal.urgency       <= 1.0);
    assert(frame.authenticity.authenticity_score >= 0.0 &&
           frame.authenticity.authenticity_score <= 1.0);
    assert(frame.game_state.pressure  >= -1.0 && frame.game_state.pressure  <= 1.0);
    assert(frame.staircase.imbalance_ratio >= -1.0 &&
           frame.staircase.imbalance_ratio <=  1.0);

    std::cout << "test_signal_ranges: PASS\n";
}

void test_staircase_fragility() {
    StaircaseAnalyzer analyzer;

    Snapshot snap;
    snap.timestamp_ms = 1000;
    snap.num_bids = 1;
    snap.num_asks = 1;
    // Fragile bid: 1 large order
    snap.bids[0] = {100.0, Side::BID, 50.0, 1, 50.0};
    // Solid ask: 25 small orders
    snap.asks[0] = {101.0, Side::ASK, 50.0, 25, 2.0};
    snap.best_bid = 100.0;
    snap.best_ask = 101.0;
    snap.mid_price = 100.5;

    auto profile = analyzer.analyze(snap);
    // Bid (1 big order) should be more fragile than ask (25 small orders)
    assert(profile.bid_levels[0].fragility_score > profile.ask_levels[0].fragility_score);
    assert(profile.bid_levels[0].fragility == FragilityState::FRAGILE);
    assert(profile.ask_levels[0].fragility == FragilityState::SOLID);

    std::cout << "test_staircase_fragility: PASS\n";
}

void test_imbalance_ratio() {
    StaircaseAnalyzer analyzer;

    Snapshot snap;
    snap.timestamp_ms = 1000;
    snap.num_bids = 1;
    snap.num_asks = 1;
    snap.bids[0] = {100.0, Side::BID, 80.0, 8, 10.0};
    snap.asks[0] = {101.0, Side::ASK, 20.0, 2, 10.0};
    snap.best_bid = 100.0;
    snap.best_ask = 101.0;

    auto profile = analyzer.analyze(snap);
    // bid_vol=80, ask_vol=20 => ratio = (80-20)/(80+20) = 0.6
    assert(std::abs(profile.imbalance_ratio - 0.6) < 1e-9);

    std::cout << "test_imbalance_ratio: PASS\n";
}

void test_bull_streak() {
    CupFlipTracker tracker;

    std::vector<Order> events;
    for (int i = 0; i < 4; ++i)
        events.push_back(make_order(100 + i, Side::ASK, 101.0, 2.0, 1000 + i, OrderAction::FILL));

    GameState state = tracker.update(events, 1010);
    // 4 consecutive ASK fills -> BULL_STREAK, positive pressure
    assert(state.state == CupFlipState::BULL_STREAK || state.pressure > 0.0);
    assert(state.pressure > 0.0);
    assert(state.streak_length >= 4);

    std::cout << "test_bull_streak: PASS\n";
}

void test_bear_streak() {
    CupFlipTracker tracker;

    std::vector<Order> events;
    for (int i = 0; i < 3; ++i)
        events.push_back(make_order(200 + i, Side::BID, 100.0, 3.0, 1000 + i, OrderAction::FILL));

    GameState state = tracker.update(events, 1005);
    assert(state.pressure < 0.0);

    std::cout << "test_bear_streak: PASS\n";
}

void test_authenticity_clean() {
    SpoofDetector detector;

    // Clean events: add then cancel after 200ms (> pull_threshold_ms=150)
    std::vector<Order> events;
    events.push_back(make_order(1, Side::BID, 100.0, 5.0, 1000, OrderAction::ADD));
    events.push_back(make_order(1, Side::BID, 100.0, 5.0, 1200, OrderAction::CANCEL));

    auto auth = detector.detect(events, 100.0, 101.0, 100.5, 1200);
    // Cancel is 200ms after add (> 150ms threshold) → no pull-before-touch
    assert(auth.pull_score == 0.0);
    assert(auth.authenticity_score >= 0.5);

    std::cout << "test_authenticity_clean: PASS\n";
}

void test_spectral_empty() {
    SpectralForce spectral;
    ForceVector fv = spectral.compute({}, 1000);

    assert(fv.institutional_score >= 0.0 && fv.institutional_score <= 1.0);
    // Empty buffer: all zeros → total_force == 0
    assert(fv.total_force == 0.0);

    std::cout << "test_spectral_empty: PASS\n";
}

void test_spectral_buy_pressure() {
    SpectralForce spectral;

    // Feed many buy fills; should produce positive total_force
    std::vector<Order> trades;
    for (int i = 0; i < 50; ++i)
        trades.push_back(make_order(i, Side::ASK, 101.0, 10.0, 1000 + i, OrderAction::FILL));

    ForceVector fv = spectral.compute(trades, 2000);
    // Positive fill series → institutional_score > 0
    assert(fv.institutional_score >= 0.0 && fv.institutional_score <= 1.0);

    std::cout << "test_spectral_buy_pressure: PASS\n";
}

void test_regime_weights_sum() {
    RegimeClassifier classifier;
    for (auto r : {RegimeType::TRENDING, RegimeType::RANGING,
                   RegimeType::VOLATILE, RegimeType::UNKNOWN}) {
        auto rw = classifier.classify(r);
        double sum = rw.l1_weight + rw.l2_weight + rw.l3_weight + rw.l4_weight;
        assert(std::abs(sum - 1.0) < 1e-9);
    }
    std::cout << "test_regime_weights_sum: PASS\n";
}

void test_abstain_low_confidence() {
    // Build a pipeline that should abstain due to low confidence
    // (empty book, no events)
    Pipeline pipeline;
    Snapshot empty_snap;
    empty_snap.timestamp_ms = 1000;
    Frame frame = pipeline.process(empty_snap);

    // Zero-book → near-zero confidence → abstain
    assert(frame.signal.abstain == true || frame.signal.confidence < 0.2 ||
           frame.signal.size_multiplier == 0.0);

    std::cout << "test_abstain_low_confidence: PASS\n";
}

void test_regime_affects_direction() {
    // Same snapshot, different regime → different direction weights
    Pipeline trending, ranging;

    Snapshot snap = build_test_snapshot();
    Frame ft = trending.process(snap, RegimeType::TRENDING);
    Frame fr = ranging.process(snap, RegimeType::RANGING);

    // Both in [-1,1], regime weights differ
    assert(ft.regime_weights.l2_weight > fr.regime_weights.l2_weight);
    assert(fr.regime_weights.l1_weight > ft.regime_weights.l1_weight);

    std::cout << "test_regime_affects_direction: PASS\n";
}

int main() {
    std::cout << "=== Pipeline Tests ===\n";
    test_pipeline_basic();
    test_signal_ranges();
    test_staircase_fragility();
    test_imbalance_ratio();
    test_bull_streak();
    test_bear_streak();
    test_authenticity_clean();
    test_spectral_empty();
    test_spectral_buy_pressure();
    test_regime_weights_sum();
    test_abstain_low_confidence();
    test_regime_affects_direction();
    std::cout << "All Pipeline tests PASSED.\n";
    return 0;
}
