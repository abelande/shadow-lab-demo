#include "p6/orderbook.hpp"
#include "p6/types.hpp"
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

// ── Tests ──────────────────────────────────────────────────────────────────────

void test_add_cancel() {
    OrderBook book(10);

    book.apply(make_order(1, Side::BID, 100.0, 10.0, 1000, OrderAction::ADD));
    assert(book.total_orders() == 1);

    auto snap = book.build_snapshot(1000, {}, {});
    assert(snap.num_bids == 1);
    assert(snap.bids[0].price == 100.0);
    assert(std::abs(snap.bids[0].volume - 10.0) < 1e-9);
    assert(snap.bids[0].order_count == 1);
    assert(snap.best_bid == 100.0);

    book.apply(make_order(1, Side::BID, 100.0, 10.0, 1001, OrderAction::CANCEL));
    assert(book.total_orders() == 0);

    snap = book.build_snapshot(1001, {}, {});
    assert(snap.num_bids == 0);

    std::cout << "test_add_cancel: PASS\n";
}

void test_bid_ask_sorted() {
    OrderBook book(10);

    book.apply(make_order(1, Side::BID, 100.0, 10.0, 1000, OrderAction::ADD));
    book.apply(make_order(2, Side::BID,  99.0,  5.0, 1000, OrderAction::ADD));
    book.apply(make_order(3, Side::BID, 101.0,  8.0, 1000, OrderAction::ADD));
    book.apply(make_order(4, Side::ASK, 102.0,  3.0, 1000, OrderAction::ADD));
    book.apply(make_order(5, Side::ASK, 103.0,  7.0, 1000, OrderAction::ADD));

    auto snap = book.build_snapshot(1000, {}, {});
    assert(snap.num_bids == 3);
    assert(snap.num_asks == 2);

    // Bids highest first
    assert(snap.bids[0].price == 101.0);
    assert(snap.bids[1].price == 100.0);
    assert(snap.bids[2].price ==  99.0);

    // Asks lowest first
    assert(snap.asks[0].price == 102.0);
    assert(snap.asks[1].price == 103.0);

    assert(snap.best_bid == 101.0);
    assert(snap.best_ask == 102.0);
    assert(std::abs(snap.mid_price - 101.5) < 1e-9);
    assert(std::abs(snap.spread    -   1.0) < 1e-9);

    std::cout << "test_bid_ask_sorted: PASS\n";
}

void test_modify_same_price() {
    OrderBook book(10);

    book.apply(make_order(1, Side::BID, 100.0, 10.0, 1000, OrderAction::ADD));
    book.apply(make_order(1, Side::BID, 100.0, 15.0, 1001, OrderAction::MODIFY));

    auto snap = book.build_snapshot(1001, {}, {});
    assert(snap.num_bids == 1);
    assert(std::abs(snap.bids[0].volume - 15.0) < 1e-9);

    std::cout << "test_modify_same_price: PASS\n";
}

void test_modify_price_change() {
    OrderBook book(10);

    book.apply(make_order(1, Side::BID, 100.0, 10.0, 1000, OrderAction::ADD));
    // Move from 100 to 99
    book.apply(make_order(1, Side::BID,  99.0, 10.0, 1001, OrderAction::MODIFY));

    auto snap = book.build_snapshot(1001, {}, {});
    assert(snap.num_bids == 1);
    assert(snap.bids[0].price == 99.0);

    std::cout << "test_modify_price_change: PASS\n";
}

void test_partial_fill() {
    OrderBook book(10);

    book.apply(make_order(1, Side::BID, 100.0, 10.0, 1000, OrderAction::ADD));
    book.apply(make_order(1, Side::BID, 100.0,  4.0, 1001, OrderAction::FILL));
    assert(book.total_orders() == 1);

    auto snap = book.build_snapshot(1001, {}, {});
    assert(std::abs(snap.bids[0].volume - 6.0) < 1e-9);

    // Full fill
    book.apply(make_order(1, Side::BID, 100.0, 6.0, 1002, OrderAction::FILL));
    assert(book.total_orders() == 0);

    snap = book.build_snapshot(1002, {}, {});
    assert(snap.num_bids == 0);

    std::cout << "test_partial_fill: PASS\n";
}

void test_multiple_orders_per_level() {
    OrderBook book(10);

    book.apply(make_order(1, Side::ASK, 105.0, 3.0, 1000, OrderAction::ADD));
    book.apply(make_order(2, Side::ASK, 105.0, 5.0, 1001, OrderAction::ADD));
    book.apply(make_order(3, Side::ASK, 105.0, 2.0, 1002, OrderAction::ADD));

    auto snap = book.build_snapshot(1002, {}, {});
    assert(snap.num_asks == 1);
    assert(snap.asks[0].order_count == 3);
    assert(std::abs(snap.asks[0].volume - 10.0) < 1e-9);
    assert(std::abs(snap.asks[0].avg_order_size - 10.0 / 3.0) < 1e-9);

    std::cout << "test_multiple_orders_per_level: PASS\n";
}

void test_level_cap() {
    OrderBook book(3);

    for (int i = 1; i <= 5; ++i) {
        book.apply(make_order(i, Side::BID, 100.0 - i, 10.0, 1000, OrderAction::ADD));
    }

    auto snap = book.build_snapshot(1000, {}, {});
    assert(snap.num_bids == 3);
    assert(snap.bids[0].price == 99.0);
    assert(snap.bids[1].price == 98.0);
    assert(snap.bids[2].price == 97.0);

    std::cout << "test_level_cap: PASS\n";
}

void test_clear() {
    OrderBook book(10);
    book.apply(make_order(1, Side::BID, 100.0, 5.0, 1000, OrderAction::ADD));
    book.apply(make_order(2, Side::ASK, 101.0, 3.0, 1000, OrderAction::ADD));
    book.clear();
    assert(book.total_orders() == 0);
    auto snap = book.build_snapshot(1000, {}, {});
    assert(snap.num_bids == 0);
    assert(snap.num_asks == 0);
    std::cout << "test_clear: PASS\n";
}

void test_recent_events_passthrough() {
    OrderBook book(10);
    book.apply(make_order(1, Side::BID, 100.0, 5.0, 1000, OrderAction::ADD));

    std::vector<Order> events = {make_order(99, Side::ASK, 101.0, 2.0, 1001, OrderAction::FILL)};
    auto snap = book.build_snapshot(1001, events, {});
    assert(snap.recent_events.size() == 1);
    assert(snap.recent_events[0].order_id == 99);
    std::cout << "test_recent_events_passthrough: PASS\n";
}

int main() {
    std::cout << "=== OrderBook Tests ===\n";
    test_add_cancel();
    test_bid_ask_sorted();
    test_modify_same_price();
    test_modify_price_change();
    test_partial_fill();
    test_multiple_orders_per_level();
    test_level_cap();
    test_clear();
    test_recent_events_passthrough();
    std::cout << "All OrderBook tests PASSED.\n";
    return 0;
}
