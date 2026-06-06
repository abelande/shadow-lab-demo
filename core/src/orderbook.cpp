#include "p6/orderbook.hpp"
#include <algorithm>
#include <cmath>

namespace p6 {

void PriceLevel::recalc(const std::unordered_map<uint64_t, Order>& all_orders,
                        const std::vector<uint64_t>& order_ids) {
    total_volume = 0.0;
    order_count = 0;
    for (auto id : order_ids) {
        auto it = all_orders.find(id);
        if (it != all_orders.end()) {
            total_volume += it->second.size;
            ++order_count;
        }
    }
}

OrderBook::OrderBook(int num_levels) : num_levels_(num_levels) {}

void OrderBook::apply(const Order& order) {
    switch (order.action) {
        case OrderAction::ADD:    add_order(order);    break;
        case OrderAction::CANCEL: cancel_order(order); break;
        case OrderAction::MODIFY: modify_order(order); break;
        case OrderAction::FILL:   fill_order(order);   break;
    }
}

void OrderBook::add_order(const Order& order) {
    orders_[order.order_id] = order;
    auto& levels = (order.side == Side::BID) ? bid_levels_ : ask_levels_;
    levels[order.price].push_back(order.order_id);
}

void OrderBook::remove_from_level(uint64_t order_id, double price, Side side) {
    auto& levels = (side == Side::BID) ? bid_levels_ : ask_levels_;
    auto it = levels.find(price);
    if (it != levels.end()) {
        auto& ids = it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), order_id), ids.end());
        if (ids.empty()) {
            levels.erase(it);
        }
    }
}

void OrderBook::cancel_order(const Order& order) {
    auto it = orders_.find(order.order_id);
    if (it == orders_.end()) return;
    const Order& existing = it->second;
    remove_from_level(order.order_id, existing.price, existing.side);
    orders_.erase(it);
}

void OrderBook::modify_order(const Order& order) {
    auto it = orders_.find(order.order_id);
    if (it == orders_.end()) {
        // Unknown order: treat as add
        add_order(order);
        return;
    }
    Order& existing = it->second;
    if (existing.price != order.price) {
        // Price changed: move to new level
        remove_from_level(order.order_id, existing.price, existing.side);
        auto& levels = (existing.side == Side::BID) ? bid_levels_ : ask_levels_;
        levels[order.price].push_back(order.order_id);
    }
    existing.size = order.size;
    existing.price = order.price;
    existing.timestamp_ms = order.timestamp_ms;
}

void OrderBook::fill_order(const Order& order) {
    auto it = orders_.find(order.order_id);
    if (it == orders_.end()) return;
    Order& existing = it->second;
    existing.size -= order.size;
    if (existing.size <= 1e-9) {
        remove_from_level(order.order_id, existing.price, existing.side);
        orders_.erase(it);
    }
}

Snapshot OrderBook::build_snapshot(int64_t timestamp_ms,
                                    const std::vector<Order>& recent_events,
                                    const std::vector<Order>& recent_trades) const {
    Snapshot snap;
    snap.timestamp_ms = timestamp_ms;
    snap.recent_events = recent_events;
    snap.recent_trades = recent_trades;

    // Bids: highest first (reverse iteration)
    int bid_count = 0;
    for (auto it = bid_levels_.rbegin(); it != bid_levels_.rend() && bid_count < num_levels_; ++it) {
        double vol = 0.0;
        int cnt = 0;
        for (auto id : it->second) {
            auto oit = orders_.find(id);
            if (oit != orders_.end()) {
                vol += oit->second.size;
                ++cnt;
            }
        }
        if (cnt == 0) continue;
        BookLevel& bl = snap.bids[bid_count++];
        bl.price = it->first;
        bl.side = Side::BID;
        bl.volume = vol;
        bl.order_count = cnt;
        bl.avg_order_size = vol / cnt;
    }
    snap.num_bids = bid_count;

    // Asks: lowest first (forward iteration)
    int ask_count = 0;
    for (auto it = ask_levels_.begin(); it != ask_levels_.end() && ask_count < num_levels_; ++it) {
        double vol = 0.0;
        int cnt = 0;
        for (auto id : it->second) {
            auto oit = orders_.find(id);
            if (oit != orders_.end()) {
                vol += oit->second.size;
                ++cnt;
            }
        }
        if (cnt == 0) continue;
        BookLevel& bl = snap.asks[ask_count++];
        bl.price = it->first;
        bl.side = Side::ASK;
        bl.volume = vol;
        bl.order_count = cnt;
        bl.avg_order_size = vol / cnt;
    }
    snap.num_asks = ask_count;

    if (snap.num_bids > 0) snap.best_bid = snap.bids[0].price;
    if (snap.num_asks > 0) snap.best_ask = snap.asks[0].price;
    if (snap.num_bids > 0 && snap.num_asks > 0) {
        snap.mid_price = (snap.best_bid + snap.best_ask) / 2.0;
        snap.spread = snap.best_ask - snap.best_bid;
    }
    return snap;
}

void OrderBook::clear() {
    orders_.clear();
    bid_levels_.clear();
    ask_levels_.clear();
}

} // namespace p6
