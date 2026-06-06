#pragma once
/**
 * L3 Order Book — high-performance order book with individual order tracking.
 * 
 * Uses a flat map (price -> PriceLevel) with sorted price iteration.
 * Individual orders stored in unordered_map for O(1) lookup by order_id.
 */

#include "p6/types.hpp"
#include <map>
#include <unordered_map>

namespace p6 {

struct PriceLevel {
    double price = 0.0;
    Side side = Side::BID;
    double total_volume = 0.0;
    int32_t order_count = 0;
    
    void recalc(const std::unordered_map<uint64_t, Order>& all_orders,
                const std::vector<uint64_t>& order_ids);
};

class OrderBook {
public:
    explicit OrderBook(int num_levels = 10);
    
    /// Apply a single order event (ADD/CANCEL/MODIFY/FILL)
    void apply(const Order& order);
    
    /// Build a Snapshot from current book state
    Snapshot build_snapshot(int64_t timestamp_ms,
                           const std::vector<Order>& recent_events,
                           const std::vector<Order>& recent_trades) const;
    
    /// Reset the book
    void clear();
    
    /// Stats
    size_t total_orders() const { return orders_.size(); }
    
private:
    int num_levels_;
    
    // order_id -> Order
    std::unordered_map<uint64_t, Order> orders_;
    
    // price -> list of order_ids (sorted map for bid desc / ask asc iteration)
    // Bids: reverse iteration (highest first)
    // Asks: forward iteration (lowest first)
    std::map<double, std::vector<uint64_t>> bid_levels_;
    std::map<double, std::vector<uint64_t>> ask_levels_;
    
    void add_order(const Order& order);
    void cancel_order(const Order& order);
    void modify_order(const Order& order);
    void fill_order(const Order& order);
    
    void remove_from_level(uint64_t order_id, double price, Side side);
};

} // namespace p6
