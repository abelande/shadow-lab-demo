#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "p6/types.hpp"
#include "p6/orderbook.hpp"
#include "p6/pipeline.hpp"

namespace py = pybind11;
using namespace p6;

PYBIND11_MODULE(_p6_core, m) {
    m.doc() = "P6 C++ core engine — zero-allocation order book, 5-layer analysis pipeline.";

    // ── Enums ──────────────────────────────────────────────────────────────────
    py::enum_<Side>(m, "Side")
        .value("BID", Side::BID)
        .value("ASK", Side::ASK)
        .export_values();

    py::enum_<OrderAction>(m, "OrderAction")
        .value("ADD",    OrderAction::ADD)
        .value("CANCEL", OrderAction::CANCEL)
        .value("MODIFY", OrderAction::MODIFY)
        .value("FILL",   OrderAction::FILL)
        .export_values();

    py::enum_<FragilityState>(m, "FragilityState")
        .value("SOLID",    FragilityState::SOLID)
        .value("MODERATE", FragilityState::MODERATE)
        .value("FRAGILE",  FragilityState::FRAGILE)
        .export_values();

    py::enum_<CupFlipState>(m, "CupFlipState")
        .value("BALANCED",    CupFlipState::BALANCED)
        .value("BULL_STREAK", CupFlipState::BULL_STREAK)
        .value("BEAR_STREAK", CupFlipState::BEAR_STREAK)
        .value("BULL_STALL",  CupFlipState::BULL_STALL)
        .value("BEAR_STALL",  CupFlipState::BEAR_STALL)
        .value("STOP_RUN",    CupFlipState::STOP_RUN)
        .export_values();

    py::enum_<FrequencyBand>(m, "FrequencyBand")
        .value("INSTITUTIONAL", FrequencyBand::INSTITUTIONAL)
        .value("FUND",          FrequencyBand::FUND)
        .value("DAYTRADING",    FrequencyBand::DAYTRADING)
        .value("HFT",           FrequencyBand::HFT)
        .export_values();

    py::enum_<SpoofType>(m, "SpoofType")
        .value("PULL_BEFORE_TOUCH", SpoofType::PULL_BEFORE_TOUCH)
        .value("LAYERING",          SpoofType::LAYERING)
        .value("ICEBERG",           SpoofType::ICEBERG)
        .value("PHANTOM_WALL",      SpoofType::PHANTOM_WALL)
        .value("STUFFING",          SpoofType::STUFFING)
        .export_values();

    py::enum_<RegimeType>(m, "RegimeType")
        .value("TRENDING", RegimeType::TRENDING)
        .value("RANGING",  RegimeType::RANGING)
        .value("VOLATILE", RegimeType::VOLATILE)
        .value("UNKNOWN",  RegimeType::UNKNOWN)
        .export_values();

    // ── Order ──────────────────────────────────────────────────────────────────
    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id",     &Order::order_id)
        .def_readwrite("side",         &Order::side)
        .def_readwrite("price",        &Order::price)
        .def_readwrite("size",         &Order::size)
        .def_readwrite("timestamp_ms", &Order::timestamp_ms)
        .def_readwrite("action",       &Order::action)
        .def_readwrite("is_aggressive",&Order::is_aggressive);

    // ── BookLevel ──────────────────────────────────────────────────────────────
    py::class_<BookLevel>(m, "BookLevel")
        .def(py::init<>())
        .def_readonly("price",          &BookLevel::price)
        .def_readonly("side",           &BookLevel::side)
        .def_readonly("volume",         &BookLevel::volume)
        .def_readonly("order_count",    &BookLevel::order_count)
        .def_readonly("avg_order_size", &BookLevel::avg_order_size);

    // ── Snapshot ───────────────────────────────────────────────────────────────
    py::class_<Snapshot>(m, "Snapshot")
        .def(py::init<>())
        .def_readwrite("timestamp_ms",   &Snapshot::timestamp_ms)
        .def_readwrite("recent_events",  &Snapshot::recent_events)
        .def_readwrite("recent_trades",  &Snapshot::recent_trades)
        .def_readonly("best_bid",        &Snapshot::best_bid)
        .def_readonly("best_ask",        &Snapshot::best_ask)
        .def_readonly("mid_price",       &Snapshot::mid_price)
        .def_readonly("spread",          &Snapshot::spread)
        .def_readonly("num_bids",        &Snapshot::num_bids)
        .def_readonly("num_asks",        &Snapshot::num_asks)
        .def("bids", [](const Snapshot& s) {
            std::vector<BookLevel> v;
            for (int i = 0; i < s.num_bids; ++i) v.push_back(s.bids[i]);
            return v;
        })
        .def("asks", [](const Snapshot& s) {
            std::vector<BookLevel> v;
            for (int i = 0; i < s.num_asks; ++i) v.push_back(s.asks[i]);
            return v;
        });

    // ── OrderBook ──────────────────────────────────────────────────────────────
    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<int>(), py::arg("num_levels") = 10)
        .def("apply",          &OrderBook::apply)
        .def("build_snapshot", &OrderBook::build_snapshot,
             py::arg("timestamp_ms"),
             py::arg("recent_events"),
             py::arg("recent_trades"))
        .def("clear",          &OrderBook::clear)
        .def("total_orders",   &OrderBook::total_orders);

    // ── LevelProfile ───────────────────────────────────────────────────────────
    py::class_<LevelProfile>(m, "LevelProfile")
        .def(py::init<>())
        .def_readonly("price",            &LevelProfile::price)
        .def_readonly("side",             &LevelProfile::side)
        .def_readonly("volume",           &LevelProfile::volume)
        .def_readonly("order_count",      &LevelProfile::order_count)
        .def_readonly("avg_order_size",   &LevelProfile::avg_order_size)
        .def_readonly("aggressive_ratio", &LevelProfile::aggressive_ratio)
        .def_readonly("fragility",        &LevelProfile::fragility)
        .def_readonly("fragility_score",  &LevelProfile::fragility_score);

    // ── StaircaseProfile ───────────────────────────────────────────────────────
    py::class_<StaircaseProfile>(m, "StaircaseProfile")
        .def(py::init<>())
        .def_readonly("timestamp_ms",     &StaircaseProfile::timestamp_ms)
        .def_readonly("num_bid_levels",   &StaircaseProfile::num_bid_levels)
        .def_readonly("num_ask_levels",   &StaircaseProfile::num_ask_levels)
        .def_readonly("bid_total_volume", &StaircaseProfile::bid_total_volume)
        .def_readonly("ask_total_volume", &StaircaseProfile::ask_total_volume)
        .def_readonly("imbalance_ratio",  &StaircaseProfile::imbalance_ratio)
        .def("bid_levels", [](const StaircaseProfile& sp) {
            std::vector<LevelProfile> v;
            for (int i = 0; i < sp.num_bid_levels; ++i) v.push_back(sp.bid_levels[i]);
            return v;
        })
        .def("ask_levels", [](const StaircaseProfile& sp) {
            std::vector<LevelProfile> v;
            for (int i = 0; i < sp.num_ask_levels; ++i) v.push_back(sp.ask_levels[i]);
            return v;
        });

    // ── GameState ──────────────────────────────────────────────────────────────
    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def_readonly("state",          &GameState::state)
        .def_readonly("streak_length",  &GameState::streak_length)
        .def_readonly("streak_velocity",&GameState::streak_velocity)
        .def_readonly("streak_depth",   &GameState::streak_depth)
        .def_readonly("pressure",       &GameState::pressure)
        .def_readonly("stall_count",    &GameState::stall_count)
        .def_readonly("stop_run_side",  &GameState::stop_run_side)
        .def_readonly("has_stop_run",   &GameState::has_stop_run)
        .def_readonly("timestamp_ms",   &GameState::timestamp_ms);

    // ── BandEnergy ─────────────────────────────────────────────────────────────
    py::class_<BandEnergy>(m, "BandEnergy")
        .def(py::init<>())
        .def_readonly("band",           &BandEnergy::band)
        .def_readonly("energy",         &BandEnergy::energy)
        .def_readonly("sign",           &BandEnergy::sign)
        .def_readonly("weighted_force", &BandEnergy::weighted_force);

    // ── ForceVector ────────────────────────────────────────────────────────────
    py::class_<ForceVector>(m, "ForceVector")
        .def(py::init<>())
        .def_readonly("total_force",         &ForceVector::total_force)
        .def_readonly("institutional_score", &ForceVector::institutional_score)
        .def_readonly("dominant_band",       &ForceVector::dominant_band)
        .def_readonly("timestamp_ms",        &ForceVector::timestamp_ms)
        .def("bands", [](const ForceVector& fv) {
            return std::vector<BandEnergy>(fv.bands.begin(), fv.bands.end());
        });

    // ── SpoofEvent ─────────────────────────────────────────────────────────────
    py::class_<SpoofEvent>(m, "SpoofEvent")
        .def(py::init<>())
        .def_readonly("type",         &SpoofEvent::type)
        .def_readonly("price",        &SpoofEvent::price)
        .def_readonly("side",         &SpoofEvent::side)
        .def_readonly("confidence",   &SpoofEvent::confidence)
        .def_readonly("timestamp_ms", &SpoofEvent::timestamp_ms);

    // ── AuthenticityProfile ────────────────────────────────────────────────────
    py::class_<AuthenticityProfile>(m, "AuthenticityProfile")
        .def(py::init<>())
        .def_readonly("authenticity_score", &AuthenticityProfile::authenticity_score)
        .def_readonly("pull_score",         &AuthenticityProfile::pull_score)
        .def_readonly("layering_score",     &AuthenticityProfile::layering_score)
        .def_readonly("phantom_score",      &AuthenticityProfile::phantom_score)
        .def_readonly("iceberg_score",      &AuthenticityProfile::iceberg_score)
        .def_readonly("num_spoof_events",   &AuthenticityProfile::num_spoof_events)
        .def_readonly("timestamp_ms",       &AuthenticityProfile::timestamp_ms)
        .def("spoof_events", [](const AuthenticityProfile& ap) {
            std::vector<SpoofEvent> v;
            for (int i = 0; i < ap.num_spoof_events; ++i) v.push_back(ap.spoof_events[i]);
            return v;
        });

    // ── RegimeWeights ──────────────────────────────────────────────────────────
    py::class_<RegimeWeights>(m, "RegimeWeights")
        .def(py::init<>())
        .def_readonly("regime",    &RegimeWeights::regime)
        .def_readonly("l1_weight", &RegimeWeights::l1_weight)
        .def_readonly("l2_weight", &RegimeWeights::l2_weight)
        .def_readonly("l3_weight", &RegimeWeights::l3_weight)
        .def_readonly("l4_weight", &RegimeWeights::l4_weight)
        .def_readonly("abstain",   &RegimeWeights::abstain);

    // ── Signal ─────────────────────────────────────────────────────────────────
    py::class_<Signal>(m, "Signal")
        .def(py::init<>())
        .def_readonly("direction",       &Signal::direction)
        .def_readonly("confidence",      &Signal::confidence)
        .def_readonly("urgency",         &Signal::urgency)
        .def_readonly("size_multiplier", &Signal::size_multiplier)
        .def_readonly("regime",          &Signal::regime)
        .def_readonly("abstain",         &Signal::abstain)
        .def_readonly("l1_component",    &Signal::l1_component)
        .def_readonly("l2_component",    &Signal::l2_component)
        .def_readonly("l3_component",    &Signal::l3_component)
        .def_readonly("l4_component",    &Signal::l4_component)
        .def_readonly("timestamp_ms",    &Signal::timestamp_ms);

    // ── Frame ──────────────────────────────────────────────────────────────────
    py::class_<Frame>(m, "Frame")
        .def(py::init<>())
        .def_readonly("timestamp_ms",   &Frame::timestamp_ms)
        .def_readonly("staircase",      &Frame::staircase)
        .def_readonly("game_state",     &Frame::game_state)
        .def_readonly("force_vector",   &Frame::force_vector)
        .def_readonly("authenticity",   &Frame::authenticity)
        .def_readonly("regime_weights", &Frame::regime_weights)
        .def_readonly("signal",         &Frame::signal)
        .def_readonly("snapshot",       &Frame::snapshot);

    // ── Pipeline ───────────────────────────────────────────────────────────────
    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<>())
        .def("process", &Pipeline::process,
             py::arg("snapshot"),
             py::arg("regime_hint") = RegimeType::UNKNOWN);

    // ── Conversion helper ──────────────────────────────────────────────────────
    // Create an Order from a Python dict for convenience
    m.def("make_order", [](py::dict d) {
        Order o;
        if (d.contains("order_id"))     o.order_id      = d["order_id"].cast<uint64_t>();
        if (d.contains("side"))         o.side          = d["side"].cast<Side>();
        if (d.contains("price"))        o.price         = d["price"].cast<double>();
        if (d.contains("size"))         o.size          = d["size"].cast<double>();
        if (d.contains("timestamp_ms")) o.timestamp_ms  = d["timestamp_ms"].cast<int64_t>();
        if (d.contains("action"))       o.action        = d["action"].cast<OrderAction>();
        if (d.contains("is_aggressive"))o.is_aggressive = d["is_aggressive"].cast<bool>();
        return o;
    }, "Create an Order from a Python dict");
}
