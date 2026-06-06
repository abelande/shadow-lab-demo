/**
 * chart.js — TradingView Lightweight Charts initialization
 * Manages the candlestick chart with dark theme, proper OHLC time bucketing.
 * Supports configurable timeframes: 1s, 5s, 15s, 60s.
 */
const ChartModule = (() => {
  let chart = null;
  let candleSeries = null;
  let tickLineSeries = null;
  let lastPrice = null;

  // OHLC bucket state
  let currentBucket = null;  // { time (sec), open, high, low, close }
  let bucketSizeMs = 15000;  // default 15-second candles
  let lastPushedTimeSec = 0; // track monotonic time for Lightweight Charts

  // ET timezone offset: Lightweight Charts renders UTC timestamps directly on
  // the axis. To display ET, shift every timestamp by the current ET/UTC offset.
  // EDT (Mar-Nov) = UTC-4 = -14400s, EST (Nov-Mar) = UTC-5 = -18000s.
  function _etOffsetSec() {
    // Get the current wall-clock ET offset from the browser engine.
    const now = new Date();
    const utcStr = now.toLocaleString('en-US', { timeZone: 'UTC' });
    const etStr  = now.toLocaleString('en-US', { timeZone: 'America/New_York' });
    return Math.round((new Date(etStr) - new Date(utcStr)) / 1000);
  }
  const ET_OFFSET_SEC = _etOffsetSec();
  console.log('[Chart] ET offset sec:', ET_OFFSET_SEC);

  /** Shift a UTC Unix-seconds timestamp to ET for chart display. */
  function _toET(utcSec) {
    return utcSec + ET_OFFSET_SEC;
  }

  /**
   * Initialize the chart inside the given container element.
   * @param {HTMLElement} container
   * @returns {{ chart, candleSeries }}
   */
  function init(container) {
    chart = LightweightCharts.createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: 'solid', color: '#0a0a0a' },
        textColor: '#888888',
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(42, 42, 42, 0.5)' },
        horzLines: { color: 'rgba(42, 42, 42, 0.5)' },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { color: 'rgba(255,255,255,0.2)', style: 2, width: 1 },
        horzLine: { color: 'rgba(255,255,255,0.2)', style: 2, width: 1 },
      },
      rightPriceScale: {
        borderColor: '#2a2a2a',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: '#2a2a2a',
        timeVisible: true,
        secondsVisible: true,
      },
      handleScroll: true,
      handleScale: true,
    });

    candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
      upColor: '#2ecc71',
      downColor: '#e74c3c',
      borderUpColor: '#2ecc71',
      borderDownColor: '#e74c3c',
      wickUpColor: '#2ecc71',
      wickDownColor: '#e74c3c',
      borderVisible: true,
    });

    // Tick line series — shows individual trades within the active candle
    tickLineSeries = chart.addSeries(LightweightCharts.LineSeries, {
      color: '#f1c40f',
      lineWidth: 1,
      lineStyle: 0,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    const resizeObserver = new ResizeObserver(() => {
      if (chart && container) {
        chart.applyOptions({
          width: container.clientWidth,
          height: container.clientHeight,
        });
      }
    });
    resizeObserver.observe(container);

    // Re-render level bars on chart pan/zoom so they stay aligned to price
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
      if (typeof LevelRenderer !== 'undefined') LevelRenderer.render();
    });
    chart.subscribeCrosshairMove(() => {
      if (typeof LevelRenderer !== 'undefined') LevelRenderer.render();
    });

    return { chart, candleSeries };
  }

  /**
   * Set the OHLC bucket size and reset candle history.
   * @param {number} ms - Bucket duration in milliseconds (1000, 5000, 15000, 60000)
   */
  function setTimeframe(ms) {
    bucketSizeMs = ms;
    reset();
  }

  /**
   * Reset candle history and aggregation state.
   * Call this when switching feeds or modes.
   */
  function reset() {
    currentBucket = null;
    lastPrice = null;
    lastPushedTimeSec = 0;
    if (candleSeries) {
      try { candleSeries.setData([]); } catch (e) { /* ignore */ }
    }
  }

  /**
   * Core price processor — feeds one price+timestamp into the OHLC bucket.
   * Called once per trade (or once for fallback DOM price).
   *
   * Track 3E fix: The error recovery path previously advanced lastPushedTimeSec
   * unconditionally, which could cause it to leap ahead of actual data time.
   * All subsequent real data would then fall "behind" and never form new candles.
   * Now: error recovery resets to the data's actual bucket time.
   *
   * @param {number} price
   * @param {number} tsMs - epoch milliseconds
   */
  function _processPrice(price, tsMs) {
    if (!candleSeries || price === null || !Number.isFinite(price)) return;
    if (price <= 0) return;  // Track 3E: reject invalid prices
    lastPrice = price;

    const bucketSizeSec = Math.max(1, Math.floor(bucketSizeMs / 1000));
    const tsSec = Math.floor(tsMs / 1000);
    const bucketTimeSec = Math.floor(tsSec / bucketSizeSec) * bucketSizeSec;

    if (currentBucket === null) {
      currentBucket = { time: bucketTimeSec, open: price, high: price, low: price, close: price };
    } else if (bucketTimeSec > currentBucket.time) {
      const t = Math.max(bucketTimeSec, lastPushedTimeSec + bucketSizeSec);
      currentBucket = { time: t, open: price, high: price, low: price, close: price};
    } else if (bucketTimeSec < currentBucket.time - bucketSizeSec * 2) {
      console.warn('[Chart] Timestamp jump detected, resetting to data time');
      lastPushedTimeSec = Math.max(0, bucketTimeSec - bucketSizeSec);
      currentBucket = { time: bucketTimeSec, open: price, high: price, low: price, close: price };
    } else {
      if (price > currentBucket.high) currentBucket.high = price;
      if (price < currentBucket.low) currentBucket.low = price;
      currentBucket.close = price;
    }

    // Push to chart with ET-shifted time for display only.
    // Internal state (currentBucket.time, lastPushedTimeSec) stays UTC.
    try {
      candleSeries.update({
        time: _toET(currentBucket.time),
        open: currentBucket.open,
        high: currentBucket.high,
        low: currentBucket.low,
        close: currentBucket.close,
      });
      lastPushedTimeSec = currentBucket.time;
    } catch (e) {
      console.warn('[Chart] update error:', e.message);
      const recoveryTime = Math.max(bucketTimeSec, lastPushedTimeSec + bucketSizeSec);
      currentBucket = { time: recoveryTime, open: price, high: price, low: price, close: price };
      try {
        candleSeries.update({ time: _toET(recoveryTime), open: price, high: price, low: price, close: price });
        lastPushedTimeSec = recoveryTime;
      } catch (e2) {
        lastPushedTimeSec = recoveryTime;
      }
    }
  }

  /**
   * Update chart with tick data from a new frame.
   * Processes ALL tape trades for proper OHLC highs/lows.
   * Falls back to DOM mid / staircase when no tape is present (L1 BBO mode).
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!candleSeries || !frame) return;

    // Parse frame timestamp as fallback for trades without their own ts
    let rawTs = frame.timestamp_ms;
    if (rawTs && typeof rawTs === 'object') {
      rawTs = rawTs.timestamp_ms || rawTs.value || Date.now();
    }
    const frameTsMs = Number.isFinite(Number(rawTs)) ? Number(rawTs) : Date.now();

    if (frame.tape && frame.tape.length > 0) {
      // Process every trade in the frame — each contributes to OHLC
      for (const trade of frame.tape) {
        if (!trade || trade.price == null || trade.price <= 0) continue;
        const tMs = (trade.timestamp_ms != null && Number.isFinite(Number(trade.timestamp_ms)))
          ? Number(trade.timestamp_ms)
          : frameTsMs;
        _processPrice(trade.price, tMs);
      }
    } else {
      // No tape — fall back to DOM mid / staircase (L1 BBO or gap frames)
      const price = _extractPrice(frame);
      if (price !== null) _processPrice(price, frameTsMs);
    }
  }

  /**
   * Extract trade price from frame data for OHLC candles.
   * Priority: most recent trade (cleanest source) → DOM mid → staircase → bars.
   * Trade prices produce proper OHLC candles that reflect actual fills,
   * not book midpoint which barely moves between snapshots.
   * @param {object} frame
   * @returns {number|null}
   */
  function _extractPrice(frame) {
    // 1. Most recent trade — best source for price action
    if (frame.tape && frame.tape.length > 0) {
      const last = frame.tape[frame.tape.length - 1];
      if (last && last.price != null && last.price > 0) return last.price;
    }

    // 2. DOM rows — best bid/ask midpoint
    if (frame.dom_rows && frame.dom_rows.length > 0) {
      let bestBid = null, bestAsk = null;
      for (const row of frame.dom_rows) {
        if (!row || row.price == null) continue;
        if (row.side === 'BID' && (bestBid === null || row.price > bestBid)) bestBid = row.price;
        if (row.side === 'ASK' && (bestAsk === null || row.price < bestAsk)) bestAsk = row.price;
      }
      if (bestBid !== null && bestAsk !== null) return (bestBid + bestAsk) / 2;
      if (bestBid !== null) return bestBid;
      if (bestAsk !== null) return bestAsk;
    }

    // 3. Staircase levels
    if (frame.staircase) {
      const sc = frame.staircase;
      const bid = sc.bid_levels && sc.bid_levels[0] ? sc.bid_levels[0].price : null;
      const ask = sc.ask_levels && sc.ask_levels[0] ? sc.ask_levels[0].price : null;
      if (bid !== null && ask !== null) return (bid + ask) / 2;
    }

    // 4. Bar prices
    if (frame.bid_bars && frame.bid_bars.length > 0 && frame.bid_bars[0].price != null) return frame.bid_bars[0].price;
    if (frame.ask_bars && frame.ask_bars.length > 0 && frame.ask_bars[0].price != null) return frame.ask_bars[0].price;

    return lastPrice;
  }

  /**
   * Bulk-load a pre-built candle array (for replay mode historical load).
   * @param {Array} candles - Array of { time, open, high, low, close, volume }
   */
  function loadCandles(candles) {
    if (!candleSeries || !Array.isArray(candles) || candles.length === 0) return;
    reset();
    try {
      // Sort by time ascending and shift to ET for display
      const sorted = candles.map(c => ({ ...c, time: _toET(c.time) })).sort((a, b) => a.time - b.time);
      candleSeries.setData(sorted);
      if (sorted.length > 0) {
        lastPushedTimeSec = candles[candles.length - 1].time;  // UTC, not ET-shifted
        lastPrice = sorted[sorted.length - 1].close;
      }
    } catch (e) {
      console.warn('[Chart] loadCandles error:', e.message);
    }
  }

  /**
   * Add or update a single candle (for incremental replay updates).
   * @param {object} candle - { time, open, high, low, close, volume }
   */
  function addCandle(candle) {
    if (!candleSeries || !candle) return;
    try {
      const etCandle = { ...candle, time: _toET(candle.time) };
      candleSeries.update(etCandle);
      lastPushedTimeSec = candle.time;
      lastPrice = candle.close;
    } catch (e) {
      console.warn('[Chart] addCandle error:', e.message);
    }
  }

  // ── Tick-based updates (live mode fast-path) ─────────────────────
  let _pendingTick = null;
  let _tickRafId = null;

  /**
   * Process a price_tick for smooth live chart updates.
   * Uses requestAnimationFrame coalescing — only the latest tick
   * is rendered per animation frame, keeping the chart smooth.
   * @param {object} tick - { price, bid, ask, ts }
   */
  function updateFromTick(tick) {
    if (!tick || tick.price == null) return;
    _pendingTick = tick;
    if (_tickRafId === null) {
      _tickRafId = requestAnimationFrame(_flushTick);
    }
  }

  function _flushTick() {
    _tickRafId = null;
    const tick = _pendingTick;
    if (!tick) return;
    _pendingTick = null;
    const tsMs = tick.ts || Date.now();
    _processPrice(tick.price, tsMs);
  }

  /**
   * Set the tick line data for the active candle.
   * Draws individual trade prices as a line within the candle's time window.
   * @param {Array} ticks - Array of { ts (Unix seconds, fractional), price }
   * @param {number} candleTime - Candle start time in Unix seconds
   * @param {number} timeframeS - Candle duration in seconds
   */
  function setTickLine(ticks, candleTime, timeframeS) {
    if (!tickLineSeries) return;

    if (!ticks || ticks.length === 0) {
      try { tickLineSeries.setData([]); } catch (e) { /* ignore */ }
      return;
    }

    // Map ticks to sub-second time points within the candle window.
    // LWC needs strictly increasing time values.
    // Distribute ticks evenly across the candle duration so they
    // span the candle's time range on the chart.
    const data = [];
    const step = ticks.length > 1 ? timeframeS / ticks.length : 0;
    for (let i = 0; i < ticks.length; i++) {
      const t = candleTime + (i * step);
      data.push({ time: t, value: ticks[i].price });
    }

    try {
      tickLineSeries.setData(data);
    } catch (e) {
      console.warn('[Chart] setTickLine error:', e.message);
    }
  }

  /** Clear the tick line overlay. */
  function clearTickLine() {
    if (tickLineSeries) {
      try { tickLineSeries.setData([]); } catch (e) { /* ignore */ }
    }
  }

  function getChart() { return chart; }
  function getCandleSeries() { return candleSeries; }
  function getLastPrice() { return lastPrice; }

  /**
   * Update the forming candle with a single tick during replay playback.
   * The candle grows incrementally as ticks arrive.
   * @param {object} tick - { price, ts (epoch nanoseconds), side, size }
   * @param {number} candleTimeSec - candle start time in Unix seconds
   */
  function updateFormingCandle(tick, candleTimeSec) {
    if (!candleSeries || !tick || tick.price == null) return;
    const price = tick.price;

    if (currentBucket === null || currentBucket.time !== candleTimeSec) {
      currentBucket = { time: candleTimeSec, open: price, high: price, low: price, close: price };
    } else {
      if (price > currentBucket.high) currentBucket.high = price;
      if (price < currentBucket.low) currentBucket.low = price;
      currentBucket.close = price;
    }

    try {
      candleSeries.update({
        time: _toET(currentBucket.time),
        open: currentBucket.open,
        high: currentBucket.high,
        low: currentBucket.low,
        close: currentBucket.close,
      });
      lastPushedTimeSec = currentBucket.time;
      lastPrice = price;
    } catch (e) {
      // ignore — time ordering issues during scrub
    }
  }

  /**
   * Slice chart to show only candles [0..idx]. Used by replay scrubber.
   * @param {number} idx - Last candle index to show
   * @param {Array} allCandles - Full candle array
   */
  function sliceTo(idx, allCandles) {
    if (!candleSeries || !Array.isArray(allCandles) || allCandles.length === 0) return;
    const slice = allCandles.slice(0, idx + 1);
    try {
      const sorted = slice.map(c => ({ ...c, time: _toET(c.time) }))
                          .sort((a, b) => a.time - b.time);
      candleSeries.setData(sorted);
      if (sorted.length > 0) {
        lastPushedTimeSec = allCandles[Math.min(idx, allCandles.length - 1)].time;  // UTC
        lastPrice = sorted[sorted.length - 1].close;
      }
    } catch (e) {
      console.warn('[Chart] sliceTo error:', e.message);
    }
  }

  // ── Bar Countdown ─────────────────────────────────────────────
  let _countdownTimer = null;
  let _countdownTf = 15; // seconds

  function startBarCountdown(timeframeSec) {
    _countdownTf = timeframeSec || 15;
    stopBarCountdown();
    const el = document.getElementById('bar-countdown');
    if (!el) return;
    el.style.display = 'block';

    function _tick() {
      if (!candleSeries || lastPushedTimeSec === 0) return;
      const nowSec = Math.floor(Date.now() / 1000);
      // Use the shifted ET time so alignment matches the chart axis
      const barStartEt = lastPushedTimeSec + ET_OFFSET_SEC;
      const barEndEt = barStartEt + _countdownTf;
      const remaining = Math.max(0, barEndEt - (nowSec + (ET_OFFSET_SEC)));
      el.textContent = remaining + 's';

      // Y-position: query priceToCoordinate for the last close price
      if (candleSeries && lastPrice != null) {
        try {
          const y = candleSeries.priceToCoordinate(lastPrice);
          if (y != null && Number.isFinite(y)) {
            // Place below the price label (approx +14px below)
            el.style.top = (y + 14) + 'px';
          }
        } catch (e) { /* ignore */ }
      }
    }

    _tick();
    _countdownTimer = setInterval(_tick, 500); // update twice/sec for smooth feel
  }

  function stopBarCountdown() {
    if (_countdownTimer) { clearInterval(_countdownTimer); _countdownTimer = null; }
    const el = document.getElementById('bar-countdown');
    if (el) el.style.display = 'none';
  }

  function clear() { reset(); }

  return { init, updateFromFrame, updateFromTick, loadCandles, addCandle, setTickLine, clearTickLine, updateFormingCandle, getChart, getCandleSeries, getLastPrice, setTimeframe, reset, clear, sliceTo, startBarCountdown, stopBarCountdown };
})();
