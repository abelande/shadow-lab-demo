/**
 * tape_feed.js — Time & Sales (Tape) scrolling feed.
 * Displays: TIME | SIDE | PRICE | SIZE
 * In live mode: uses real frame.tape entries.
 * In replay/paused mode: synthesizes trades from frame direction + staircase.
 */
const TapeFeed = (() => {
  let listEl = null;
  const MAX_ROWS = 60;

  // State for synthesis
  let lastTimestampMs = 0;
  let lastFillCount = 0;

  /**
   * Initialize the tape feed panel.
   * @param {HTMLElement} el - Container element
   */
  function init(el) {
    listEl = el;
  }

  /**
   * Reset the tape feed (call when switching modes).
   */
  function reset() {
    if (listEl) listEl.innerHTML = '';
    lastTimestampMs = 0;
    lastFillCount = 0;
  }

  /**
   * Update tape from a new frame.
   * Uses real tape data when available; synthesizes otherwise.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!frame || !listEl) return;

    const hasTape = frame.tape && frame.tape.length > 0;

    if (hasTape) {
      for (const trade of frame.tape) {
        if (trade) _addRow(trade, false);
      }
    } else {
      _synthesizeFromFrame(frame);
    }

    // Trim oldest rows (at bottom since newest is prepended to top)
    while (listEl.children.length > MAX_ROWS) {
      listEl.removeChild(listEl.lastChild);
    }
  }

  /**
   * Synthesize a tape entry from frame direction + staircase data.
   * Fires at most once per unique timestamp and only when |direction| > 0.15.
   * @param {object} frame
   */
  function _synthesizeFromFrame(frame) {
    const ts = typeof frame.timestamp_ms === 'number' ? frame.timestamp_ms : Date.now();

    // Don't synthesize the same timestamp twice
    if (ts === lastTimestampMs) return;
    lastTimestampMs = ts;

    const direction = frame.direction || 0;

    // Skip very weak directional signals
    if (Math.abs(direction) < 0.15) return;

    // Determine aggressor side and price
    const sc = frame.staircase;
    let side, price;

    if (direction > 0) {
      side = 'ASK';  // buyer hitting the ask
      price = sc && sc.ask_levels && sc.ask_levels[0] ? sc.ask_levels[0].price : null;
    } else {
      side = 'BID';  // seller hitting the bid
      price = sc && sc.bid_levels && sc.bid_levels[0] ? sc.bid_levels[0].price : null;
    }

    // Fallback to dom_rows
    if (price == null && frame.dom_rows) {
      for (const r of frame.dom_rows) {
        if (r && r.side === side) { price = r.price; break; }
      }
    }

    if (price == null) return;

    // Estimate size from size_multiplier and fills delta
    const fillCount = (frame.stats && frame.stats.fill_count) || 0;
    const fillDelta = Math.max(0, fillCount - lastFillCount);
    lastFillCount = fillCount;

    const baseSize = Math.max(1, Math.round((frame.size_multiplier || 1) * 2));
    const size = fillDelta > 0 ? Math.min(fillDelta * baseSize, 50) : baseSize;

    _addRow({ timestamp_ms: ts, side, price, size }, true);
  }

  /**
   * Add a single trade row to the tape.
   * @param {object} trade - { timestamp_ms, side, price, size }
   * @param {boolean} synthetic - Mark synthetic entries with lower opacity
   */
  function _addRow(trade, synthetic) {
    const isBuy = trade.side === 'BID';
    const cls = isBuy ? 'buy' : 'sell';
    const time = _fmtTime(trade.timestamp_ms);
    const side = trade.side || '--';
    const price = trade.price != null ? Number(trade.price).toFixed(2) : '--';
    const size = trade.size != null ? trade.size : '--';

    const row = document.createElement('div');
    row.className = `tape-row ${cls}${synthetic ? ' synthetic' : ''}`;
    row.innerHTML =
      `<span class="tape-time">${time}</span>` +
      `<span class="tape-side">${side}</span>` +
      `<span class="tape-price">${price}</span>` +
      `<span class="tape-size">${size}</span>`;

    // Prepend so newest is always at the top
    listEl.prepend(row);

    // Mark this as the latest row, clear previous latest
    const prev = listEl.querySelector('.tape-latest');
    if (prev) prev.classList.remove('tape-latest');
    row.classList.add('tape-latest');

    // Large fill highlight (≥10 contracts)
    if (trade.size != null && Number(trade.size) >= 10) {
      row.classList.add('tape-large');
    }

    // Flash new rows briefly
    row.classList.add('tape-flash');
    setTimeout(() => row.classList.remove('tape-flash'), 300);
  }

  /**
   * Format timestamp (ms) → HH:MM:SS.mmm
   * @param {number} ms
   * @returns {string}
   */
  function _fmtTime(ms) {
    if (ms == null) return '--:--:--';
    const d = new Date(ms);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    const ml = String(d.getMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss}.${ml}`;
  }

  return { init, reset, updateFromFrame };
})();
