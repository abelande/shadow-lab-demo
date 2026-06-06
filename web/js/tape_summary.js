/**
 * tape_summary.js — Per-candle tape digest panel for replay mode.
 *
 * Shows: buy/sell volume, delta, largest fill, iceberg hits, cancel pressure.
 * Updates on each replay_frame from the server.
 */
const TapeSummary = (() => {
  let _container = null;

  /**
   * Initialize the tape summary panel.
   * @param {HTMLElement} container - Container element
   */
  function init(container) {
    _container = container;
    if (!container) return;
    _render(null);
  }

  /**
   * Update the panel from a replay frame.
   * @param {object} frame - replay_frame from server
   */
  function updateFromFrame(frame) {
    if (!frame || frame.type !== 'replay_frame') return;
    _render(frame.tape_summary || null);
  }

  function _render(tape) {
    if (!_container) return;

    if (!tape) {
      _container.innerHTML = `
        <div class="tape-sum-title">TAPE SUMMARY</div>
        <div class="tape-sum-empty">Waiting for data...</div>
      `;
      return;
    }

    const buyVol = tape.buy_volume || 0;
    const sellVol = tape.sell_volume || 0;
    const delta = tape.delta || 0;
    const total = buyVol + sellVol;
    const buyPct = total > 0 ? (buyVol / total) * 100 : 50;
    const iceHits = tape.iceberg_hits || 0;
    const cxlBid = tape.cancel_count_bid || 0;
    const cxlAsk = tape.cancel_count_ask || 0;

    const deltaClass = delta > 0 ? 'tape-sum-positive' : delta < 0 ? 'tape-sum-negative' : '';
    const deltaSign = delta > 0 ? '+' : '';

    let largestFill = '';
    if (tape.largest_fill_size > 0) {
      const sideArrow = tape.largest_fill_side === 'ASK' ? '▲' : '▼';
      const sideClass = tape.largest_fill_side === 'ASK' ? 'tape-sum-positive' : 'tape-sum-negative';
      const price = tape.largest_fill_price != null ? tape.largest_fill_price.toFixed(2) : '?';
      largestFill = `
        <div class="tape-sum-row">
          <span class="tape-sum-label">Largest fill</span>
          <span class="tape-sum-value ${sideClass}">${Math.round(tape.largest_fill_size)} @ ${price} ${sideArrow}</span>
        </div>`;
    }

    let iceDisplay = '';
    if (iceHits > 0) {
      const side = cxlBid > cxlAsk ? 'bid' : 'ask';
      iceDisplay = `
        <div class="tape-sum-row">
          <span class="tape-sum-label">Iceberg hits</span>
          <span class="tape-sum-value tape-sum-ice">${iceHits} (${side} side)</span>
        </div>`;
    }

    const cancelSide = cxlAsk > cxlBid ? 'ask heavy' : cxlBid > cxlAsk ? 'bid heavy' : 'balanced';
    const cancelTotal = cxlBid + cxlAsk;

    _container.innerHTML = `
      <div class="tape-sum-title">TAPE SUMMARY</div>
      <div class="tape-sum-row">
        <span class="tape-sum-label">▲ Bought</span>
        <span class="tape-sum-value tape-sum-positive">${_fmt(buyVol)} contracts</span>
      </div>
      <div class="tape-sum-row">
        <span class="tape-sum-label">▼ Sold</span>
        <span class="tape-sum-value tape-sum-negative">${_fmt(sellVol)} contracts</span>
      </div>
      <div class="tape-sum-row tape-sum-delta">
        <span class="tape-sum-label">Δ Delta</span>
        <span class="tape-sum-value ${deltaClass}">${deltaSign}${_fmt(delta)}</span>
      </div>
      ${largestFill}
      ${iceDisplay}
      <div class="tape-sum-row">
        <span class="tape-sum-label">Cancels</span>
        <span class="tape-sum-value tape-sum-muted">${cancelTotal} (${cancelSide})</span>
      </div>
      <div class="tape-sum-bar-wrap">
        <div class="tape-sum-bar">
          <div class="tape-sum-bar-buy" style="width:${buyPct.toFixed(1)}%"></div>
        </div>
        <span class="tape-sum-bar-label">${buyPct.toFixed(0)}% buy</span>
      </div>
    `;
  }

  function _fmt(n) {
    n = n || 0;
    if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + 'k';
    return Math.round(n).toString();
  }

  return { init, updateFromFrame };
})();
