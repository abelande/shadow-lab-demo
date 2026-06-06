/**
 * exec_sim.js — Execution simulation tool.
 * Simulated buy/sell with SL/TP price lines on the LWC chart.
 * No real orders. All state is in-memory.
 */
const ExecSim = (() => {
  let _positions = [];   // { id, side, qty, entryPrice, sl, tp, slLine, tpLine }
  let _orderType = 'limit';  // 'limit' | 'market'
  let _lastPrice = null;
  let _posIdCounter = 0;

  // Drag state for SL/TP lines
  let _dragging = null; // { posId, lineType: 'sl'|'tp', startY }

  function init() {
    _bindToggle();
    _bindQty();
    _bindOrderButtons();
    _bindSimToggle();
    _bindDrag();
  }

  function _bindSimToggle() {
    // Add [SIM] toggle button to stats-header
    const header = document.getElementById('stats-header');
    if (!header) return;
    const btn = document.createElement('button');
    btn.id = 'exec-sim-toggle';
    btn.className = 'exec-toggle';
    btn.textContent = 'SIM';
    btn.style.cssText = 'margin-left:auto; margin-right:8px; padding:3px 10px;';
    btn.addEventListener('click', () => {
      const panel = document.getElementById('exec-sim-panel');
      if (!panel) return;
      const visible = panel.style.display !== 'none';
      panel.style.display = visible ? 'none' : 'flex';
    });
    header.appendChild(btn);
  }

  function _bindToggle() {
    document.getElementById('exec-toggle-limit')?.addEventListener('click', () => _setType('limit'));
    document.getElementById('exec-toggle-mkt')?.addEventListener('click', () => _setType('market'));
  }

  function _setType(type) {
    _orderType = type;
    document.querySelectorAll('.exec-toggle').forEach(b => b.classList.toggle('active', b.dataset.type === type));
    const pg = document.getElementById('exec-price-group');
    if (pg) pg.style.display = type === 'limit' ? 'flex' : 'none';
  }

  function _bindQty() {
    document.getElementById('exec-qty-minus')?.addEventListener('click', () => {
      const inp = document.getElementById('exec-qty');
      if (inp) inp.value = Math.max(1, parseInt(inp.value || '1', 10) - 1);
    });
    document.getElementById('exec-qty-plus')?.addEventListener('click', () => {
      const inp = document.getElementById('exec-qty');
      if (inp) inp.value = Math.min(100, parseInt(inp.value || '1', 10) + 1);
    });
  }

  function _bindOrderButtons() {
    document.getElementById('exec-buy')?.addEventListener('click', () => _placeOrder('BUY'));
    document.getElementById('exec-sell')?.addEventListener('click', () => _placeOrder('SELL'));
  }

  function _placeOrder(side) {
    const chart = typeof ChartModule !== 'undefined' ? ChartModule.getChart() : null;
    const series = typeof ChartModule !== 'undefined' ? ChartModule.getCandleSeries() : null;
    if (!chart || !series) return;

    const qty = parseInt(document.getElementById('exec-qty')?.value || '1', 10);
    let entryPrice;

    if (_orderType === 'market') {
      entryPrice = _lastPrice || ChartModule.getLastPrice();
      if (!entryPrice) { console.warn('[ExecSim] No last price for market order'); return; }
    } else {
      entryPrice = parseFloat(document.getElementById('exec-price')?.value);
      if (!entryPrice || isNaN(entryPrice)) { console.warn('[ExecSim] Invalid limit price'); return; }
    }

    // Default SL/TP: 10 and 20 ticks away (0.25 tick size for ES/NQ)
    const tickSize = 0.25;
    const slOffset = 10 * tickSize;
    const tpOffset = 20 * tickSize;

    const sl = side === 'BUY' ? entryPrice - slOffset : entryPrice + slOffset;
    const tp = side === 'BUY' ? entryPrice + tpOffset : entryPrice - tpOffset;

    const slLine = series.createPriceLine({
      price: sl,
      color: '#e74c3c',
      lineWidth: 1,
      lineStyle: 2,  // dashed
      axisLabelVisible: true,
      title: `SL (${qty})`,
    });

    const tpLine = series.createPriceLine({
      price: tp,
      color: '#2ecc71',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: `TP (${qty})`,
    });

    const entryLine = series.createPriceLine({
      price: entryPrice,
      color: side === 'BUY' ? '#2ecc71' : '#e74c3c',
      lineWidth: 1,
      lineStyle: 0,
      axisLabelVisible: true,
      title: `${side} ${qty}`,
    });

    const pos = {
      id: ++_posIdCounter,
      side,
      qty,
      entryPrice,
      sl,
      tp,
      slLine,
      tpLine,
      entryLine,
    };

    _positions.push(pos);
    _renderPositions();
  }

  function _closePosition(id) {
    const series = typeof ChartModule !== 'undefined' ? ChartModule.getCandleSeries() : null;
    const idx = _positions.findIndex(p => p.id === id);
    if (idx === -1) return;
    const pos = _positions[idx];
    if (series) {
      try { series.removePriceLine(pos.slLine); } catch (e) { /* ignore */ }
      try { series.removePriceLine(pos.tpLine); } catch (e) { /* ignore */ }
      try { series.removePriceLine(pos.entryLine); } catch (e) { /* ignore */ }
    }
    _positions.splice(idx, 1);
    _renderPositions();
  }

  function _renderPositions() {
    const el = document.getElementById('exec-positions');
    if (!el) return;
    el.innerHTML = '';
    const currentPrice = _lastPrice || (typeof ChartModule !== 'undefined' ? ChartModule.getLastPrice() : null);
    for (const pos of _positions) {
      const pnlPts = currentPrice
        ? (pos.side === 'BUY' ? currentPrice - pos.entryPrice : pos.entryPrice - currentPrice) * pos.qty
        : 0;
      const pnlClass = pnlPts >= 0 ? 'pos' : 'neg';
      const chip = document.createElement('div');
      chip.className = `exec-position-chip ${pos.side === 'BUY' ? 'long' : 'short'}`;
      chip.innerHTML = `
        <span>${pos.side} ${pos.qty} @ ${pos.entryPrice.toFixed(2)}</span>
        <span>SL:${pos.sl.toFixed(2)}</span>
        <span>TP:${pos.tp.toFixed(2)}</span>
        <span class="exec-position-pnl ${pnlClass}">${pnlPts >= 0 ? '+' : ''}${pnlPts.toFixed(2)}pts</span>
        <span class="exec-position-close" data-id="${pos.id}">✕</span>
      `;
      chip.querySelector('.exec-position-close').addEventListener('click', () => _closePosition(pos.id));
      el.appendChild(chip);
    }
  }

  // ── SL/TP Drag via mouse events ──────────────────────────────

  function _bindDrag() {
    const container = document.getElementById('chart-container');
    if (!container) return;

    container.addEventListener('mousedown', (e) => {
      if (_positions.length === 0) return;
      const series = typeof ChartModule !== 'undefined' ? ChartModule.getCandleSeries() : null;
      if (!series) return;

      const rect = container.getBoundingClientRect();
      const mouseY = e.clientY - rect.top;

      // Check if mouse is near any SL or TP line (±4px)
      for (const pos of _positions) {
        for (const lineType of ['sl', 'tp']) {
          const price = pos[lineType];
          try {
            const lineY = series.priceToCoordinate(price);
            if (lineY != null && Math.abs(mouseY - lineY) <= 4) {
              _dragging = { posId: pos.id, lineType, startY: mouseY };
              e.preventDefault();
              container.style.cursor = 'ns-resize';
              return;
            }
          } catch (err) { /* ignore */ }
        }
      }
    });

    document.addEventListener('mousemove', (e) => {
      if (!_dragging) return;
      const series = typeof ChartModule !== 'undefined' ? ChartModule.getCandleSeries() : null;
      const container = document.getElementById('chart-container');
      if (!series || !container) return;

      const rect = container.getBoundingClientRect();
      const mouseY = e.clientY - rect.top;

      let newPrice;
      try {
        newPrice = series.coordinateToPrice(mouseY);
      } catch (err) { return; }
      if (newPrice == null || !Number.isFinite(newPrice)) return;

      // Snap to tick (0.25)
      const tickSize = 0.25;
      newPrice = Math.round(newPrice / tickSize) * tickSize;

      const pos = _positions.find(p => p.id === _dragging.posId);
      if (!pos) { _dragging = null; return; }

      // Remove old line and create new one
      const lt = _dragging.lineType;
      const oldLine = lt === 'sl' ? pos.slLine : pos.tpLine;
      try { series.removePriceLine(oldLine); } catch (err) { /* ignore */ }

      const newLine = series.createPriceLine({
        price: newPrice,
        color: lt === 'sl' ? '#e74c3c' : '#2ecc71',
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: true,
        title: `${lt.toUpperCase()} (${pos.qty})`,
      });

      pos[lt] = newPrice;
      if (lt === 'sl') pos.slLine = newLine;
      else pos.tpLine = newLine;
    });

    document.addEventListener('mouseup', () => {
      if (_dragging) {
        _dragging = null;
        const container = document.getElementById('chart-container');
        if (container) container.style.cursor = '';
        _renderPositions();
      }
    });
  }

  /** Called by app.js on every frame/tick to keep last price and refresh P&L. */
  function updatePrice(price) {
    if (!price || !Number.isFinite(price)) return;
    _lastPrice = price;
    if (_positions.length > 0) _renderPositions();
  }

  return { init, updatePrice };
})();
