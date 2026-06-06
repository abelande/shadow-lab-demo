/**
 * virtual_order_tool.js — §10.2 Virtual Order Tool
 *
 * Interaction:
 * 1) Enable mode via toolbar toggle
 * 2) Click price level on chart/DOM
 * 3) Modal: side, size, order type, max horizon
 * 4) POST /api/fill_sim/interactive
 * 5) Server runs fill_simulator.simulate_interactive
 * 6) Animate trajectory on chart
 * 7) Show FillOutcome summary sidebar
 */

class VirtualOrderTool {
  constructor({ chart, wsClient, container }) {
    this.chart = chart;
    this.wsClient = wsClient;
    this.container = container;
    this.enabled = false;
    this.pendingClick = null;
    this.currentTrajectory = [];
    this.animationTimer = null;

    this._buildUI();
    this._bindChartClick();
  }

  _buildUI() {
    this.toggleBtn = document.createElement('button');
    this.toggleBtn.textContent = 'Place Virtual Order';
    this.toggleBtn.className = 'vo-toggle-btn';
    this.toggleBtn.onclick = () => this.toggle();

    this.summary = document.createElement('div');
    this.summary.className = 'vo-summary';
    this.summary.style.cssText = 'font-size:12px;color:#ddd;padding:8px;border-left:1px solid #333;';
    this.summary.innerHTML = '<b>Virtual Order</b><br/>Idle';

    this.container.append(this.toggleBtn, this.summary);
  }

  _bindChartClick() {
    if (!this.chart || !this.chart.container) return;
    this.chart.container.addEventListener('click', (e) => {
      if (!this.enabled) return;
      const coords = this._extractChartCoords(e);
      this.pendingClick = coords;
      this._openOrderModal(coords);
    });
  }

  toggle() {
    this.enabled = !this.enabled;
    this.toggleBtn.textContent = this.enabled ? 'Virtual Order: ON' : 'Place Virtual Order';
    this.toggleBtn.style.background = this.enabled ? '#2e7d32' : '';
  }

  _extractChartCoords(e) {
    const rect = this.chart.container.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      timestamp_ms: Date.now(), // replace with replay time if available
      price: this.chart.priceFromY ? this.chart.priceFromY(e.clientY - rect.top) : null,
      symbol: this.chart.symbol || 'NQ'
    };
  }

  _openOrderModal(coords) {
    const side = prompt(`Side (buy/sell) @ ${coords.price ?? 'price'}:`, 'buy');
    if (!side) return;
    const size = Number(prompt('Size (contracts):', '1'));
    const orderType = prompt('Order type (limit/market/step-ahead):', 'limit');
    const maxHorizonSec = Number(prompt('Max horizon seconds:', '120'));

    const payload = {
      symbol: coords.symbol,
      timestamp_ms: coords.timestamp_ms,
      price: coords.price,
      side,
      size,
      order_type: orderType,
      max_horizon_sec: maxHorizonSec,
      mode: 'interactive'
    };

    this._submitInteractiveOrder(payload);
  }

  async _submitInteractiveOrder(payload) {
    try {
      const resp = await fetch('/api/fill_sim/interactive', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const result = await resp.json();
      this.currentTrajectory = result.trajectory || [];
      this._animateTrajectory(this.currentTrajectory);
      this._renderSummary(result);
    } catch (err) {
      console.error('[VirtualOrderTool] submit failed:', err);
      this.summary.innerHTML = `<b>Virtual Order</b><br/><span style="color:#f44336">${err.message}</span>`;
    }
  }

  _animateTrajectory(trajectory) {
    if (this.animationTimer) clearInterval(this.animationTimer);
    if (!trajectory.length) return;

    let i = 0;
    this.animationTimer = setInterval(() => {
      const step = trajectory[i];
      if (!step) {
        clearInterval(this.animationTimer);
        return;
      }

      // Placeholder drawing hooks into existing chart overlays
      this._drawQueueBar(step);
      this._drawOrderMarker(step);
      this._updateLiveTooltip(step);

      i += 1;
    }, 120);
  }

  _drawQueueBar(step) {
    // queue-position bar shrinking as orders ahead are filled/cancelled
    // integrate with depth_overlay.js or custom overlay layer
    // step fields expected: {timestamp_ms, queue_position, volume_ahead, p_fill_estimate}
  }

  _drawOrderMarker(step) {
    // color behavior:
    // green flash on fill, red flash on adverse_exit, gray on timeout
    // step.status expected: pending|filled|adverse_exit|timeout|cancelled
  }

  _updateLiveTooltip(step) {
    // show current P(fill), adverse ticks, realized PnL-in-progress
  }

  _renderSummary(outcome) {
    const reasonColor = {
      full: '#4caf50', partial: '#8bc34a', adverse_exit: '#f44336', timeout: '#9e9e9e', cancelled: '#9e9e9e'
    }[outcome.fill_reason] || '#ddd';

    this.summary.innerHTML = `
      <b>FillOutcome</b><br/>
      filled: ${outcome.filled}<br/>
      filled_size: ${outcome.filled_size}<br/>
      queue_entry: ${outcome.queue_position_at_entry}<br/>
      queue_fill: ${outcome.queue_position_at_fill ?? '-'}<br/>
      adverse_ticks: ${outcome.adverse_ticks_at_fill}<br/>
      realized_pnl: ${Number(outcome.realized_pnl).toFixed(2)}<br/>
      reason: <span style="color:${reasonColor}">${outcome.fill_reason}</span>
    `;
  }
}

if (typeof module !== 'undefined') module.exports = VirtualOrderTool;
