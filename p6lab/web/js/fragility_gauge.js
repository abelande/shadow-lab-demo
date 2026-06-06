/**
 * fragility_gauge.js — §10.5 Fragility Gauge Panel
 *
 * Displays:
 * - FI_fast (large arc gauge)
 * - FI_full (smaller arc gauge)
 * - 6 sub-index bars: DF/CF/RF/SF/FT/CIS
 * Threshold behavior:
 * - >0.5 => yellow
 * - >0.7 => red
 * Integration behaviors:
 * - FI_fast > 0.6 => lower signal detection threshold 0.55 -> 0.40
 * - FI_full > 0.7 => backtest max size multiplier -> 0.5
 */

class FragilityGauge {
  constructor({ container, wsClient, signalBar, backtestControls }) {
    this.container = container;
    this.wsClient = wsClient;
    this.signalBar = signalBar;
    this.backtestControls = backtestControls;

    this.state = {
      FI_fast: 0,
      FI_full: 0,
      DF: 0, CF: 0, RF: 0, SF: 0, FT: 0, CIS: 0
    };

    this._buildUI();
    this._bindWs();
  }

  _buildUI() {
    this.root = document.createElement('div');
    this.root.className = 'fragility-gauge-root';

    this.canvasFast = document.createElement('canvas');
    this.canvasFast.width = 180; this.canvasFast.height = 110;
    this.canvasFull = document.createElement('canvas');
    this.canvasFull.width = 140; this.canvasFull.height = 90;

    this.bars = document.createElement('div');
    this.bars.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:11px;';

    this.barEls = {};
    ['DF','CF','RF','SF','FT','CIS'].forEach((k) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:6px;';
      const label = document.createElement('span'); label.textContent = k; label.style.width = '26px';
      const outer = document.createElement('div'); outer.style.cssText = 'flex:1;height:8px;background:#333;';
      const fill = document.createElement('div'); fill.style.cssText = 'height:100%;width:0%;background:#4caf50;';
      outer.appendChild(fill);
      row.append(label, outer);
      this.bars.appendChild(row);
      this.barEls[k] = fill;
    });

    this.root.append(this.canvasFast, this.canvasFull, this.bars);
    this.container.appendChild(this.root);
  }

  _bindWs() {
    this.wsClient?.on('fragility_update', (msg) => this._onUpdate(msg));
  }

  _onUpdate(msg) {
    // Expected: {FI_fast, FI_full, DF, CF, RF, SF, FT, CIS, timestamp_ms}
    Object.assign(this.state, msg);
    this._render();
    this._applyThresholdBehaviors();
  }

  _color(v) {
    if (v > 0.7) return '#f44336';
    if (v > 0.5) return '#ffeb3b';
    return '#4caf50';
  }

  _drawArc(canvas, value, label) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    const cx = w/2, cy = h*0.9, r = Math.min(w,h)*0.7;
    const start = Math.PI;
    const end = 2*Math.PI;
    const angle = start + (end-start)*Math.max(0, Math.min(1, value));

    // background arc
    ctx.lineWidth = 10;
    ctx.strokeStyle = '#333';
    ctx.beginPath(); ctx.arc(cx, cy, r, start, end); ctx.stroke();

    // value arc
    ctx.strokeStyle = this._color(value);
    ctx.beginPath(); ctx.arc(cx, cy, r, start, angle); ctx.stroke();

    ctx.fillStyle = '#ddd';
    ctx.font = '12px sans-serif';
    ctx.fillText(label, 8, 14);
    ctx.font = '16px monospace';
    ctx.fillStyle = this._color(value);
    ctx.fillText(value.toFixed(3), cx - 22, cy - 6);
  }

  _render() {
    this._drawArc(this.canvasFast, this.state.FI_fast, 'FI_fast');
    this._drawArc(this.canvasFull, this.state.FI_full, 'FI_full');

    ['DF','CF','RF','SF','FT','CIS'].forEach((k) => {
      const v = this.state[k] ?? 0;
      const fill = this.barEls[k];
      fill.style.width = `${Math.max(0, Math.min(1, v))*100}%`;
      fill.style.background = this._color(v);
    });
  }

  _applyThresholdBehaviors() {
    if (this.state.FI_fast > 0.6) {
      // OB-reference L1701 behavior
      this.signalBar?.setDetectionThreshold?.(0.40);
    } else {
      this.signalBar?.setDetectionThreshold?.(0.55);
    }

    if (this.state.FI_full > 0.7) {
      // OB-reference L1721-1729 behavior
      this.backtestControls?.setMaxSizeMultiplier?.(0.5);
    } else {
      this.backtestControls?.setMaxSizeMultiplier?.(1.0);
    }
  }
}

if (typeof module !== 'undefined') module.exports = FragilityGauge;
