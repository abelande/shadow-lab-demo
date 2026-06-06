/**
 * stats_header.js — Top stats bar.
 * Displays: CVD, T/s, ADD, CXL, MOD, FIL, LIVE from frame.stats
 */
const StatsHeader = (() => {
  let els = {};

  /**
   * Initialize stats header by caching element references.
   */
  function init() {
    els = {
      symbol: document.getElementById('stat-symbol'),
      cvd: document.getElementById('stat-cvd'),
      tps: document.getElementById('stat-tps'),
      add: document.getElementById('stat-add'),
      cxl: document.getElementById('stat-cxl'),
      mod: document.getElementById('stat-mod'),
      fil: document.getElementById('stat-fil'),
      live: document.getElementById('stat-live'),
      connDot: document.getElementById('conn-dot'),
      connText: document.getElementById('conn-text'),
    };
  }

  /**
   * Update stats from a new frame.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!frame) return;

    // Symbol
    if (els.symbol && frame.symbol) {
      els.symbol.textContent = frame.symbol;
    }

    const stats = frame.stats || {};

    _updateStat(els.cvd, stats.cvd, true);
    _updateStat(els.tps, stats.trades_per_sec);
    _updateStat(els.add, stats.add_count);
    _updateStat(els.cxl, stats.cancel_count);
    _updateStat(els.mod, stats.modify_count);
    _updateStat(els.fil, stats.fill_count);
    _updateStat(els.live, stats.live_orders);
  }

  /**
   * Update a single stat element.
   * @param {HTMLElement} el - The stat value element
   * @param {number|null} value - The value to display
   * @param {boolean} colorize - Apply positive/negative coloring
   */
  function _updateStat(el, value, colorize) {
    if (!el) return;
    if (value == null) {
      el.textContent = '--';
      el.className = 'stat-value';
      return;
    }
    el.textContent = typeof value === 'number' ? value.toLocaleString() : value;
    if (colorize) {
      el.className = 'stat-value ' + (value >= 0 ? 'positive' : 'negative');
    }
  }

  /**
   * Update connection status indicator.
   * @param {boolean} connected
   */
  function setConnected(connected) {
    if (els.connDot) {
      els.connDot.className = connected ? 'connection-dot connected' : 'connection-dot';
    }
    if (els.connText) {
      els.connText.textContent = connected ? 'LIVE' : 'DISCONNECTED';
    }
  }

  return { init, updateFromFrame, setConnected };
})();
