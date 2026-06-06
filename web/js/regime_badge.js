/**
 * regime_badge.js — Regime badge overlay (top-left of chart).
 * Shows: "TRENDING ▲ 87%" / "RANGING ↔ 72%" / "VOLATILE ⚡ 91%"
 * If abstain=true, shows "⚠ ABSTAIN" in red.
 */
const RegimeBadge = (() => {
  let badgeEl = null;

  /**
   * Initialize the regime badge.
   * @param {HTMLElement} el - The badge DOM element
   */
  function init(el) {
    badgeEl = el;
  }

  /**
   * Update from a new frame.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!badgeEl || !frame) return;

    const rw = frame.regime_weights;
    if (!rw) {
      badgeEl.textContent = 'REGIME: --';
      badgeEl.className = 'badge badge-regime';
      return;
    }

    // Check abstain first
    if (rw.abstain) {
      badgeEl.textContent = '⚠ ABSTAIN';
      badgeEl.className = 'badge badge-regime abstain';
      return;
    }

    const regime = (rw.regime || 'UNKNOWN').toUpperCase();
    const conf = frame.confidence != null ? Math.round(frame.confidence * 100) : '--';

    let icon, cls;
    switch (regime) {
      case 'TRENDING':
        icon = '▲';
        cls = 'trending';
        break;
      case 'RANGING':
        icon = '↔';
        cls = 'ranging';
        break;
      case 'VOLATILE':
        icon = '⚡';
        cls = 'volatile';
        break;
      default:
        icon = '?';
        cls = 'ranging';
    }

    badgeEl.textContent = `${regime} ${icon} ${conf}%`;
    badgeEl.className = `badge badge-regime ${cls}`;
  }

  return { init, updateFromFrame };
})();
