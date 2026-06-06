/**
 * force_arrows.js — Institutional force direction arrow overlay.
 * Shows arrow ▲/▼ with magnitude and institutional score.
 */
const ForceArrows = (() => {
  let arrowEl = null;
  let labelEl = null;

  /**
   * Initialize the force arrow overlay.
   * @param {HTMLElement} arrowElement - Element for the arrow character
   * @param {HTMLElement} labelElement - Element for the label text
   */
  function init(arrowElement, labelElement) {
    arrowEl = arrowElement;
    labelEl = labelElement;
  }

  /**
   * Update from a new frame.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!frame || !frame.force_vector) {
      _clear();
      return;
    }

    const fv = frame.force_vector;
    const force = fv.total_force != null ? fv.total_force : 0;
    const instScore = fv.institutional_score != null ? fv.institutional_score : 0;
    const isBullish = force >= 0;

    if (arrowEl) {
      arrowEl.textContent = isBullish ? '▲' : '▼';
      arrowEl.className = 'force-arrow ' + (isBullish ? 'bullish' : 'bearish');
      // Scale font size based on magnitude (16px to 28px)
      const mag = Math.min(Math.abs(force), 1);
      arrowEl.style.fontSize = (16 + mag * 12) + 'px';
    }

    if (labelEl) {
      const sign = force >= 0 ? '+' : '';
      const pct = Math.round(instScore * 100);
      labelEl.textContent = `INST FORCE: ${sign}${force.toFixed(2)} (${pct}%)`;
    }
  }

  /** Clear display when no data */
  function _clear() {
    if (arrowEl) { arrowEl.textContent = '-'; arrowEl.className = 'force-arrow'; }
    if (labelEl) { labelEl.textContent = 'INST FORCE: --'; }
  }

  return { init, updateFromFrame };
})();
