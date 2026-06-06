/**
 * cup_flip_badge.js — Game state badge (below regime badge).
 * Shows: "BULL STREAK (5)" / "BEAR STREAK (3)" / "STALL" / "⚠ STOP RUN"
 * Color coded and with streak velocity gauge.
 */
const CupFlipBadge = (() => {
  let badgeEl = null;
  let velocityFillEl = null;

  /**
   * Initialize the cup flip badge.
   * @param {HTMLElement} el - Badge element
   * @param {HTMLElement} fillEl - Velocity gauge fill element
   */
  function init(el, fillEl) {
    badgeEl = el;
    velocityFillEl = fillEl;
  }

  /**
   * Update from a new frame.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!badgeEl || !frame) return;

    const gs = frame.game_state;
    if (!gs) {
      badgeEl.className = 'badge badge-game';
      _setText('STATE: --');
      _setVelocity(0, '#888');
      return;
    }

    const state = (gs.state || 'BALANCED').toUpperCase();
    const streakLen = gs.streak_length != null ? gs.streak_length : 0;
    const velocity = gs.streak_velocity != null ? gs.streak_velocity : 0;

    let text, cls, velColor;

    if (state.includes('BULL')) {
      text = `BULL STREAK (${streakLen})`;
      cls = 'bull';
      velColor = '#2ecc71';
    } else if (state.includes('BEAR')) {
      text = `BEAR STREAK (${streakLen})`;
      cls = 'bear';
      velColor = '#ff6b6b';
    } else if (state.includes('STALL') || state === 'BALANCED') {
      text = state === 'BALANCED' ? 'BALANCED' : 'STALL';
      cls = 'stall';
      velColor = '#f39c12';
    } else if (state.includes('STOP_RUN') || state.includes('STOP RUN')) {
      text = '⚠ STOP RUN';
      cls = 'stop-run';
      velColor = '#e74c3c';
    } else {
      text = state;
      cls = 'stall';
      velColor = '#888';
    }

    badgeEl.className = `badge badge-game ${cls}`;
    _setText(text);

    // Velocity gauge: normalize to 0-100% (assume max ~20 fills/sec)
    const pct = Math.min(100, (velocity / 20) * 100);
    _setVelocity(pct, velColor);
  }

  /** Set badge text content (excluding gauge) */
  function _setText(text) {
    // Keep the velocity gauge span, update only text
    const gaugeHTML = velocityFillEl ? velocityFillEl.parentElement.outerHTML : '';
    badgeEl.innerHTML = text + ' ' + gaugeHTML;
    // Re-cache velocity fill after innerHTML change
    if (badgeEl.querySelector('.velocity-fill')) {
      velocityFillEl = badgeEl.querySelector('.velocity-fill');
    }
  }

  /** Set velocity gauge fill width and color */
  function _setVelocity(pct, color) {
    if (!velocityFillEl) return;
    velocityFillEl.style.width = pct + '%';
    velocityFillEl.style.background = color;
  }

  return { init, updateFromFrame };
})();
