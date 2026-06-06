/**
 * signal_bar.js — Bottom signal summary bar.
 * Shows aggregated signal: direction, confidence, urgency, size recommendation.
 */
const SignalBar = (() => {
  let dirEl = null;
  let confEl = null;
  let urgEl = null;
  let sizeEl = null;
  let forceArrowEl = null;
  let forceValueEl = null;
  let forceFillEl = null;

  /**
   * Initialize the signal bar.
   * @param {HTMLElement} directionEl - Direction indicator element
   * @param {HTMLElement} confidenceEl - Confidence text element
   * @param {HTMLElement} urgencyEl - Urgency badge element
   * @param {HTMLElement} sizeEl_ - Size recommendation element
   */
  function init(directionEl, confidenceEl, urgencyEl, sizeEl_) {
    dirEl = directionEl;
    confEl = confidenceEl;
    urgEl = urgencyEl;
    sizeEl = sizeEl_;
    forceArrowEl = document.getElementById('sig-force-arrow');
    forceValueEl = document.getElementById('sig-force-value');
    forceFillEl = document.getElementById('sig-force-fill');
  }

  /**
   * Update from a new frame.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!frame) return;

    const direction = frame.direction != null ? frame.direction : 0;
    const confidence = frame.confidence != null ? frame.confidence : 0;
    const urgency = frame.urgency != null ? frame.urgency : 0;
    const abstain = frame.regime_weights && frame.regime_weights.abstain;

    // Direction
    if (dirEl) {
      if (abstain || confidence < 0.3) {
        dirEl.textContent = '⚪ FLAT';
        dirEl.className = 'signal-direction flat';
      } else if (direction > 0.1) {
        dirEl.textContent = '🟢 BUY';
        dirEl.className = 'signal-direction buy';
      } else if (direction < -0.1) {
        dirEl.textContent = '🔴 SELL';
        dirEl.className = 'signal-direction sell';
      } else {
        dirEl.textContent = '⚪ FLAT';
        dirEl.className = 'signal-direction flat';
      }
    }

    // Confidence
    if (confEl) {
      const pct = Math.round(confidence * 100);
      confEl.textContent = `Confidence: ${pct}%`;
    }

    // Urgency
    if (urgEl) {
      let label, cls;
      if (urgency >= 0.7) { label = 'HIGH'; cls = 'high'; }
      else if (urgency >= 0.4) { label = 'MEDIUM'; cls = 'medium'; }
      else { label = 'LOW'; cls = 'low'; }

      if (abstain) { label = 'ABSTAIN'; cls = 'low'; }

      urgEl.textContent = `Urgency: ${label}`;
      urgEl.className = `signal-urgency ${cls}`;
    }

    // Size recommendation based on confidence
    if (sizeEl) {
      if (abstain) {
        sizeEl.textContent = '';
      } else {
        const size = (0.5 + confidence * 1.0).toFixed(1);
        sizeEl.textContent = `Size: ${size}x`;
      }
    }

    // Institutional force gauge
    _updateForce(frame);
  }

  /**
   * Render institutional force in the signal bar.
   */
  function _updateForce(frame) {
    const fv = frame.force_vector;
    if (!fv) {
      if (forceArrowEl) forceArrowEl.textContent = '-';
      if (forceValueEl) forceValueEl.textContent = 'FORCE: --';
      if (forceFillEl) { forceFillEl.style.width = '0%'; forceFillEl.style.background = '#888'; }
      return;
    }

    const force = fv.total_force != null ? fv.total_force : 0;
    const instScore = fv.institutional_score != null ? fv.institutional_score : 0;
    const isBullish = force >= 0;

    // Arrow
    if (forceArrowEl) {
      forceArrowEl.textContent = isBullish ? '▲' : '▼';
      forceArrowEl.className = 'force-arrow-inline ' + (isBullish ? 'bullish' : 'bearish');
    }

    // Value text — show magnitude + institutional dominance %
    if (forceValueEl) {
      const sign = force >= 0 ? '+' : '';
      const instPct = Math.round(instScore * 100);
      forceValueEl.textContent = `FORCE: ${sign}${force.toFixed(2)} · ${instPct}% inst`;
    }

    // Gauge bar — fills based on absolute magnitude (cap at 1.0)
    if (forceFillEl) {
      const mag = Math.min(1, Math.abs(force));
      const pct = Math.round(mag * 100);
      forceFillEl.style.width = pct + '%';
      // Color: green if bullish, red if bearish, intensity by magnitude
      if (isBullish) {
        forceFillEl.style.background = `rgba(46, 204, 113, ${0.5 + mag * 0.5})`;
      } else {
        forceFillEl.style.background = `rgba(231, 76, 60, ${0.5 + mag * 0.5})`;
      }
    }
  }

  return { init, updateFromFrame };
})();
