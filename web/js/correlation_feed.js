/**
 * correlation_feed.js — L6 Pattern Correlation sidebar feed.
 *
 * Consumes ``frame.correlation_matches`` (populated server-side by the
 * Wave 5 Phase 5A thesis-chain wire) and renders the top-N active pattern
 * matches with tier color + expected direction + score. Mirrors TapeFeed's
 * prepend-newest / trim-oldest shape; de-duplicates adjacent repeats of the
 * same pattern_id so low-match-rate patterns don't hammer the list.
 */
const CorrelationFeed = (() => {
  let listEl = null;
  let emptyEl = null;
  const MAX_ROWS = 40;
  let lastKey = '';

  function init(containerEl) {
    listEl = containerEl;
    if (!listEl) return;
    emptyEl = document.createElement('div');
    emptyEl.className = 'corr-empty';
    emptyEl.textContent = 'awaiting matches…';
    listEl.appendChild(emptyEl);
  }

  function reset() {
    if (!listEl) return;
    listEl.innerHTML = '';
    lastKey = '';
    if (emptyEl) listEl.appendChild(emptyEl);
  }

  function updateFromFrame(frame) {
    if (!frame || !listEl) return;
    const matches = frame.correlation_matches;
    if (!Array.isArray(matches) || matches.length === 0) return;

    for (const m of matches) {
      if (!m) continue;
      const key = `${m.pattern_id}|${m.match_window_end_ms}`;
      if (key === lastKey) continue;
      lastKey = key;
      _prependRow(m, frame.timestamp_ms);
    }

    while (listEl.children.length > MAX_ROWS) {
      listEl.removeChild(listEl.lastChild);
    }
    if (emptyEl && emptyEl.parentNode === listEl && listEl.children.length > 1) {
      listEl.removeChild(emptyEl);
    }
  }

  function _prependRow(match, frameTs) {
    const row = document.createElement('div');
    const tier = String(match.confidence_tier || 'C').toUpperCase();
    row.className = `corr-row tier-${tier.toLowerCase()}`;

    const ts = match.match_window_end_ms || frameTs || Date.now();
    const t = new Date(ts).toLocaleTimeString('en-US', {
      timeZone: 'America/New_York', hour12: false,
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });

    const dir = String(match.expected_direction || 'neutral');
    const dirChar = dir === 'bull' ? '▲' : (dir === 'bear' ? '▼' : '–');
    const score = (Number(match.ensemble_score) || 0).toFixed(3);
    const atr = (Number(match.expected_move_atr) || 0).toFixed(2);
    const pid = String(match.pattern_id || '?');
    const shortPid = pid.length > 26 ? pid.slice(0, 25) + '…' : pid;

    row.innerHTML =
      `<span class="corr-time">${t}</span>` +
      `<span class="corr-tier">${tier}</span>` +
      `<span class="corr-pid" title="${pid}">${shortPid}</span>` +
      `<span class="corr-dir corr-dir-${dir}">${dirChar}</span>` +
      `<span class="corr-score">${score}</span>` +
      `<span class="corr-atr">${atr} ATR</span>`;

    listEl.insertBefore(row, listEl.firstChild);
  }

  return { init, reset, updateFromFrame };
})();
