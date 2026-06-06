/**
 * correlation_feed.js — Live Signal Dock (spec §10.4, roadmap Q2).
 *
 * Chart-independent live view of engine match events. Subscribes to the
 * WebSocket `correlation_match` channel (ultimately fed by the in-process
 * `MatchBroker` on the server) and renders a polished signal dock:
 *
 *   - top-row tier counters with 60s sparklines (A/B/C)
 *   - per-match card with pattern_id, tier badge, direction, ATR badge
 *   - click-to-expand score breakdown (template / mahalanobis / contextual)
 *   - filter bar: tier × instrument × regime
 *   - audio alert toggle on tier A
 *   - pin button: pinned matches stay at the top
 *   - connection-health dot (WebSocket state + match-age)
 *
 * The dock works standalone — it does not require `triple_view_panel.js`
 * to be mounted. A future chart-level overlay subscribes to the same
 * WebSocket channel; both coexist as peer renderers of the broker stream.
 */

const DEFAULT_OPTS = {
  maxRows: 500,
  histogramWindowMs: 60_000,
  healthStaleMs: 15_000,
  pingSrc: null, // optional URL to an audio file for tier-A pings
};

class CorrelationFeed {
  constructor({ container, wsClient, replayControls, tripleView, options } = {}) {
    this.container = container;
    this.wsClient = wsClient;
    this.replayControls = replayControls;
    this.tripleView = tripleView;
    this.opts = { ...DEFAULT_OPTS, ...(options || {}) };

    this.matches = [];          // newest at end, cap opts.maxRows
    this.pinned = new Map();    // key: `${ts}|${pattern_id}` → match
    this.filters = { A: true, B: true, C: true, instrument: 'all', regime: 'all' };
    this.expanded = new Set();  // match keys currently expanded
    this.audioOn = false;
    this.lastMatchAt = null;    // Date.now() of most-recent received match
    this.wsAlive = Boolean(this.wsClient);

    this._buildUI();
    this._bindWs();
    this._tick = setInterval(() => this._renderHealth(), 2_000);
  }

  destroy() {
    clearInterval(this._tick);
    if (this.wsClient?.off) this.wsClient.off('correlation_match', this._onMatchBound);
    this.root?.remove();
  }

  // --------------------------------------------------------------------
  // UI construction
  // --------------------------------------------------------------------

  _buildUI() {
    this.root = document.createElement('div');
    this.root.className = 'signal-dock';

    // Header row: title + health dot + audio toggle
    const hdr = document.createElement('div');
    hdr.className = 'signal-dock__header';
    hdr.innerHTML = `
      <span class="signal-dock__title">Live Signal Dock</span>
      <span class="signal-dock__health" title="WebSocket health"></span>
      <label class="signal-dock__audio">
        <input type="checkbox" />
        <span>Alert on tier A</span>
      </label>
    `;
    this.health = hdr.querySelector('.signal-dock__health');
    this.audioToggle = hdr.querySelector('input[type=checkbox]');
    this.audioToggle.addEventListener('change', (e) => { this.audioOn = e.target.checked; });

    // Tier summary row
    this.summary = document.createElement('div');
    this.summary.className = 'signal-dock__summary';

    // Sparkline row
    this.hist = document.createElement('canvas');
    this.hist.className = 'signal-dock__sparkline';
    this.hist.height = 40;

    // Filter bar
    this.filterBar = document.createElement('div');
    this.filterBar.className = 'signal-dock__filters';
    this.filterBar.innerHTML = `
      <label><input type="checkbox" data-tier="A" checked>A</label>
      <label><input type="checkbox" data-tier="B" checked>B</label>
      <label><input type="checkbox" data-tier="C" checked>C</label>
      <select data-filter="instrument"><option value="all">All instruments</option></select>
      <select data-filter="regime"><option value="all">All regimes</option></select>
    `;
    this.filterBar.addEventListener('change', (e) => this._onFilterChange(e));

    // Pinned + scrolling list
    this.pinnedList = document.createElement('div');
    this.pinnedList.className = 'signal-dock__pinned';
    this.list = document.createElement('div');
    this.list.className = 'signal-dock__list';

    this.root.append(hdr, this.summary, this.hist, this.filterBar,
                     this.pinnedList, this.list);
    this.container.appendChild(this.root);

    this._renderHealth();
    this._renderSummary();
  }

  // --------------------------------------------------------------------
  // WebSocket wiring
  // --------------------------------------------------------------------

  _bindWs() {
    if (!this.wsClient?.on) return;
    this._onMatchBound = (m) => this._onMatch(m);
    this.wsClient.on('correlation_match', this._onMatchBound);
    // Optional connection-state hooks (graceful if missing)
    this.wsClient.on?.('open',  () => { this.wsAlive = true;  this._renderHealth(); });
    this.wsClient.on?.('close', () => { this.wsAlive = false; this._renderHealth(); });
  }

  _onMatch(match) {
    if (!match || typeof match !== 'object') return;
    this.lastMatchAt = Date.now();

    this.matches.push(match);
    if (this.matches.length > this.opts.maxRows) this.matches.shift();

    // Populate filter dropdowns as new instruments/regimes appear
    this._ensureFilterOption('instrument', match.instrument);
    this._ensureFilterOption('regime',     match.regime);

    // Audio alert for tier A (if enabled)
    if (this.audioOn && match.tier === 'A' && this.opts.pingSrc) {
      try { new Audio(this.opts.pingSrc).play(); } catch { /* user-gesture gating */ }
    }

    this._renderSummary();
    this._renderSparkline();
    this._renderList();
    this._renderHealth();
  }

  // --------------------------------------------------------------------
  // Filter management
  // --------------------------------------------------------------------

  _onFilterChange(e) {
    const el = e.target;
    if (el.dataset.tier) {
      this.filters[el.dataset.tier] = el.checked;
    } else if (el.dataset.filter) {
      this.filters[el.dataset.filter] = el.value;
    }
    this._renderList();
    this._renderSummary();
  }

  _ensureFilterOption(kind, value) {
    if (!value) return;
    const sel = this.filterBar.querySelector(`select[data-filter="${kind}"]`);
    if (!sel) return;
    if (![...sel.options].some((o) => o.value === value)) {
      const opt = document.createElement('option');
      opt.value = value;
      opt.textContent = value;
      sel.appendChild(opt);
    }
  }

  _passesFilter(m) {
    if (!this.filters[m.tier]) return false;
    if (this.filters.instrument !== 'all' && m.instrument !== this.filters.instrument) return false;
    if (this.filters.regime !== 'all' && m.regime !== this.filters.regime) return false;
    return true;
  }

  // --------------------------------------------------------------------
  // Rendering
  // --------------------------------------------------------------------

  _renderSummary() {
    const cutoff = Date.now() - this.opts.histogramWindowMs;
    const counts = { A: 0, B: 0, C: 0 };
    this.matches.forEach((m) => {
      if (m.timestamp_ms < cutoff) return;
      if (!this._passesFilter(m)) return;
      if (counts[m.tier] !== undefined) counts[m.tier] += 1;
    });
    this.summary.innerHTML = `
      <div class="tier-pill tier-pill--A"><span class="dot"></span>A <b>${counts.A}</b></div>
      <div class="tier-pill tier-pill--B"><span class="dot"></span>B <b>${counts.B}</b></div>
      <div class="tier-pill tier-pill--C"><span class="dot"></span>C <b>${counts.C}</b></div>
      <div class="tier-pill tier-pill--total">60s total <b>${counts.A + counts.B + counts.C}</b></div>
    `;
  }

  _renderSparkline() {
    const canvas = this.hist;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth || 300;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const bins = 30;
    const binMs = this.opts.histogramWindowMs / bins;
    const now = Date.now();
    const counts = Array.from({ length: bins }, () => ({ A: 0, B: 0, C: 0 }));

    this.matches.forEach((m) => {
      if (m.timestamp_ms < now - this.opts.histogramWindowMs) return;
      if (!this._passesFilter(m)) return;
      const idx = Math.min(bins - 1,
        Math.floor((m.timestamp_ms - (now - this.opts.histogramWindowMs)) / binMs));
      if (counts[idx][m.tier] !== undefined) counts[idx][m.tier] += 1;
    });

    const max = Math.max(1, ...counts.map((c) => c.A + c.B + c.C));
    const bw = w / bins;
    counts.forEach((c, i) => {
      let y = h;
      [['C', '#9e9e9e'], ['B', '#ffeb3b'], ['A', '#4caf50']].forEach(([tier, color]) => {
        const bh = (c[tier] / max) * (h - 2);
        ctx.fillStyle = color;
        ctx.fillRect(i * bw + 0.5, y - bh, Math.max(1, bw - 1), bh);
        y -= bh;
      });
    });
  }

  _renderList() {
    // Pinned zone (always on top)
    this.pinnedList.innerHTML = '';
    this.pinned.forEach((m, key) => {
      this.pinnedList.appendChild(this._buildRow(m, key, { pinned: true }));
    });

    // Scrolling zone: newest first, up to 100 visible matches
    this.list.innerHTML = '';
    [...this.matches].reverse().forEach((m) => {
      if (!this._passesFilter(m)) return;
      const key = this._keyOf(m);
      if (this.pinned.has(key)) return; // don't duplicate
      this.list.appendChild(this._buildRow(m, key, { pinned: false }));
      if (this.list.childElementCount >= 100) return;
    });
  }

  _buildRow(m, key, { pinned }) {
    const row = document.createElement('div');
    row.className = `signal-row signal-row--${m.tier}` + (pinned ? ' signal-row--pinned' : '');

    const ts = new Date(m.timestamp_ms).toISOString().slice(11, 23);
    const arrow = m.expected_direction === 'bull' ? '▲'
                : m.expected_direction === 'bear' ? '▼' : '→';
    const atr = typeof m.expected_move_atr === 'number'
      ? `${m.expected_move_atr.toFixed(2)} ATR` : '';

    row.innerHTML = `
      <span class="tier-badge tier-badge--${m.tier}">${m.tier}</span>
      <span class="pattern-id">${m.pattern_id}</span>
      <span class="direction">${arrow}</span>
      <span class="score">${(m.ensemble_score ?? 0).toFixed(3)}</span>
      <span class="atr">${atr}</span>
      <span class="ts" title="${m.timestamp_ms}">${ts}</span>
      <button class="pin-btn" title="${pinned ? 'Unpin' : 'Pin'}">${pinned ? '📌' : '📍'}</button>
    `;

    row.querySelector('.pin-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      if (pinned) this.pinned.delete(key); else this.pinned.set(key, m);
      this._renderList();
    });

    row.addEventListener('click', () => {
      if (this.expanded.has(key)) this.expanded.delete(key);
      else this.expanded.add(key);
      // Jump the replay chart if available, but don't hard-couple to it
      this.replayControls?.seekTo?.(m.timestamp_ms);
      this._renderList();
    });

    if (this.expanded.has(key)) {
      const detail = document.createElement('div');
      detail.className = 'signal-row__detail';
      detail.innerHTML = `
        <div>template similarity: <b>${(m.template_similarity ?? NaN).toFixed(3)}</b></div>
        <div>mahalanobis: <b>${(m.mahalanobis_score ?? NaN).toFixed(3)}</b></div>
        <div>contextual: <b>${(m.contextual_score ?? NaN).toFixed(3)}</b></div>
        <div>stage1 prescreen: <b>${(m.stage1_score ?? NaN).toFixed(3)}</b></div>
        <div>regime: <b>${m.regime ?? '—'}</b>   instrument: <b>${m.instrument ?? '—'}</b></div>
        <div>window: ${m.match_window_start_ms} → ${m.match_window_end_ms}</div>
      `;
      row.appendChild(detail);
    }
    return row;
  }

  _renderHealth() {
    if (!this.health) return;
    const age = this.lastMatchAt ? Date.now() - this.lastMatchAt : Infinity;
    let cls = 'health--red', title = 'no WebSocket';
    if (this.wsAlive) {
      if (age < this.opts.healthStaleMs) { cls = 'health--green'; title = `last match ${(age/1000).toFixed(1)}s ago`; }
      else                                { cls = 'health--yellow'; title = `stale — last match ${(age/1000).toFixed(0)}s ago`; }
    }
    this.health.className = `signal-dock__health ${cls}`;
    this.health.title = title;
  }

  // --------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------

  _keyOf(m) {
    return `${m.timestamp_ms}|${m.pattern_id}`;
  }
}

if (typeof module !== 'undefined') module.exports = CorrelationFeed;
