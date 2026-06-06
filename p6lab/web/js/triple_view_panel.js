/**
 * triple_view_panel.js — §10.1 Triple View Panel
 * P6 Research Lab Frontend
 *
 * Three stacked panels with shared x-axis and shared scrubber:
 *   ┌─────────────────────────────────────────────────────────┐
 *   │ Main chart (40%) — existing lightweight-charts candles   │
 *   ├─────────────────────────────────────────────────────────┤
 *   │ L3 event lane (20%) — per-order timeline: add/mod/cancel/fill │
 *   │   Queue-position bars, lifecycle colors, hover tooltips │
 *   ├─────────────────────────────────────────────────────────┤
 *   │ L2 depth heatmap (20%) — 20-level heat, bid blue/ask red     │
 *   │   Template-match cosine similarity trace on top axis    │
 *   ├─────────────────────────────────────────────────────────┤
 *   │ L1 footprint (20%) — BBO line + spread shading          │
 *   │   Tick-direction-streak + tick-acceleration subplot     │
 *   └─────────────────────────────────────────────────────────┘
 *     shared scrubber ← existing replay_controls.js
 *
 * Data source: WebSocket message type "triple_frame" from server (§11.1)
 * Payload schema matches TripleFrame (§3.1) in JSON form.
 *
 * Replay mode: reads from /api/triple_view endpoint (§11.1)
 * Live mode: computed on-the-fly by engine_runner as frames arrive
 *
 * Activated by: toggle button in existing control_panel.js
 * Integrates with: replay_controls.js (shared scrubber), chart.js,
 *                  depth_overlay.js, websocket_client.js
 *
 * Spec: §10.1 (p6-notebook-lab-spec.md)
 */

class TripleViewPanel {
  /**
   * @param {Object} opts
   * @param {HTMLElement} opts.container - Parent container element
   * @param {Object} opts.wsClient - Existing WebSocket client instance
   * @param {Object} opts.replayControls - Existing replay_controls.js instance
   * @param {Object} opts.chart - Existing chart.js instance (lightweight-charts)
   */
  constructor(opts) {
    this.container = opts.container;
    this.wsClient = opts.wsClient;
    this.replayControls = opts.replayControls;
    this.chart = opts.chart;

    this.active = false;
    this.frameBuffer = [];          // ring buffer for recent triple_frames
    this.bufferMaxSize = 600;       // ~60s at 100ms granularity
    this.currentTimestamp = null;

    // Correlation-match overlay — populated by WebSocket `correlation_match`
    // subscriber via MatchBroker. Drawn on top of the L2 heatmap as tier-
    // colored boxes at match window_start/end ranges.
    this.matchOverlay = [];         // list of {tier, start_ms, end_ms, score, pattern_id}
    this.matchOverlayTtlMs = 30_000;   // fade boxes after 30s

    // Panel DOM elements (created in _buildDOM)
    this.l3Panel = null;
    this.l2Panel = null;
    this.l1Panel = null;

    // Canvas contexts for direct drawing
    this.l3Ctx = null;
    this.l2Ctx = null;
    this.l1Ctx = null;

    this._buildDOM();
    this._bindEvents();
  }

  // ── DOM Construction ──────────────────────────────────────────

  _buildDOM() {
    this.root = document.createElement('div');
    this.root.className = 'triple-view-root';
    this.root.style.cssText = 'display:none; flex-direction:column; height:60%;';

    // L3 Event Lane (top third of our 60%)
    this.l3Panel = this._createPanel('L3 Events', 'triple-view-l3');
    this.l3Canvas = this._createCanvas(this.l3Panel);
    this.l3Ctx = this.l3Canvas.getContext('2d');

    // L2 Depth Heatmap (middle third)
    this.l2Panel = this._createPanel('L2 Depth Heatmap', 'triple-view-l2');
    this.l2Canvas = this._createCanvas(this.l2Panel);
    this.l2Ctx = this.l2Canvas.getContext('2d');

    // L1 Footprint (bottom third)
    this.l1Panel = this._createPanel('L1 Footprint', 'triple-view-l1');
    this.l1Canvas = this._createCanvas(this.l1Panel);
    this.l1Ctx = this.l1Canvas.getContext('2d');

    this.root.append(this.l3Panel, this.l2Panel, this.l1Panel);
    this.container.appendChild(this.root);

    // Tooltip overlay
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'triple-view-tooltip';
    this.tooltip.style.cssText =
      'display:none; position:absolute; background:#1a1a2e; color:#eee; ' +
      'padding:8px 12px; border-radius:4px; font-size:12px; pointer-events:none; ' +
      'border:1px solid #333; z-index:1000; max-width:300px;';
    this.container.appendChild(this.tooltip);
  }

  _createPanel(title, className) {
    const panel = document.createElement('div');
    panel.className = `triple-view-panel ${className}`;
    panel.style.cssText = 'flex:1; position:relative; border-bottom:1px solid #333; overflow:hidden;';

    const label = document.createElement('div');
    label.className = 'triple-view-label';
    label.textContent = title;
    label.style.cssText =
      'position:absolute; top:2px; left:6px; font-size:10px; color:#888; z-index:5;';
    panel.appendChild(label);

    return panel;
  }

  _createCanvas(parent) {
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width:100%; height:100%;';
    parent.appendChild(canvas);
    return canvas;
  }

  // ── Event Binding ─────────────────────────────────────────────

  _bindEvents() {
    // Listen for triple_frame WebSocket messages
    if (this.wsClient) {
      this.wsClient.on('triple_frame', (data) => this._onTripleFrame(data));
      // Correlation-match overlay: subscribe to the same broker stream the
      // Live Signal Dock consumes. Both views co-render the same source.
      this.wsClient.on('correlation_match', (m) => this._onCorrelationMatch(m));
    }

    // Synchronize with replay scrubber (shared x-axis)
    if (this.replayControls) {
      this.replayControls.on('seek', (ts) => this._onSeek(ts));
    }

    // Cross-panel click synchronization
    [this.l3Canvas, this.l2Canvas, this.l1Canvas].forEach((canvas) => {
      canvas.addEventListener('click', (e) => this._onPanelClick(e));
      canvas.addEventListener('mousemove', (e) => this._onPanelHover(e));
      canvas.addEventListener('mouseleave', () => this._hideTooltip());
    });

    // Resize observer for canvas sizing
    const ro = new ResizeObserver(() => this._resizeCanvases());
    ro.observe(this.root);
  }

  // ── Public API ────────────────────────────────────────────────

  /** Toggle triple view on/off. Called by control_panel.js toggle button. */
  toggle() {
    this.active = !this.active;
    this.root.style.display = this.active ? 'flex' : 'none';

    if (this.active) {
      // Shrink main chart to 40% height
      if (this.chart && this.chart.container) {
        this.chart.container.style.height = '40%';
      }
      this._resizeCanvases();
      this._requestInitialData();
    } else {
      // Restore main chart to full height
      if (this.chart && this.chart.container) {
        this.chart.container.style.height = '100%';
      }
    }

    return this.active;
  }

  /** Load triple-view data for a specific time range (replay mode). */
  async loadRange(symbol, startMs, endMs, granularity = '1s') {
    const url = `/api/triple_view?symbol=${encodeURIComponent(symbol)}` +
      `&start_ms=${startMs}&end_ms=${endMs}&granularity=${granularity}`;

    try {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const frames = await resp.json();
      this.frameBuffer = frames;
      this._renderAll();
    } catch (err) {
      console.error('[TripleView] Failed to load range:', err);
    }
  }

  // ── WebSocket Handler ─────────────────────────────────────────

  /**
   * Handle incoming triple_frame WebSocket message.
   *
   * Payload schema (TripleFrame §3.1 as JSON):
   * {
   *   timestamp_ms: int,
   *   symbol: string,
   *   l3_events: [{ order_id, side, price, size, event_type, lifecycle_phase }],
   *   l2_features: [12 floats],
   *   l2_book_vector: [40 floats],
   *   l1_features: [16 floats]
   * }
   */
  _onTripleFrame(frame) {
    if (!this.active) return;

    this.frameBuffer.push(frame);
    if (this.frameBuffer.length > this.bufferMaxSize) {
      this.frameBuffer.shift();
    }

    this.currentTimestamp = frame.timestamp_ms;
    this._renderAll();
  }

  /**
   * Correlation-match overlay handler.
   * Receives each match from MatchBroker via the WebSocket `correlation_match`
   * channel. Entries expire after `matchOverlayTtlMs` to prevent unbounded
   * growth and to fade old signals off-screen.
   */
  _onCorrelationMatch(m) {
    if (!m || typeof m !== 'object') return;
    this.matchOverlay.push({
      tier:          m.tier,
      pattern_id:    m.pattern_id,
      score:         m.ensemble_score,
      start_ms:      m.match_window_start_ms ?? m.timestamp_ms,
      end_ms:        m.match_window_end_ms   ?? m.timestamp_ms,
      received_at:   Date.now(),
    });
    // Drop entries past TTL
    const cutoff = Date.now() - this.matchOverlayTtlMs;
    this.matchOverlay = this.matchOverlay.filter((x) => x.received_at >= cutoff);
    if (this.active) this._renderL2Heatmap();
  }

  // ── Rendering ─────────────────────────────────────────────────

  _renderAll() {
    this._renderL3Events();
    this._renderL2Heatmap();
    this._renderL1Footprint();
  }

  /**
   * L3 Event Lane: per-order timeline.
   * - Horizontal: time (shared x-axis with scrubber)
   * - Vertical: price levels (y-axis aligned with main chart)
   * - Markers: add (green dot), modify (yellow), cancel (red x), fill (green check)
   * - Queue-position bars: horizontal bars showing relative queue depth
   * - Lifecycle colors: birth=blue → active=green → cancelled=red / filled=gold
   * - Hover → tooltip with order_id + full lifecycle
   */
  _renderL3Events() {
    const ctx = this.l3Ctx;
    const { width, height } = this.l3Canvas;
    ctx.clearRect(0, 0, width, height);

    if (!this.frameBuffer.length) return;

    const EVENT_COLORS = {
      add: '#4caf50',       // green
      modify: '#ffeb3b',    // yellow
      cancel: '#f44336',    // red
      fill: '#ffd700',      // gold
    };

    const LIFECYCLE_COLORS = {
      birth: '#2196f3',     // blue
      active: '#4caf50',    // green
      cancelled: '#f44336', // red
      filled: '#ffd700',    // gold
    };

    // TODO: Map frameBuffer l3_events to canvas coordinates
    // Each event: x = timestamp_to_x(event.timestamp_ms), y = price_to_y(event.price)
    // Draw event type marker + queue-position horizontal bar
    // Queue bar width proportional to volume_ahead / total_level_volume

    ctx.fillStyle = '#555';
    ctx.font = '11px monospace';
    ctx.fillText('L3 Events — wire rendering from frameBuffer.l3_events', 10, height / 2);
  }

  /**
   * L2 Depth Heatmap: 20-level heat, bid blue / ask red.
   * - X-axis: time (shared)
   * - Y-axis: 20 price levels centered on mid
   * - Color intensity: depth at each level (normalized by InstrumentNormalizer §3.3)
   * - Bid levels: blue gradient (deeper = brighter)
   * - Ask levels: red gradient (deeper = brighter)
   * - Overlay: template-match cosine similarity trace on top axis
   *   (from l2_book_vector used by TemplateMatcher §5.4)
   */
  _renderL2Heatmap() {
    const ctx = this.l2Ctx;
    const { width, height } = this.l2Canvas;
    ctx.clearRect(0, 0, width, height);

    if (!this.frameBuffer.length) return;

    // TODO: For each frame in visible range:
    // - Extract l2_book_vector (40-dim: 20 bid + 20 ask levels)
    // - Map to heatmap pixels: x=time column, y=level row
    // - Bid levels (0-19): blue intensity = depth / max_depth
    // - Ask levels (20-39): red intensity = depth / max_depth
    // - Draw template-match cosine similarity as line trace on top margin

    ctx.fillStyle = '#555';
    ctx.font = '11px monospace';
    ctx.fillText('L2 Heatmap — wire from frameBuffer.l2_book_vector (40-dim)', 10, height / 2);

    // Correlation-match overlay: tier-colored rectangles spanning each
    // match's window in the time axis. Drawn on top of the heatmap with
    // alpha that fades as the match ages (age/TTL).
    this._renderCorrelationOverlay(ctx, width, height);
  }

  /**
   * Draw tier-colored translucent rectangles on the L2 heatmap for each
   * live correlation match currently in `this.matchOverlay`.
   *
   * Assumes the heatmap's x-axis is time, spanning the visible frameBuffer
   * range. Y-spans the full panel height (matches apply to all levels).
   */
  _renderCorrelationOverlay(ctx, width, height) {
    if (!this.matchOverlay.length || !this.frameBuffer.length) return;

    const TIER_FILL = {
      A: 'rgba(76, 175, 80, 0.25)',     // green
      B: 'rgba(255, 235, 59, 0.20)',    // yellow
      C: 'rgba(158, 158, 158, 0.15)',   // gray
    };
    const TIER_STROKE = {
      A: '#4caf50',
      B: '#ffeb3b',
      C: '#9e9e9e',
    };

    const t0 = this.frameBuffer[0].timestamp_ms;
    const t1 = this.frameBuffer[this.frameBuffer.length - 1].timestamp_ms;
    const span = Math.max(1, t1 - t0);
    const now = Date.now();

    this.matchOverlay.forEach((m) => {
      // Ignore matches outside the visible time window
      if (m.end_ms < t0 || m.start_ms > t1) return;

      const age = now - m.received_at;
      const alpha = Math.max(0.15, 1 - age / this.matchOverlayTtlMs);

      const x0 = ((Math.max(m.start_ms, t0) - t0) / span) * width;
      const x1 = ((Math.min(m.end_ms,   t1) - t0) / span) * width;
      const w  = Math.max(2, x1 - x0);

      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.fillStyle   = TIER_FILL[m.tier]   || TIER_FILL.C;
      ctx.strokeStyle = TIER_STROKE[m.tier] || TIER_STROKE.C;
      ctx.lineWidth   = 1;
      ctx.fillRect(x0, 0, w, height);
      ctx.strokeRect(x0 + 0.5, 0.5, w - 1, height - 1);

      // Compact label: tier + score in the top-left corner of the box
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px system-ui';
      ctx.fillText(
        `${m.tier} ${(m.score ?? 0).toFixed(2)}`,
        x0 + 3, 12,
      );
      ctx.restore();
    });
  }

  /**
   * L1 Footprint: BBO line + spread shading + subplots.
   * - Main area: best_bid and best_ask lines with spread shading between
   * - BBO refresh markers: dots at bid_refresh_rate > threshold
   * - Subplot (bottom 30%): tick_direction_streak (bar chart) +
   *   tick_acceleration (line overlay)
   *
   * L1 features used (indices into l1_features[16]):
   *   [0] spread_ticks, [1] spread_bps_l1, [2] best_bid_size,
   *   [3] best_ask_size, [4] top_imbalance, [5] bid_refresh_rate,
   *   [6] ask_refresh_rate, [7] bid_retreat_velocity,
   *   [8] ask_advance_velocity, [9] spread_compression_rate,
   *   [10] tick_direction_streak, [11] tick_acceleration,
   *   [12] trade_at_bid_ratio, [13] size_spike_ratio,
   *   [14] microprice_velocity, [15] l1_shape_vector
   */
  _renderL1Footprint() {
    const ctx = this.l1Ctx;
    const { width, height } = this.l1Canvas;
    ctx.clearRect(0, 0, width, height);

    if (!this.frameBuffer.length) return;

    // TODO: Draw BBO lines from l1_features[2] (bid_size) and l1_features[3] (ask_size)
    // Spread shading between bid and ask
    // Refresh markers when l1_features[5] or [6] exceed threshold
    // Bottom subplot: tick_direction_streak (l1_features[10]) as bars,
    //   tick_acceleration (l1_features[11]) as line

    ctx.fillStyle = '#555';
    ctx.font = '11px monospace';
    ctx.fillText('L1 Footprint — wire from frameBuffer.l1_features[16]', 10, height / 2);
  }

  // ── Cross-Panel Synchronization ───────────────────────────────

  /** Click any panel at time t → highlight corresponding frames in other two. */
  _onPanelClick(event) {
    const rect = event.target.getBoundingClientRect();
    const xRatio = (event.clientX - rect.left) / rect.width;

    // Map x position to timestamp
    if (!this.frameBuffer.length) return;
    const startTs = this.frameBuffer[0].timestamp_ms;
    const endTs = this.frameBuffer[this.frameBuffer.length - 1].timestamp_ms;
    const clickTs = startTs + Math.round(xRatio * (endTs - startTs));

    // Drive the shared replay scrubber to this timestamp
    if (this.replayControls) {
      this.replayControls.seekTo(clickTs);
    }
  }

  /** Hover: show tooltip with frame data at cursor position. */
  _onPanelHover(event) {
    if (!this.frameBuffer.length) return;

    const rect = event.target.getBoundingClientRect();
    const xRatio = (event.clientX - rect.left) / rect.width;
    const idx = Math.floor(xRatio * this.frameBuffer.length);
    const frame = this.frameBuffer[Math.min(idx, this.frameBuffer.length - 1)];

    if (!frame) return;

    const lines = [
      `Time: ${new Date(frame.timestamp_ms).toISOString().slice(11, 23)}`,
      `L3 events: ${(frame.l3_events || []).length}`,
      `Spread: ${frame.l1_features ? frame.l1_features[1].toFixed(2) : '?'} bps`,
      `Imbalance: ${frame.l1_features ? frame.l1_features[4].toFixed(3) : '?'}`,
    ];

    this.tooltip.innerHTML = lines.join('<br>');
    this.tooltip.style.display = 'block';
    this.tooltip.style.left = (event.clientX + 12) + 'px';
    this.tooltip.style.top = (event.clientY - 40) + 'px';
  }

  _hideTooltip() {
    this.tooltip.style.display = 'none';
  }

  _onSeek(timestampMs) {
    this.currentTimestamp = timestampMs;
    this._renderAll();
  }

  // ── Utility ───────────────────────────────────────────────────

  _resizeCanvases() {
    [this.l3Canvas, this.l2Canvas, this.l1Canvas].forEach((canvas) => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      const ctx = canvas.getContext('2d');
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    });
    if (this.frameBuffer.length) this._renderAll();
  }

  _requestInitialData() {
    // In replay mode, request data for current visible range from /api/triple_view
    // In live mode, frames arrive via WebSocket — no initial request needed
    console.log('[TripleView] Active. Waiting for triple_frame messages or loadRange() call.');
  }

  destroy() {
    this.root.remove();
    this.tooltip.remove();
  }
}

// Export for use by control_panel.js toggle
if (typeof module !== 'undefined') module.exports = TripleViewPanel;
