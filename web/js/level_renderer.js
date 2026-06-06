/**
 * level_renderer.js — Renders order book levels as filled horizontal bars
 * anchored to the right side of the chart (next to the price axis).
 *
 * Replaces the old priceLine-based renderer with canvas-drawn rectangles
 * that show resting bid/ask liquidity at each price level. Bar width is
 * proportional to volume (normalized across visible levels), color encodes
 * side (blue=bid, red=ask), and opacity/animation encodes lifecycle state.
 *
 * Uses the chart's priceToCoordinate() for Y positioning so bars pan and
 * zoom natively with the chart.
 */
const LevelRenderer = (() => {
  let _chart = null;
  let _candleSeries = null;
  let _canvas = null;
  let _ctx = null;

  // Track 3D: Feature flag — enabled by default
  let _enabled = true;

  // Current level data
  let _levels = [];

  // Fade-out tracking for BROKEN/PULLED levels
  // Map: "price|side" -> { level, fadeFrame, maxFadeFrames }
  const _fading = new Map();
  const MAX_FADE_FRAMES = 6;

  // Pulse state for breathing animation
  let _pulsePhase = 0;
  let _animFrameId = null;
  let _debugCount = 0;

  // Colors
  const COLOR = {
    BID_RESTING:  { r: 41,  g: 128, b: 185 },  // #2980b9 deep blue
    BID_FORMING:  { r: 52,  g: 152, b: 219 },  // #3498db light blue
    BID_TESTED:   { r: 93,  g: 173, b: 226 },  // #5dade2 pale blue
    BID_DEFENDED: { r: 93,  g: 173, b: 226 },  // #5dade2 bright
    ASK_RESTING:  { r: 192, g: 57,  b: 43  },  // #c0392b deep red
    ASK_FORMING:  { r: 231, g: 76,  b: 60  },  // #e74c3c standard red
    ASK_TESTED:   { r: 231, g: 76,  b: 60  },  // #e74c3c
    ASK_DEFENDED: { r: 231, g: 76,  b: 60  },  // #e74c3c bright
    SPOOF:        { r: 230, g: 126, b: 34  },  // #e67e22 orange
    FADING:       { r: 100, g: 100, b: 100 },  // grey
  };

  /**
   * Initialize the renderer with a chart, series, and canvas element.
   * @param {object} chart - LightweightCharts chart instance
   * @param {object} candleSeries - Candlestick series for priceToCoordinate()
   * @param {HTMLCanvasElement} [canvasEl] - Canvas to draw on (defaults to #depth-overlay)
   */
  function init(chart, candleSeries, canvasEl) {
    _chart = chart;
    _candleSeries = candleSeries;
    _canvas = canvasEl || document.getElementById('depth-overlay');

    if (_canvas) {
      _ctx = _canvas.getContext('2d');
      _resize();
      window.addEventListener('resize', _resize);
      const chartContainer = document.getElementById('chart-container');
      if (chartContainer) {
        new ResizeObserver(_resize).observe(chartContainer);
      }
    }

    // Re-render on zoom/pan
    if (chart) {
      const ts = chart.timeScale();
      if (typeof ts.subscribeVisibleLogicalRangeChange === 'function') {
        ts.subscribeVisibleLogicalRangeChange(() => render());
      }
    }

    // Start animation loop for breathing effect
    _startAnimLoop();
    console.log('[LevelRenderer] Initialized. canvas:', !!_canvas, 'chart:', !!_chart, 'series:', !!_candleSeries);
  }

  function _resize() {
    if (!_canvas) return;
    const chartContainer = document.getElementById('chart-container');
    if (!chartContainer) return;
    _canvas.width = chartContainer.clientWidth;
    _canvas.height = chartContainer.clientHeight;
    render();
  }

  /**
   * Update levels from a frame (works for both live and replay).
   * Accepts either frame.level_states (live) or frame.levels (replay).
   * @param {Array} levels - Array of LevelState-like objects
   */
  function updateFromLevels(levels) {
    if (!Array.isArray(levels)) return;
    if (levels.length > 0 && _debugCount < 5) {
      console.log('[LevelRenderer] Received', levels.length, 'levels. First:', JSON.stringify(levels[0]));
      _debugCount++;
    }

    const incomingKeys = new Set();
    for (const lvl of levels) {
      incomingKeys.add(_key(lvl));
    }

    // Move disappeared levels to fading if they were visible
    for (const prev of _levels) {
      const k = _key(prev);
      if (!incomingKeys.has(k) && !_fading.has(k)) {
        const lc = prev.lifecycle;
        if (lc !== 'FORMING') {
          _fading.set(k, { level: prev, fadeFrame: 0, maxFadeFrames: MAX_FADE_FRAMES });
        }
      }
    }

    _levels = levels;
    // render() is called by the animation loop
  }

  /**
   * Update from a full DepthIndicatorFrame (convenience for live mode).
   * @param {object} frame
   */
  function updateFromFrame(frame) {
    if (!frame) return;
    const levels = frame.level_states || frame.levels || [];
    if (_debugCount < 3) {
      console.log('[LevelRenderer] updateFromFrame called. level_states:', (frame.level_states || []).length,
        'levels:', (frame.levels || []).length, 'canvas:', !!_canvas, 'series:', !!_candleSeries);
    }
    updateFromLevels(levels);
  }

  function _key(lvl) {
    return `${lvl.price}|${lvl.side}`;
  }

  /** Animation loop — drives breathing pulse and re-renders. */
  function _startAnimLoop() {
    if (_animFrameId) return;
    function tick() {
      _pulsePhase += 0.04; // ~60fps → full cycle every ~157 frames (~2.6s)
      render();
      _animFrameId = requestAnimationFrame(tick);
    }
    _animFrameId = requestAnimationFrame(tick);
  }

  /**
   * Main render — draws all active levels + fading levels as filled bars.
   * Track 3D: Respects _enabled flag — clears canvas and returns if disabled.
   */
  function render() {
    if (!_ctx || !_canvas || !_candleSeries) return;
    _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
    if (!_enabled) return;  // Track 3D: feature flag

    const allLevels = [..._levels];
    if (allLevels.length === 0 && _fading.size === 0) return;

    // Layout: bars anchor to right edge, extend left
    const priceScaleWidth = Math.min(80, Math.max(50, Math.round(_canvas.width * 0.068)));
    const rightEdge = _canvas.width - priceScaleWidth;
    const maxBarWidth = Math.min(rightEdge * 0.35, 300); // max 35% of chart width
    const barHeight = 8;

    // Normalize volume across visible levels
    const maxVol = Math.max(...allLevels.map(l => l.volume || 0), 1);

    // Draw active levels
    for (const lvl of allLevels) {
      _drawLevel(lvl, rightEdge, maxBarWidth, barHeight, maxVol, false, 0);
    }

    // Draw fading levels
    const toRemove = [];
    for (const [key, entry] of _fading.entries()) {
      entry.fadeFrame++;
      if (entry.fadeFrame > entry.maxFadeFrames) {
        toRemove.push(key);
        continue;
      }
      _drawLevel(entry.level, rightEdge, maxBarWidth, barHeight, maxVol, true, entry.fadeFrame / entry.maxFadeFrames);
    }
    for (const k of toRemove) _fading.delete(k);
  }

  /**
   * Draw a single level bar.
   * @param {object} lvl - LevelState-like object
   * @param {number} rightEdge - X position of the right edge (before price scale)
   * @param {number} maxBarWidth - Maximum bar width in pixels
   * @param {number} barHeight - Bar height in pixels
   * @param {number} maxVol - Max volume for normalization
   * @param {boolean} isFading - Whether this level is in fade-out
   * @param {number} fadeProgress - 0-1, how far through the fade
   */
  function _drawLevel(lvl, rightEdge, maxBarWidth, barHeight, maxVol, isFading, fadeProgress) {
    if (!lvl || lvl.price == null) return;

    // Y from chart price scale
    let y;
    try {
      y = _candleSeries.priceToCoordinate(lvl.price);
    } catch (e) {
      return;
    }
    if (y == null || !Number.isFinite(y) || y < -barHeight || y > _canvas.height + barHeight) return;

    const isBid = lvl.side === 'BID';
    const lc = lvl.lifecycle || 'RESTING';
    const sig = lvl.significance || 0.5;
    const auth = lvl.authenticity != null ? lvl.authenticity : 1.0;
    const isSpoof = lvl.spoof_type != null || auth < 0.5;
    const isIceberg = lvl.iceberg_suspected === true;

    // Bar width: volume-proportional, scaled by significance
    const volRatio = Math.min(1, (lvl.volume || 0) / maxVol);
    const barWidth = Math.max(4, volRatio * maxBarWidth * (0.5 + sig * 0.5));
    const x0 = rightEdge - barWidth;

    // Determine color and opacity
    let color, opacity;

    if (isFading) {
      color = COLOR.FADING;
      opacity = 0.4 * (1 - fadeProgress);
    } else if (isSpoof) {
      color = COLOR.SPOOF;
      opacity = _spoofOpacity(lc);
    } else {
      const colorKey = `${isBid ? 'BID' : 'ASK'}_${_lifecycleColorKey(lc)}`;
      color = COLOR[colorKey] || (isBid ? COLOR.BID_RESTING : COLOR.ASK_RESTING);
      opacity = _lifecycleOpacity(lc, isBid, isSpoof);
    }

    // Authenticity reduction
    if (!isFading && auth < 0.5) {
      opacity *= 0.7;
    }

    // Apply breathing pulse for applicable states
    opacity = _applyPulse(opacity, lc, isBid, isFading);

    // Draw filled bar
    const { r, g, b } = color;
    _ctx.fillStyle = `rgba(${r},${g},${b},${opacity.toFixed(2)})`;
    _ctx.fillRect(x0, y - barHeight / 2, barWidth, barHeight);

    // Defended glow
    if (lc === 'DEFENDED' && !isFading) {
      _ctx.save();
      _ctx.shadowColor = `rgb(${r},${g},${b})`;
      _ctx.shadowBlur = 8;
      _ctx.fillStyle = `rgba(${r},${g},${b},${(opacity * 0.3).toFixed(2)})`;
      _ctx.fillRect(x0, y - barHeight / 2, barWidth, barHeight);
      _ctx.restore();
    }

    // Dashed border for low-authenticity or PULLED
    if ((auth < 0.5 || lc === 'PULLED') && !isFading) {
      _ctx.save();
      _ctx.setLineDash([3, 3]);
      _ctx.strokeStyle = `rgba(${r},${g},${b},${(opacity * 0.8).toFixed(2)})`;
      _ctx.lineWidth = 1;
      _ctx.strokeRect(x0, y - barHeight / 2, barWidth, barHeight);
      _ctx.setLineDash([]);
      _ctx.restore();
    }

    // Iceberg dotted extension
    if (isIceberg && !isFading) {
      _ctx.save();
      _ctx.setLineDash([2, 4]);
      _ctx.strokeStyle = `rgba(${r},${g},${b},0.40)`;
      _ctx.lineWidth = 1;
      _ctx.beginPath();
      _ctx.moveTo(x0 - 2, y);
      _ctx.lineTo(x0 - 20, y);
      _ctx.stroke();
      _ctx.setLineDash([]);
      _ctx.restore();
    }

    // Order count label inside bar (only if bar is wide enough)
    if (barWidth > 28 && !isFading) {
      const orderCount = lvl.order_count || '';
      _ctx.font = '10px JetBrains Mono, monospace';
      _ctx.fillStyle = `rgba(255,255,255,${Math.min(0.9, opacity + 0.2).toFixed(2)})`;
      _ctx.textBaseline = 'middle';
      _ctx.fillText(String(orderCount), x0 + 4, y);
    }

    // Spoof warning marker
    if (isSpoof && !isFading && barWidth > 16) {
      _ctx.font = '9px sans-serif';
      _ctx.fillStyle = 'rgba(230,126,34,0.9)';
      _ctx.textBaseline = 'middle';
      _ctx.textAlign = 'right';
      _ctx.fillText('\u26A0', rightEdge - 3, y);
      _ctx.textAlign = 'left'; // reset
    }
  }

  function _lifecycleColorKey(lc) {
    switch (lc) {
      case 'FORMING':  return 'FORMING';
      case 'RESTING':  return 'RESTING';
      case 'TESTED':   return 'TESTED';
      case 'DEFENDED': return 'DEFENDED';
      default:         return 'RESTING';
    }
  }

  function _lifecycleOpacity(lc, isBid, isSpoof) {
    switch (lc) {
      case 'FORMING':  return isBid ? 0.40 : 0.45;
      case 'RESTING':  return isBid ? 0.70 : 0.85;
      case 'TESTED':   return isBid ? 0.55 : 0.65;
      case 'DEFENDED': return 1.0;
      case 'BROKEN':   return 0.25;
      case 'PULLED':   return 0.20;
      default:         return 0.50;
    }
  }

  function _spoofOpacity(lc) {
    return lc === 'RESTING' ? 0.40 : 0.30;
  }

  /**
   * Apply breathing pulse to opacity for FORMING/TESTED (bid) and TESTED (ask).
   * Ask RESTING levels are solid — no pulse.
   */
  function _applyPulse(opacity, lc, isBid, isFading) {
    if (isFading) return opacity;

    // Pulse amplitude and speed vary by lifecycle
    let amplitude = 0;
    let speed = 1.0;

    if (isBid && (lc === 'RESTING' || lc === 'FORMING')) {
      amplitude = 0.10;
      speed = 0.8;  // slow breath
    } else if (lc === 'TESTED') {
      amplitude = 0.12;
      speed = 1.5;  // faster flicker
    }

    if (amplitude > 0) {
      const pulse = Math.sin(_pulsePhase * speed) * amplitude;
      return Math.max(0.15, Math.min(1.0, opacity + pulse));
    }
    return opacity;
  }

  /** Clear all levels and fading state. */
  function clear() {
    _levels = [];
    _fading.clear();
    if (_ctx && _canvas) {
      _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
    }
  }

  /** Track 3D: Enable/disable level visuals. */
  function setEnabled(on) {
    _enabled = !!on;
    if (!_enabled) clear();
  }

  function isEnabled() { return _enabled; }

  function toggle() {
    setEnabled(!_enabled);
    return _enabled;
  }

  return { init, updateFromLevels, updateFromFrame, render, clear, setEnabled, isEnabled, toggle };
})();
