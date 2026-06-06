/**
 * depth_overlay.js — Depth bar overlay aligned to chart price scale.
 * Uses LightweightCharts priceToCoordinate() for accurate Y positioning.
 * Bid bars extend LEFT from center, ask bars extend RIGHT from center.
 */
const DepthOverlay = (() => {
  let canvas = null;
  let ctx = null;
  let currentBidBars = [];
  let currentAskBars = [];

  /**
   * Initialize the overlay canvas.
   * @param {HTMLCanvasElement} canvasEl
   */
  function init(canvasEl) {
    canvas = canvasEl;
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    // Canvas is now a sibling of #chart-container inside #chart-panel.
    // Size to the chart container, not parentElement (which is chart-panel).
    _resize();
    window.addEventListener('resize', _resize);
    const chartContainer = document.getElementById('chart-container');
    if (chartContainer) {
      new ResizeObserver(_resize).observe(chartContainer);
    }
  }

  function _resize() {
    if (!canvas) return;
    const chartContainer = document.getElementById('chart-container');
    if (!chartContainer) return;
    canvas.width = chartContainer.clientWidth;
    canvas.height = chartContainer.clientHeight;
    render(); // re-render after resize so bars don't drift
  }

  /**
   * Update bar data from new frame.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!frame) return;
    currentBidBars = frame.bid_bars || [];
    currentAskBars = frame.ask_bars || [];
    render();
  }

  /**
   * Render depth bars aligned to the chart's price scale.
   * Requires ChartModule.getCandleSeries() to be available.
   */
  function render() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const series = ChartModule.getCandleSeries();
    if (!series) return;

    const allBars = [...currentBidBars, ...currentAskBars];
    if (allBars.length === 0) return;

    // Right price scale width — dynamic to avoid drift on different screen widths
    const priceScaleWidth = Math.min(80, Math.max(50, Math.round(canvas.width * 0.068)));
    const drawAreaWidth = canvas.width - priceScaleWidth;

    // Center divider: splits bid (left) and ask (right) halves
    const centerX = Math.floor(drawAreaWidth * 0.5);
    const maxHalfWidth = centerX - 24;  // 24px padding from edges

    // Max volume for normalized bar_length fallback
    const maxVol = Math.max(...allBars.map(b => (b && b.volume) ? b.volume : 0), 1);

    // Draw asks (right, red) behind bids so blue overlaps at center
    _drawBarsAligned(currentAskBars, '#e74c3c', centerX, maxHalfWidth, maxVol, 'right', series, priceScaleWidth);
    _drawBarsAligned(currentBidBars, '#3498db', centerX, maxHalfWidth, maxVol, 'left', series, priceScaleWidth);

    // Subtle center divider line
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 6]);
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /**
   * Draw bars aligned to the chart price scale.
   * @param {Array} bars
   * @param {string} baseColor - Hex color string
   * @param {number} centerX - Horizontal divider position
   * @param {number} maxHalfWidth - Max bar length in pixels
   * @param {number} maxVol - Max volume for normalization
   * @param {'left'|'right'} side
   * @param {object} series - LightweightCharts ISeriesApi
   */
  function _drawBarsAligned(bars, baseColor, centerX, maxHalfWidth, maxVol, side, series, priceScaleWidth) {
    const rgb = _hexToRgb(baseColor);
    const barHeight = 6;

    for (const bar of bars) {
      if (!bar || bar.price == null) continue;

      // Map price → canvas Y via chart's own price scale
      let y;
      try {
        y = series.priceToCoordinate(bar.price);
      } catch (e) {
        continue;
      }
      if (y == null || !Number.isFinite(y) || y < 0 || y > canvas.height) continue;

      // Bar length: prefer bar_length field, else normalize volume
      const lengthRatio = bar.bar_length != null
        ? Math.min(1, Math.max(0, bar.bar_length))
        : Math.min(1, bar.volume / maxVol);
      const barLen = Math.max(2, lengthRatio * maxHalfWidth);

      // Opacity from authenticity score
      const alpha = bar.authenticity != null
        ? Math.max(0.2, Math.min(1.0, bar.authenticity))
        : 0.7;

      const isRound = bar.is_round_number === true || (Number(bar.price) % 25 === 0);
      const { r, g, b } = rgb;

      if (side === 'left') {
        const x0 = centerX - barLen;

        // Subtle connector line from left edge to bar start
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(x0, y);
        ctx.stroke();

        ctx.fillStyle = `rgba(${r},${g},${b},${(alpha * 0.55).toFixed(2)})`;
        ctx.fillRect(x0, y - barHeight / 2, barLen, barHeight);

        if (isRound) {
          ctx.save();
          ctx.shadowColor = baseColor;
          ctx.shadowBlur = 10;
          ctx.strokeStyle = `rgba(${r},${g},${b},1.0)`;
          ctx.lineWidth = 1;
          ctx.strokeRect(x0 - 1, y - barHeight / 2 - 1, barLen + 2, barHeight + 2);
          ctx.restore();
        }
      } else {
        // Subtle connector line from bar end to right edge
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(centerX + barLen, y);
        ctx.lineTo(canvas.width - priceScaleWidth, y);
        ctx.stroke();

        ctx.fillStyle = `rgba(${r},${g},${b},${(alpha * 0.55).toFixed(2)})`;
        ctx.fillRect(centerX, y - barHeight / 2, barLen, barHeight);

        if (isRound) {
          ctx.save();
          ctx.shadowColor = baseColor;
          ctx.shadowBlur = 10;
          ctx.strokeStyle = `rgba(${r},${g},${b},1.0)`;
          ctx.lineWidth = 1;
          ctx.strokeRect(centerX - 1, y - barHeight / 2 - 1, barLen + 2, barHeight + 2);
          ctx.restore();
        }
      }
    }
  }

  function _fmtVol(v) {
    if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
    return String(v);
  }

  function _hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) }
      : { r: 255, g: 255, b: 255 };
  }

  return { init, updateFromFrame, render };
})();
