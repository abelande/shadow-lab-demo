/**
 * app.js — Main entry point for the Staircase Terminal dashboard.
 * Initializes all components, wires up the WebSocket dispatcher,
 * handles keyboard shortcuts and mode detection.
 */
const App = (() => {
  let isReplayMode = false;
  let _levelRenderer = null;
  let _isLiveMode = false;

  /**
   * Initialize the entire application.
   * Called on DOMContentLoaded.
   */
  function init() {
    console.log('[App] Staircase Terminal starting...');

    // --- Initialize Chart ---
    const chartContainer = document.getElementById('chart-container');
    if (chartContainer) {
      ChartModule.init(chartContainer);
    }

    // --- Initialize Level Renderer ---
    // Uses the depth-overlay canvas for drawing filled level bars.
    // Replaces the old DepthOverlay center-split depth chart.
    const depthCanvas = document.getElementById('depth-overlay');
    console.log('[App] LevelRenderer available:', typeof LevelRenderer !== 'undefined', 'canvas:', !!depthCanvas);
    if (depthCanvas && typeof LevelRenderer !== 'undefined') {
      const chart = ChartModule.getChart();
      const series = ChartModule.getCandleSeries();
      console.log('[App] Chart:', !!chart, 'Series:', !!series);
      if (chart && series) {
        LevelRenderer.init(chart, series, depthCanvas);
        _levelRenderer = LevelRenderer;
        depthCanvas.classList.add('visible');
        console.log('[App] LevelRenderer initialized and canvas visible');
      }
    }

    // --- Initialize Stats Header ---
    StatsHeader.init();

    // --- Initialize DOM Panel ---
    const domTbody = document.getElementById('dom-tbody');
    const imbalanceEl = document.getElementById('dom-imbalance');
    if (domTbody && imbalanceEl) {
      DomPanel.init(domTbody, imbalanceEl);
    }

    // --- Initialize Tape Feed ---
    const tapeList = document.getElementById('tape-list');
    if (tapeList) {
      TapeFeed.init(tapeList);
    }

    // --- Initialize Correlation Feed (L6 thesis-chain sidebar) ---
    const corrList = document.getElementById('correlation-list');
    if (corrList && typeof CorrelationFeed !== 'undefined') {
      CorrelationFeed.init(corrList);
    }

    // --- Initialize Overlays ---
    const forceArrow = document.getElementById('force-arrow');
    const forceLabel = document.getElementById('force-label');
    ForceArrows.init(forceArrow, forceLabel);

    const regimeBadge = document.getElementById('regime-badge');
    RegimeBadge.init(regimeBadge);

    const cupBadge = document.getElementById('cup-badge');
    const velocityFill = document.getElementById('velocity-fill');
    CupFlipBadge.init(cupBadge, velocityFill);

    // --- Initialize Signal Bar ---
    const sigDir = document.getElementById('sig-direction');
    const sigConf = document.getElementById('sig-confidence');
    const sigUrg = document.getElementById('sig-urgency');
    const sigSize = document.getElementById('sig-size');
    SignalBar.init(sigDir, sigConf, sigUrg, sigSize);

    // --- Initialize Replay Controls ---
    const replayBar = document.getElementById('replay-controls-bar');
    if (typeof ReplayControls !== 'undefined') {
      ReplayControls.init(replayBar);
    }

    // --- Initialize Tape Summary ---
    const tapeSumEl = document.getElementById('tape-summary-panel');
    if (typeof TapeSummary !== 'undefined') {
      TapeSummary.init(tapeSumEl);
    }

    // --- Initialize Backtest Controls ---
    const btBar = document.getElementById('backtest-bar');
    BacktestControls.init(btBar);

    // --- Initialize Control Panel ---
    ControlPanel.init();

    // --- Initialize Execution Simulator ---
    if (typeof ExecSim !== 'undefined') ExecSim.init();

    // --- Mode Detection ---
    _detectMode();

    // --- Wire up WebSocket ---
    WebSocketClient.onFrame(_onFrame);
    WebSocketClient.onReplayFrame(_onReplayFrame);
    WebSocketClient.onPriceTick(_onPriceTick);
    WebSocketClient.init();

    // Track mode changes — drives all mode-dependent UI
    const modeSelect = document.getElementById('ctrl-mode');
    if (modeSelect) {
      modeSelect.addEventListener('change', () => _setMode(modeSelect.value));
    }

    // --- Keyboard Shortcuts ---
    document.addEventListener('keydown', _onKeyDown);

    console.log('[App] Initialization complete.');
  }

  /**
   * Central frame dispatcher for live mode frames.
   * @param {object} frame - Parsed DepthIndicatorFrame
   */
  function _onFrame(frame) {
    if (!frame) return;

    const updates = [
      () => ChartModule.updateFromFrame(frame),
      () => { if (_levelRenderer) _levelRenderer.updateFromFrame(frame); },
      () => StatsHeader.updateFromFrame(frame),
      () => DomPanel.updateFromFrame(frame),
      () => TapeFeed.updateFromFrame(frame),
      () => ForceArrows.updateFromFrame(frame),
      () => RegimeBadge.updateFromFrame(frame),
      () => CupFlipBadge.updateFromFrame(frame),
      () => SignalBar.updateFromFrame(frame),
      () => BacktestControls.updateFromFrame(frame),
      () => { if (typeof CorrelationFeed !== 'undefined') CorrelationFeed.updateFromFrame(frame); },
    ];

    for (const fn of updates) {
      try { fn(); } catch (e) { console.error('[App] Frame component error:', e); }
    }

    // ET clock overlay
    const etClock = document.getElementById('chart-et-clock');
    if (etClock && frame.timestamp_ms) {
      const d = new Date(frame.timestamp_ms);
      etClock.textContent = d.toLocaleTimeString('en-US',
        { timeZone: 'America/New_York', hour12: false,
          hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' ET';
    }

    // ExecSim price update
    const price = ChartModule.getLastPrice();
    if (typeof ExecSim !== 'undefined' && price) ExecSim.updatePrice(price);
  }

  /**
   * Price tick handler — routes to ChartModule only (live mode fast-path).
   * @param {object} tick - { type: "price_tick", price, bid, ask, ts }
   */
  function _onPriceTick(tick) {
    if (!tick) return;
    try { ChartModule.updateFromTick(tick); } catch (e) { console.error('[App] Tick error:', e); }
    if (typeof ExecSim !== 'undefined' && tick.price) ExecSim.updatePrice(tick.price);
  }

  /**
   * Replay frame dispatcher — routes replay_frame messages to new components.
   * @param {object} frame - Parsed ReplayFrame (type: "replay_frame")
   */
  function _onReplayFrame(frame) {
    if (!frame) return;

    // Update chart with new candle
    if (frame.candle && typeof ChartModule !== 'undefined') {
      try { ChartModule.addCandle(frame.candle); } catch (e) { /* ignore */ }
    }

    // Update level renderer
    if (frame.levels && _levelRenderer) {
      try { _levelRenderer.updateFromLevels(frame.levels); } catch (e) { /* ignore */ }
    }

    // Update tape summary
    if (typeof TapeSummary !== 'undefined') {
      try { TapeSummary.updateFromFrame(frame); } catch (e) { /* ignore */ }
    }

    // Update replay controls
    if (typeof ReplayControls !== 'undefined') {
      try { ReplayControls.updateFromFrame(frame); } catch (e) { /* ignore */ }
    }
  }

  /**
   * Set the application mode. Called from init (URL detection),
   * control panel mode dropdown, or server status poll.
   * @param {'live'|'replay'} mode
   */
  function _setMode(mode) {
    const tapeSumEl = document.getElementById('tape-summary-panel');
    const tapePanelEl = document.getElementById('tape-panel');

    const startBtn = document.getElementById('ctrl-start');
    const stopBtn = document.getElementById('ctrl-stop');

    if (mode === 'replay' || mode === 'backtest') {
      isReplayMode = true;
      _isLiveMode = false;
      BacktestControls.enable();
      if (typeof ReplayControls !== 'undefined') ReplayControls.enable();
      if (tapeSumEl) tapeSumEl.style.display = 'flex';
      if (tapePanelEl) tapePanelEl.style.display = 'none';

      // Feature 2-A: Hide START button in replay mode
      if (startBtn) startBtn.style.display = 'none';

      // Feature 2-B: Rename STOP → RESET, wire to ReplayControls.resetLoad()
      if (stopBtn) {
        stopBtn.textContent = '↺ RESET';
        stopBtn.title = 'Reset replay — clears loaded file and returns to initial state';
        stopBtn.onclick = () => {
          if (typeof ReplayControls !== 'undefined') ReplayControls.resetLoad();
        };
      }

      // Stop bar countdown when switching modes
      if (typeof ChartModule !== 'undefined' && ChartModule.stopBarCountdown) {
        ChartModule.stopBarCountdown();
      }
    } else {
      isReplayMode = false;
      _isLiveMode = true;
      if (typeof ReplayControls !== 'undefined') ReplayControls.disable();
      if (tapeSumEl) tapeSumEl.style.display = 'none';
      if (tapePanelEl) tapePanelEl.style.display = '';

      // Feature 2-A: Show START button in live mode
      if (startBtn) startBtn.style.display = '';

      // Feature 2-B: Restore STOP button original label and behavior
      if (stopBtn) {
        stopBtn.textContent = '■ STOP';
        stopBtn.title = 'Stop Feed';
        stopBtn.onclick = null; // restore original ControlPanel handler
      }

      // Start bar countdown in live mode
      if (typeof ChartModule !== 'undefined' && ChartModule.startBarCountdown) {
        const tfSelect = document.getElementById('ctrl-timeframe');
        const tfMs = tfSelect ? parseInt(tfSelect.value, 10) : 15000;
        ChartModule.startBarCountdown(Math.max(1, Math.floor(tfMs / 1000)));
      }
    }
  }

  /**
   * Detect initial mode from URL params or control panel default.
   */
  function _detectMode() {
    const params = new URLSearchParams(window.location.search);
    const urlMode = params.get('mode');
    if (urlMode) {
      _setMode(urlMode);
      return;
    }
    // Fall back to control panel selection
    const modeSelect = document.getElementById('ctrl-mode');
    if (modeSelect) {
      _setMode(modeSelect.value);
    }
  }

  /**
   * Handle keyboard shortcuts.
   * Space = play/pause, Right = step, +/- = speed
   * @param {KeyboardEvent} e
   */
  function _onKeyDown(e) {
    // Don't capture if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Route to ReplayControls if in replay mode, else BacktestControls
    const ctrl = (typeof ReplayControls !== 'undefined' && ReplayControls.isActive())
      ? ReplayControls
      : (BacktestControls.isActive() ? BacktestControls : null);

    switch (e.code) {
      case 'Space':
        e.preventDefault();
        if (ctrl) ctrl.togglePlayPause();
        break;
      case 'ArrowRight':
        e.preventDefault();
        if (ctrl) ctrl.step();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        if (ctrl && typeof ctrl.stepBack === 'function') ctrl.stepBack();
        break;
      case 'Equal': // + key
      case 'NumpadAdd':
        e.preventDefault();
        if (ctrl) ctrl.speedUp();
        break;
      case 'Minus':
      case 'NumpadSubtract':
        e.preventDefault();
        if (ctrl) ctrl.speedDown();
        break;
      case 'KeyL':
        // Track 3D: Toggle level visuals
        e.preventDefault();
        if (typeof LevelRenderer !== 'undefined') {
          const on = LevelRenderer.toggle();
          console.log('[App] Level visuals:', on ? 'ON' : 'OFF');
        }
        break;
    }
  }

  return { init };
})();

// --- Bootstrap on DOM ready ---
document.addEventListener('DOMContentLoaded', App.init);
