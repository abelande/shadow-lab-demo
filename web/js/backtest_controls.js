/**
 * backtest_controls.js — Backtest/replay control bar.
 * Provides play/pause/speed/step/seek controls.
 * Only visible when in replay/backtest mode.
 */
const BacktestControls = (() => {
  let barEl = null;
  let isPlaying = false;
  let currentSpeed = 1;
  let isBacktestMode = false;

  // Element references
  let playBtn, pauseBtn, speed2Btn, speed5Btn, rewindBtn, stepBtn;
  let scrubber, timestampEl;

  /**
   * Initialize backtest controls.
   * @param {HTMLElement} bar - The backtest bar container element
   */
  function init(bar) {
    barEl = bar;
    if (!barEl) return;

    playBtn = document.getElementById('bt-play');
    pauseBtn = document.getElementById('bt-pause');
    speed2Btn = document.getElementById('bt-speed2');
    speed5Btn = document.getElementById('bt-speed5');
    rewindBtn = document.getElementById('bt-rewind');
    stepBtn = document.getElementById('bt-step');
    scrubber = document.getElementById('bt-scrubber');
    timestampEl = document.getElementById('bt-timestamp');

    // Bind events
    if (playBtn) playBtn.addEventListener('click', play);
    if (pauseBtn) pauseBtn.addEventListener('click', pause);
    if (speed2Btn) speed2Btn.addEventListener('click', () => setSpeed(2));
    if (speed5Btn) speed5Btn.addEventListener('click', () => setSpeed(5));
    if (rewindBtn) rewindBtn.addEventListener('click', rewind);
    if (stepBtn) stepBtn.addEventListener('click', step);
    if (scrubber) scrubber.addEventListener('input', _onScrub);
  }

  /**
   * Enable backtest mode (show the controls bar).
   */
  function enable() {
    isBacktestMode = true;
    if (barEl) barEl.classList.add('visible');
  }

  /**
   * Disable backtest mode (hide the controls bar).
   */
  function disable() {
    isBacktestMode = false;
    if (barEl) barEl.classList.remove('visible');
  }

  /** Check if backtest mode is active */
  function isActive() { return isBacktestMode; }

  /** Play/resume playback */
  async function play() {
    isPlaying = true;
    _updateButtons();
    await _post('/api/backtest/resume');
  }

  /** Pause playback */
  async function pause() {
    isPlaying = false;
    _updateButtons();
    await _post('/api/backtest/pause');
  }

  /** Toggle play/pause */
  function togglePlayPause() {
    if (isPlaying) pause();
    else play();
  }

  /**
   * Set playback speed.
   * @param {number} speed - Multiplier (1, 2, 5)
   */
  async function setSpeed(speed) {
    currentSpeed = speed;
    _updateButtons();
    await _post('/api/backtest/resume', { speed });
  }

  /** Increase speed */
  function speedUp() {
    if (currentSpeed < 5) setSpeed(currentSpeed === 1 ? 2 : 5);
  }

  /** Decrease speed */
  function speedDown() {
    if (currentSpeed > 1) setSpeed(currentSpeed === 5 ? 2 : 1);
  }

  /** Rewind to start */
  async function rewind() {
    await _post('/api/backtest/seek', { position: 0 });
    if (scrubber) scrubber.value = 0;
  }

  /** Step forward one frame */
  async function step() {
    isPlaying = false;
    _updateButtons();
    await _post('/api/backtest/step');
  }

  /** Handle scrubber input */
  async function _onScrub() {
    if (!scrubber) return;
    const pos = parseFloat(scrubber.value);
    await _post('/api/backtest/seek', { position: pos });
  }

  /**
   * Update the timestamp display and scrubber from frame data.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!isBacktestMode || !frame) return;
    if (timestampEl && frame.timestamp_ms) {
      const d = new Date(frame.timestamp_ms);
      timestampEl.textContent = d.toISOString().replace('T', ' ').substring(0, 23);
    }
  }

  /** Update button active states */
  function _updateButtons() {
    if (playBtn) playBtn.classList.toggle('active', isPlaying);
    if (pauseBtn) pauseBtn.classList.toggle('active', !isPlaying);
    if (speed2Btn) speed2Btn.classList.toggle('active', currentSpeed === 2);
    if (speed5Btn) speed5Btn.classList.toggle('active', currentSpeed === 5);
  }

  /**
   * POST to a backtest API endpoint.
   * @param {string} path - API path
   * @param {object} body - Request body (optional)
   */
  async function _post(path, body) {
    try {
      await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      console.error('[Backtest] API call failed:', path, e);
    }
  }

  return { init, enable, disable, isActive, play, pause, togglePlayPause, setSpeed, speedUp, speedDown, rewind, step, updateFromFrame };
})();
