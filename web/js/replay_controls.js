/**
 * replay_controls.js — TradingView-style replay control bar.
 *
 * Handles play/pause, speed selection, step forward/backward,
 * scrubber, timestamp display (ET), and timeframe switching.
 * Communicates with the server via /api/replay/* endpoints.
 */
const ReplayControls = (() => {
  let _bar = null;
  let _active = false;
  let _playing = false;
  let _speed = 1.0;
  let _totalCandles = 0;
  let _currentCandle = 0;
  let _pollInterval = null;
  let _fastLoaded = false;
  let _fastLoadCandles = [];
  let _fastLoadTimeframe = 15;
  let _tickPlayTimer = null;

  const SPEEDS = [0.5, 1, 2, 5, 10];
  const TIMEFRAMES = [
    { label: '1s', value: 1 },
    { label: '5s', value: 5 },
    { label: '15s', value: 15 },
    { label: '1m', value: 60 },
    { label: '5m', value: 300 },
  ];

  /**
   * Initialize replay controls.
   * @param {HTMLElement} bar - Container element for controls
   */
  function init(bar) {
    _bar = bar;
    if (!bar) return;
    _buildUI();
    _bindEvents();
  }

  function _buildUI() {
    if (!_bar) return;
    _bar.innerHTML = `
      <div class="replay-bar" id="replay-bar" style="display:none;">
        <div class="replay-group replay-load-group">
          <button class="replay-btn replay-load-btn" id="rp-load" title="Load file for replay">LOAD</button>
          <span class="replay-load-status" id="rp-load-status"></span>
          <div class="replay-load-bars" id="rp-load-bars" style="display:none;">
            <div class="rp-bar-row"><span class="rp-bar-label">CHART</span><div class="rp-bar-track"><div class="rp-bar-fill" id="rp-bar-chart"></div></div></div>
            <div class="rp-bar-row"><span class="rp-bar-label">TICKS</span><div class="rp-bar-track"><div class="rp-bar-fill" id="rp-bar-ticks"></div></div></div>
            <div class="rp-bar-row"><span class="rp-bar-label">LEVELS</span><div class="rp-bar-track"><div class="rp-bar-fill" id="rp-bar-levels"></div></div></div>
          </div>
        </div>
        <div class="replay-group replay-transport">
          <button class="replay-btn replay-inactive" id="rp-step-back" title="Step Back (&larr;)">&#9664;&#9664;</button>
          <button class="replay-btn replay-play replay-inactive" id="rp-play" title="Play/Pause (Space)">&#9654;</button>
          <button class="replay-btn replay-inactive" id="rp-step-fwd" title="Step Forward (&rarr;)">&#9654;&#9654;</button>
        </div>
        <div class="replay-group replay-speeds">
          ${SPEEDS.map(s => `<button class="replay-speed-btn${s === 1 ? ' active' : ''}" data-speed="${s}">${s}x</button>`).join('')}
        </div>
        <div class="replay-group replay-scrubber-group">
          <input type="range" id="rp-scrubber" class="replay-scrubber" min="0" max="100" value="0" step="1" disabled>
          <span class="replay-timestamp" id="rp-timestamp">--:--:-- ET</span>
        </div>
        <div class="replay-group replay-timeframes">
          ${TIMEFRAMES.map(tf => `<button class="replay-tf-btn${tf.value === 15 ? ' active' : ''}" data-tf="${tf.value}">${tf.label}</button>`).join('')}
        </div>
      </div>
    `;
  }

  function _bindEvents() {
    const bar = _bar;
    if (!bar) return;

    // Use event delegation for ALL buttons so disabled→enabled transitions
    // don't lose listeners and clicks always bubble to the bar container.
    bar.addEventListener('click', (e) => {
      const target = e.target.closest('button');
      if (!target) return;

      if (target.id === 'rp-load') { _startFastLoad(); return; }
      if (target.id === 'rp-play') { togglePlayPause(); return; }
      if (target.id === 'rp-step-fwd') { stepForward(); return; }
      if (target.id === 'rp-step-back') { stepBack(); return; }

      if (target.classList.contains('replay-speed-btn')) {
        const spd = parseFloat(target.dataset.speed);
        setSpeed(spd);
        return;
      }
      if (target.classList.contains('replay-tf-btn')) {
        const tf = parseInt(target.dataset.tf, 10);
        setTimeframe(tf);
        return;
      }
    });

    // Scrubber
    const scrubber = bar.querySelector('#rp-scrubber');
    if (scrubber) {
      let scrubbing = false;
      scrubber.addEventListener('mousedown', () => { scrubbing = true; });
      scrubber.addEventListener('mouseup', () => {
        scrubbing = false;
        const idx = _fastLoaded
          ? parseInt(scrubber.value, 10)
          : Math.round((scrubber.value / 100) * Math.max(1, _totalCandles - 1));
        _apiSeek(idx);
      });
      scrubber.addEventListener('input', () => {
        if (!_fastLoaded || _fastLoadCandles.length === 0) return;
        const idx = parseInt(scrubber.value, 10);
        const candle = _fastLoadCandles[Math.min(idx, _fastLoadCandles.length - 1)];
        if (candle) _updateTimestampDisplay(candle.time * 1000);
        // Pause playback during manual scrub
        if (_playing) _fastPause();
        // Slice chart to show only candles up to idx
        if (typeof ChartModule !== 'undefined' && ChartModule.sliceTo) {
          ChartModule.sliceTo(idx, _fastLoadCandles);
        }
        _currentCandle = idx;
        _updateScrubber();
      });
    }
  }

  function _on(root, selector, event, fn) {
    const el = root.querySelector(selector);
    if (el) el.addEventListener(event, fn);
  }

  /** Show/activate the replay bar. */
  function enable() {
    _active = true;
    const replayBar = _bar && _bar.querySelector('#replay-bar');
    if (replayBar) replayBar.style.display = 'flex';

    // Start polling status
    if (_pollInterval) clearInterval(_pollInterval);
    _pollInterval = setInterval(_pollStatus, 1000);
  }

  function disable() {
    _active = false;
    _fastPause();
    const replayBar = _bar && _bar.querySelector('#replay-bar');
    if (replayBar) replayBar.style.display = 'none';
    if (_pollInterval) clearInterval(_pollInterval);
  }

  function isActive() { return _active; }
  function isPlaying() { return _playing; }

  async function togglePlayPause() {
    if (_fastLoaded) {
      if (_playing) { _fastPause(); } else { _fastPlay(); }
      return;
    }
    if (_playing) {
      await _apiFetch('/api/replay/pause', 'POST', {});
      _playing = false;
    } else {
      await _apiFetch('/api/replay/play', 'POST', { speed: _speed });
      _playing = true;
    }
    _updatePlayButton();
  }

  async function stepForward() {
    if (_fastLoaded) {
      if (_currentCandle < _totalCandles - 1) {
        _scrubTo(_currentCandle + 1);
      }
      return;
    }
    await _apiFetch('/api/replay/step', 'POST', {});
  }

  async function stepBack() {
    if (_fastLoaded) {
      if (_currentCandle > 0) {
        _scrubTo(_currentCandle - 1);
      }
      return;
    }
    await _apiFetch('/api/replay/step-back', 'POST', {});
  }

  async function setSpeed(speed) {
    _speed = speed;
    // Update button highlight
    if (_bar) {
      _bar.querySelectorAll('.replay-speed-btn').forEach(b => {
        b.classList.toggle('active', parseFloat(b.dataset.speed) === speed);
      });
    }
    if (_playing) {
      // Restart playback at new speed
      await _apiFetch('/api/replay/pause', 'POST', {});
      await _apiFetch('/api/replay/play', 'POST', { speed });
    }
  }

  function speedUp() {
    const idx = SPEEDS.indexOf(_speed);
    if (idx < SPEEDS.length - 1) setSpeed(SPEEDS[idx + 1]);
  }

  function speedDown() {
    const idx = SPEEDS.indexOf(_speed);
    if (idx > 0) setSpeed(SPEEDS[idx - 1]);
  }

  async function setTimeframe(tf) {
    if (_bar) {
      _bar.querySelectorAll('.replay-tf-btn').forEach(b => {
        b.classList.toggle('active', parseInt(b.dataset.tf, 10) === tf);
      });
    }
    const data = await _apiFetch('/api/replay/timeframe', 'POST', { timeframe_s: tf });
    if (data && data.total_candles != null) {
      _totalCandles = data.total_candles;
    }

    // Reload candle array for chart
    const candles = await _apiFetch(`/api/replay/candles?timeframe_s=${tf}`, 'GET');
    if (candles && candles.candles && typeof ChartModule !== 'undefined') {
      ChartModule.loadCandles(candles.candles);
    }
  }

  /** Update the frame display from a status poll. */
  function updateFromFrame(frame) {
    if (!frame) return;
    // Replay frames have a 'type' field
    if (frame.type !== 'replay_frame') return;
    if (frame.candle && frame.candle.time) {
      _updateTimestampDisplay(frame.candle.time * 1000);
    }
  }

  async function _pollStatus() {
    // Fast-load mode drives playback entirely client-side — the server has
    // no active replay and would clobber _playing / _currentCandle if polled.
    if (_fastLoaded) return;
    try {
      const data = await _apiFetch('/api/replay/status', 'GET');
      if (!data) return;
      _totalCandles = data.total_candles || 0;
      _currentCandle = data.current_candle_idx || 0;
      _playing = data.is_playing || false;
      _updatePlayButton();
      _updateScrubber();
      if (data.current_timestamp_ms) {
        _updateTimestampDisplay(data.current_timestamp_ms);
      }
    } catch (e) { /* ignore */ }
  }

  function _updatePlayButton() {
    if (!_bar) return;
    const btn = _bar.querySelector('#rp-play');
    if (btn) btn.innerHTML = _playing ? '&#10074;&#10074;' : '&#9654;';
  }

  function _updateScrubber() {
    const scrubber = _bar && _bar.querySelector('#rp-scrubber');
    if (!scrubber) return;
    if (_fastLoaded) {
      scrubber.value = _currentCandle;
    } else {
      const pct = _totalCandles > 1 ? (_currentCandle / (_totalCandles - 1)) * 100 : 0;
      scrubber.value = pct.toFixed(1);
    }
  }

  function _updateTimestampDisplay(epochMs) {
    const el = _bar && _bar.querySelector('#rp-timestamp');
    if (!el) return;
    if (epochMs == null) { el.textContent = '--:--:-- ET'; return; }
    try {
      const d = new Date(epochMs);
      const opts = { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
      el.textContent = d.toLocaleTimeString('en-US', opts) + ' ET';
    } catch (e) {
      el.textContent = new Date(epochMs).toISOString().substr(11, 8) + ' UTC';
    }
  }

  // ── Fast Load ──────────────────────────────────────────────────

  async function _startFastLoad() {
    const fileSelect = document.getElementById('ctrl-datafile');
    const tfSelect = document.getElementById('ctrl-timeframe');

    let filePath = '';
    let symbol = 'NQ';

    if (fileSelect && fileSelect.value) {
      try {
        const parsed = JSON.parse(fileSelect.value);
        filePath = parsed.path || '';
        symbol = (parsed.filter_symbol || 'NQ').toUpperCase();
      } catch (e) {
        filePath = fileSelect.value;
      }
    }

    if (!symbol || symbol === 'NQ') {
      const symSelect = document.getElementById('ctrl-symbol');
      if (symSelect && symSelect.value) {
        symbol = symSelect.value.split('.')[0].toUpperCase();
      }
    }

    const tfMs = tfSelect ? parseInt(tfSelect.value, 10) : 15000;
    const tfS = Math.max(1, Math.floor(tfMs / 1000));
    _fastLoadTimeframe = tfS;

    if (!filePath) {
      if (typeof ControlPanel !== 'undefined' && ControlPanel.getLocks) {
        const locks = ControlPanel.getLocks();
        if (!locks.datafile || !locks.datafile.locked) {
          _setLoadStatus('Lock a data file first', true);
          return;
        }
      }
      _setLoadStatus('No file selected', true);
      return;
    }

    // Read session range from control panel
    let timeStart = null, timeEnd = null, rthOnly = false;
    if (typeof ControlPanel !== 'undefined' && ControlPanel.getSessionParams) {
      const sp = ControlPanel.getSessionParams();
      timeStart = sp.time_start;
      timeEnd = sp.time_end;
      rthOnly = sp.rth_only || false;
    }

    _setLoadStatus('Loading...');
    _setTransportEnabled(false);
    _showLoadBars(true);
    _updateLoadBars(0, 0, 0);

    const resp = await _apiFetch('/api/replay/load', 'POST', {
      file_path: filePath,
      symbol: symbol,
      snapshot_interval_ms: 100,
      time_start: timeStart,
      time_end: timeEnd,
      rth_only: rthOnly,
    });

    if (!resp || resp.status !== 'loading') {
      _setLoadStatus('Load failed', true);
      _showLoadBars(false);
      return;
    }

    _pollThreePhaseProgress(tfS);
  }

  function _pollThreePhaseProgress(tfS) {
    let chartLoaded = false;

    const pollId = setInterval(async () => {
      const status = await _apiFetch('/api/replay/status', 'GET');
      if (!status) return;

      _updateLoadBars(
        status.load_progress_chart || 0,
        status.load_progress_ticks || 0,
        status.load_progress_levels || 0,
      );

      if (status.chart_ready && !chartLoaded) {
        const candles = await _apiFetch(`/api/replay/candles?timeframe_s=${tfS}`, 'GET');
        if (candles && candles.candles) {
          chartLoaded = true;
          _fastLoadCandles = candles.candles.map(c => ({
            time: c.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume,
          }));
          _totalCandles = _fastLoadCandles.length;
          _currentCandle = 0;
          _fastLoaded = true;

          if (typeof ChartModule !== 'undefined') {
            ChartModule.loadCandles(_fastLoadCandles);
          }
          _setTransportEnabled(true);
          const scrubber = _bar && _bar.querySelector('#rp-scrubber');
          if (scrubber) {
            scrubber.max = Math.max(1, _totalCandles - 1);
            scrubber.value = 0;
          }
          if (_fastLoadCandles.length > 0) {
            _updateTimestampDisplay(_fastLoadCandles[0].time * 1000);
          }

          // Start bar countdown for replay mode
          if (typeof ChartModule !== 'undefined' && ChartModule.startBarCountdown) {
            ChartModule.startBarCountdown(_fastLoadTimeframe);
          }

          // Auto-start playback from candle 0 as soon as chart data is ready
          _currentCandle = 0;
          _scrubTo(0);
          _fastPlay();
        }
      }

      // Build status text
      const parts = [];
      if (status.chart_ready) parts.push('chart \u2713');
      if (status.ticks_ready) parts.push('ticks \u2713');
      if (status.levels_ready) parts.push('levels \u2713');

      if (status.chart_ready && status.ticks_ready && status.levels_ready) {
        clearInterval(pollId);
        _setLoadStatus(`Loaded \u2014 ${_totalCandles} candles`);
        _showLoadBars(false);
        // Notify control panel that loading is complete
        if (typeof ControlPanel !== 'undefined' && ControlPanel.onLoadComplete) {
          ControlPanel.onLoadComplete();
        }
      } else {
        _setLoadStatus(`Loading\u2026 ${parts.join(' ')}`);
      }
    }, 1000);
  }

  function _showLoadBars(visible) {
    const el = _bar && _bar.querySelector('#rp-load-bars');
    if (el) el.style.display = visible ? 'block' : 'none';
  }

  function _updateLoadBars(chart, ticks, levels) {
    const set = (id, pct) => {
      const el = _bar && _bar.querySelector(id);
      if (el) el.style.width = Math.round(pct * 100) + '%';
    };
    set('#rp-bar-chart', chart);
    set('#rp-bar-ticks', ticks);
    set('#rp-bar-levels', levels);
  }

  function _setLoadStatus(text, isError) {
    const el = _bar && _bar.querySelector('#rp-load-status');
    if (!el) return;
    el.textContent = text;
    el.style.color = isError ? '#e74c3c' : '#8f8';
  }

  function _setTransportEnabled(enabled) {
    if (!_bar) return;
    ['#rp-play', '#rp-step-fwd', '#rp-step-back'].forEach(sel => {
      const btn = _bar.querySelector(sel);
      if (btn) btn.classList.toggle('replay-inactive', !enabled);
    });
    const scrubber = _bar.querySelector('#rp-scrubber');
    if (scrubber) scrubber.disabled = !enabled;
  }

  // ── Fast-Load Playback (client-side, tick-by-tick within candles) ──

  let _tickQueue = [];      // ticks for the current forming candle
  let _tickIdx = 0;         // position within _tickQueue
  let _formingCandle = null; // candle object being grown

  async function _fastPlay() {
    if (!_fastLoaded) return;
    // If at the last candle, show "End of data" and do nothing
    if (_currentCandle >= _totalCandles - 1) {
      _setLoadStatus('End of data');
      setTimeout(() => { if (!_playing) _setLoadStatus(`Loaded — ${_totalCandles} candles`); }, 2000);
      return;
    }
    _playing = true;
    _updatePlayButton();

    // Start the tick-by-tick loop for the current candle
    _advanceToCandle(_currentCandle);
  }

  async function _advanceToCandle(candleIdx) {
    if (!_playing || candleIdx >= _totalCandles) {
      _fastPause();
      return;
    }

    _currentCandle = candleIdx;
    _formingCandle = _fastLoadCandles[candleIdx];
    _updateScrubber();

    // Fetch ticks for this candle from the pre-built index
    let ticks = [];
    try {
      const resp = await _apiFetch(
        `/api/replay/ticks?candle_time=${_formingCandle.time}&timeframe_s=${_fastLoadTimeframe}`, 'GET'
      );
      if (resp && resp.ticks) ticks = resp.ticks;
    } catch (e) { /* use empty ticks — candle will snap */ }

    if (ticks.length === 0 || !_playing) {
      // No tick data for this candle — show it fully formed and move on
      if (typeof ChartModule !== 'undefined') {
        ChartModule.addCandle(_formingCandle);
      }
      _updateTimestampDisplay(_formingCandle.time * 1000);
      // Brief pause then advance to next candle
      _tickPlayTimer = setTimeout(() => _advanceToCandle(candleIdx + 1), Math.max(30, 200 / _speed));
      return;
    }

    // Stream ticks at replay speed.
    // Distribute ticks evenly across (timeframe / speed) wall-clock time.
    // At 1x speed with 15s timeframe and 100 ticks → one tick every 150ms.
    // At 10x speed → one tick every 15ms. Cap at 16ms (60fps).
    const wallTimeMs = (_fastLoadTimeframe * 1000) / _speed;
    const intervalMs = Math.max(16, wallTimeMs / ticks.length);

    _tickQueue = ticks;
    _tickIdx = 0;

    _tickPlayTimer = setInterval(() => {
      if (!_playing) {
        clearInterval(_tickPlayTimer);
        _tickPlayTimer = null;
        return;
      }

      // Feed a batch of ticks per frame to keep up at high speeds
      const batchSize = Math.max(1, Math.ceil(16 / intervalMs));
      for (let b = 0; b < batchSize && _tickIdx < _tickQueue.length; b++, _tickIdx++) {
        const tick = _tickQueue[_tickIdx];
        if (typeof ChartModule !== 'undefined' && ChartModule.updateFormingCandle) {
          ChartModule.updateFormingCandle(tick, _formingCandle.time);
        }
      }

      if (_tickIdx >= _tickQueue.length) {
        // All ticks consumed — finalize candle and advance
        clearInterval(_tickPlayTimer);
        _tickPlayTimer = null;
        _updateTimestampDisplay(_formingCandle.time * 1000);

        // Seek replay engine for level states (fire-and-forget)
        _apiFetch('/api/replay/seek', 'POST', { candle_idx: candleIdx }).catch(() => {});

        // Advance to next candle
        setTimeout(() => _advanceToCandle(candleIdx + 1), 0);
      }
    }, Math.max(16, intervalMs));
  }

  function _fastPause() {
    _playing = false;
    _updatePlayButton();
    _updateScrubber();
    if (_tickPlayTimer) {
      clearInterval(_tickPlayTimer);
      _tickPlayTimer = null;
    }
    _tickQueue = [];
    _tickIdx = 0;
  }

  async function _scrubTo(idx) {
    if (!_fastLoaded || idx < 0 || idx >= _totalCandles) return;
    _currentCandle = idx;

    const candle = _fastLoadCandles[idx];
    if (!candle) return;

    // Update scrubber position
    _updateScrubber();
    _updateTimestampDisplay(candle.time * 1000);

    // Slice chart to show only candles up to idx
    if (typeof ChartModule !== 'undefined' && ChartModule.sliceTo) {
      ChartModule.sliceTo(idx, _fastLoadCandles);
    }

    // Fetch ticks for this candle and update tick line
    try {
      const tickResp = await _apiFetch(
        `/api/replay/fast-load/ticks?candle_time=${candle.time}&timeframe_s=${_fastLoadTimeframe}`, 'GET'
      );
      if (tickResp && tickResp.ticks && typeof ChartModule !== 'undefined') {
        ChartModule.setTickLine(tickResp.ticks, candle.time, _fastLoadTimeframe);
      }
    } catch (e) { /* ignore tick fetch errors during fast scrub */ }

    // Seek replay engine for level states (if loaded)
    _apiFetch('/api/replay/seek', 'POST', { candle_idx: idx }).catch(() => {});
  }

  async function resetLoad() {
    _fastPause();
    if (typeof ChartModule !== 'undefined' && ChartModule.stopBarCountdown) {
      ChartModule.stopBarCountdown();
    }
    await _apiFetch('/api/replay/reset', 'POST', {});
    _fastLoaded = false;
    _fastLoadCandles = [];
    _totalCandles = 0;
    _currentCandle = 0;
    _setTransportEnabled(false);
    _showLoadBars(false);
    _setLoadStatus('');
    _updateTimestampDisplay(null);
    if (typeof ChartModule !== 'undefined' && ChartModule.clear) {
      ChartModule.clear();
    }
  }

  function startFromBeginning() {
    if (!_fastLoaded || _totalCandles === 0) return;
    _scrubTo(0);
    _fastPlay();
  }

  async function _apiSeek(candleIdx) {
    if (_fastLoaded) {
      _scrubTo(candleIdx);
      return;
    }
    await _apiFetch('/api/replay/seek', 'POST', { candle_idx: candleIdx });
  }

  async function _apiFetch(url, method, body) {
    try {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (method !== 'GET' && body != null) opts.body = JSON.stringify(body);
      const resp = await fetch(url, opts);
      if (!resp.ok) return null;
      return await resp.json();
    } catch (e) {
      console.warn('[ReplayControls] fetch error:', e.message);
      return null;
    }
  }

  return {
    init,
    enable,
    disable,
    isActive,
    isPlaying,
    isLoaded: () => _fastLoaded,
    togglePlayPause,
    step: stepForward,
    stepBack,
    speedUp,
    speedDown,
    setTimeframe,
    updateFromFrame,
    resetLoad,
    startFromBeginning,
    load: _startFastLoad,
  };
})();
