/**
 * control_panel.js — Dashboard control panel with sequential lock-in.
 *
 * Each dropdown must be locked in sequentially with backend confirmation
 * before the next one unlocks. Lock order varies by mode:
 *   Replay: Mode -> Data File -> Session -> Speed -> Interval -> Timeframe
 *   Live:   Mode -> Symbol -> Speed -> Interval -> Timeframe
 *
 * START button remains disabled until all steps show green checks.
 */
const ControlPanel = (() => {
  let els = {};
  let pollTimer = null;
  let filesRetryTimer = null;
  let filesLoaded = false;

  let _dataFiles = [];
  let _feedRunning = false;

  // Replay info bar elements
  let replayBar, replayFileName, replayDateRange, replayTimestamp, replayMatchedEl;

  // ── Lock state ──────────────────────────────────────────────────
  // Lock order per mode
  const LOCK_ORDER_REPLAY = ['mode', 'datafile', 'session', 'speed', 'interval', 'timeframe'];
  const LOCK_ORDER_LIVE   = ['mode', 'symbol', 'speed', 'interval', 'timeframe'];

  // Map step -> { locked: bool, value: string|null }
  let _locks = {};
  // Cached file metadata from preflight lock
  let _lockedFileMeta = null;

  function _getLockOrder() {
    const mode = _locks.mode && _locks.mode.locked ? _locks.mode.value : 'replay';
    return mode === 'live' ? LOCK_ORDER_LIVE : LOCK_ORDER_REPLAY;
  }

  function _stepElement(step) {
    const map = {
      mode: els.mode,
      datafile: els.datafile,
      symbol: els.symbol,
      session: els.session,
      speed: els.speed,
      interval: els.interval,
      timeframe: els.timeframe,
    };
    return map[step] || null;
  }

  function _stepValue(step) {
    const el = _stepElement(step);
    if (!el) return '';
    return el.value;
  }

  function _setLockIndicator(step, state) {
    const indicator = document.querySelector(`.lock-indicator[data-step="${step}"]`);
    if (!indicator) return;
    indicator.className = 'lock-indicator ' + state;
  }

  function _resetLocksFrom(stepIndex) {
    const order = _getLockOrder();
    for (let i = stepIndex; i < order.length; i++) {
      const step = order[i];
      _locks[step] = { locked: false, value: null };
      _setLockIndicator(step, 'lock-pending');
      const el = _stepElement(step);
      if (el) el.disabled = true;
    }
    _lockedFileMeta = null;
    _updateStartButton();
  }

  function _enableStep(step) {
    const el = _stepElement(step);
    if (el) el.disabled = false;
  }

  function _buildContext() {
    const ctx = {};
    const order = _getLockOrder();
    for (const step of order) {
      if (_locks[step] && _locks[step].locked) {
        ctx[step] = _locks[step].value;
      }
    }
    return ctx;
  }

  async function _lockStep(step, value) {
    _setLockIndicator(step, 'lock-checking');

    // For datafile, send the file path (parsed from JSON value)
    let sendValue = value;
    if (step === 'datafile' && value) {
      try {
        const parsed = JSON.parse(value);
        sendValue = parsed.path || value;
      } catch (e) {
        sendValue = value;
      }
    }

    const result = await _apiCall('/api/preflight/lock', 'POST', {
      step: step,
      value: sendValue,
      context: _buildContext(),
    });

    if (!result) {
      _setLockIndicator(step, 'lock-failed');
      return false;
    }

    if (result.ok) {
      _locks[step] = { locked: true, value: value };
      _setLockIndicator(step, 'lock-confirmed');

      // Store file meta — prefer backend response, fall back to _dataFiles list
      if (step === 'datafile') {
        if (result.meta && (result.meta.start || result.meta.end)) {
          _lockedFileMeta = result.meta;
        } else {
          // Fall back to metadata from the data files list
          _lockedFileMeta = _resolveFileMeta(value);
        }
      }

      // Enable next step
      const order = _getLockOrder();
      const idx = order.indexOf(step);
      if (idx >= 0 && idx < order.length - 1) {
        _enableStep(order[idx + 1]);
      }

      _updateStartButton();
      return true;
    } else {
      _setLockIndicator(step, 'lock-failed');
      return false;
    }
  }

  function _updateStartButton() {
    const order = _getLockOrder();
    const allLocked = order.every(s => _locks[s] && _locks[s].locked);
    if (els.start) {
      els.start.disabled = !allLocked;
      if (allLocked) {
        els.start.classList.add('ready-glow');
      } else {
        els.start.classList.remove('ready-glow');
      }
    }
  }

  function _onStepChange(step) {
    const order = _getLockOrder();
    const idx = order.indexOf(step);
    if (idx < 0) return;

    // Reset this step and all downstream
    _resetLocksFrom(idx);

    // Re-enable this step's element
    _enableStep(step);

    // Lock this step
    const value = _stepValue(step);
    if (value !== '' && value !== undefined) {
      _lockStep(step, value);
    }
  }

  // ── Initialization ──────────────────────────────────────────────

  function init() {
    els = {
      mode:        document.getElementById('ctrl-mode'),
      symbol:      document.getElementById('ctrl-symbol'),
      level:       document.getElementById('ctrl-level'),
      speed:       document.getElementById('ctrl-speed'),
      speedVal:    document.getElementById('ctrl-speed-val'),
      interval:    document.getElementById('ctrl-interval'),
      timeframe:   document.getElementById('ctrl-timeframe'),
      datafile:    document.getElementById('ctrl-datafile'),
      session:     document.getElementById('ctrl-session'),
      timeStart:   document.getElementById('ctrl-time-start'),
      timeEnd:     document.getElementById('ctrl-time-end'),
      timeRangeGrp:document.getElementById('ctrl-timerange-group'),
      start:       document.getElementById('ctrl-start'),
      stop:        document.getElementById('ctrl-stop'),
      status:      document.getElementById('ctrl-status'),
      fileRangeHint: document.getElementById('ctrl-file-range-hint'),
    };

    // Replay info bar
    replayBar       = document.getElementById('replay-info-bar');
    replayFileName  = document.getElementById('replay-file-name');
    replayDateRange = document.getElementById('replay-date-range');
    replayTimestamp = document.getElementById('replay-timestamp');
    replayMatchedEl = document.getElementById('replay-progress-text');

    // Initialize locks
    const allSteps = ['mode', 'datafile', 'symbol', 'session', 'speed', 'interval', 'timeframe'];
    for (const step of allSteps) {
      _locks[step] = { locked: false, value: null };
    }

    // Wire events — step changes trigger lock flow
    if (els.mode)      els.mode.addEventListener('change', () => _onModeChange());
    if (els.datafile)   els.datafile.addEventListener('change', () => _onStepChange('datafile'));
    if (els.symbol)     els.symbol.addEventListener('change', () => _onStepChange('symbol'));
    if (els.session)    els.session.addEventListener('change', () => { _onSessionChange(); _onStepChange('session'); });
    if (els.speed)      els.speed.addEventListener('input', () => { _onSpeedChange(); _onStepChange('speed'); });
    if (els.interval)   els.interval.addEventListener('change', () => _onStepChange('interval'));
    if (els.timeframe)  els.timeframe.addEventListener('change', () => { _onTimeframeChange(); _onStepChange('timeframe'); });

    if (els.start)    els.start.addEventListener('click', _startFeed);
    if (els.stop)     els.stop.addEventListener('click', _stopFeed);
    if (els.level)    els.level.addEventListener('change', _onLevelChange);

    if (els.speed && els.speedVal) {
      els.speedVal.textContent = els.speed.value + ' fps';
    }

    _loadDataFiles();
    pollTimer = setInterval(_pollStatus, 3000);
    _pollStatus();

    // Apply initial mode visibility and auto-lock mode
    _applyModeVisibility();
    _lockStep('mode', els.mode ? els.mode.value : 'replay');
  }

  // ── Mode change — reset everything ─────────────────────────────

  function _onModeChange() {
    // Reset all locks
    const allSteps = ['mode', 'datafile', 'symbol', 'session', 'speed', 'interval', 'timeframe'];
    for (const step of allSteps) {
      _locks[step] = { locked: false, value: null };
      _setLockIndicator(step, 'lock-pending');
      const el = _stepElement(step);
      if (el) el.disabled = true;
    }

    // Apply mode-specific visibility
    _applyModeVisibility();

    // Enable and lock mode
    if (els.mode) els.mode.disabled = false;
    _lockStep('mode', els.mode ? els.mode.value : 'replay');
  }

  /** Show/hide controls based on current mode. Set level programmatically. */
  function _applyModeVisibility() {
    const mode = els.mode ? els.mode.value : 'replay';
    const levelGroup = document.getElementById('ctrl-level-group');
    const symbolGroup = els.symbol ? els.symbol.closest('.ctrl-group') : null;
    const datafileGroup = els.datafile ? els.datafile.closest('.ctrl-group') : null;
    const sessionGroup = els.session ? els.session.closest('.ctrl-group') : null;
    const timeRangeGrp = els.timeRangeGrp;

    if (mode === 'replay') {
      // Replay: always L3, hide Level + Symbol, show Data File + Session
      if (levelGroup) levelGroup.classList.add('hidden');
      if (symbolGroup) symbolGroup.classList.add('hidden');
      if (datafileGroup) datafileGroup.classList.remove('hidden');
      if (sessionGroup) sessionGroup.classList.remove('hidden');
      // Set level to L3 programmatically
      if (els.level) {
        els.level.value = 'L3';
        _onLevelChange();
      }
    } else {
      // Live: always L1, hide Level + Data File + Session + time range
      if (levelGroup) levelGroup.classList.add('hidden');
      if (datafileGroup) datafileGroup.classList.add('hidden');
      if (symbolGroup) symbolGroup.classList.remove('hidden');
      if (sessionGroup) sessionGroup.classList.add('hidden');
      if (timeRangeGrp) timeRangeGrp.classList.add('hidden');
      // Set level to L1 programmatically
      if (els.level) {
        els.level.value = 'L1';
        _onLevelChange();
      }
    }
  }

  // ── Level toggle (L1 / L3) ───────────────────────────────────────

  function _onLevelChange() {
    const level = els.level ? els.level.value : 'L1';
    _applyCapabilityMask(level === 'L3'
      ? { tape: true, price: true, dom: true, cup_flip: true, spoof: true, fragility: true, iceberg: true, regime: true }
      : { tape: true, price: true, dom: true, cup_flip: true, spoof: false, fragility: false, iceberg: false, regime: false }
    );
    if (_feedRunning) _startFeed();
  }

  function _applyCapabilityMask(caps) {
    const l3Panels = [
      { id: 'regime-badge',   cap: 'regime' },
      { id: 'force-arrow',    cap: 'spoof'  },
      { id: 'force-label',    cap: 'spoof'  },
    ];
    for (const { id, cap } of l3Panels) {
      const el = document.getElementById(id);
      if (!el) continue;
      if (caps[cap]) {
        el.classList.remove('l1-disabled');
        el.title = '';
      } else {
        el.classList.add('l1-disabled');
        el.title = 'Requires L3 MBO data';
      }
    }
    const sigConf = document.getElementById('sig-confidence');
    if (sigConf) {
      sigConf.title = !caps.spoof ? 'L1 mode: confidence uses tape only (no spoof/fragility)' : '';
    }
    const domTitle = document.querySelector('.dom-title');
    if (domTitle) {
      domTitle.textContent = caps.spoof ? 'Depth of Market' : 'Depth of Market (BBO only)';
    }
  }

  // ── Session toggle ────────────────────────────────────────────────

  function _onSessionChange() {
    const val = els.session ? els.session.value : 'full';
    if (els.timeRangeGrp) {
      if (val === 'custom') {
        els.timeRangeGrp.classList.remove('hidden');
        _autoFillCustomRange();
      } else {
        els.timeRangeGrp.classList.add('hidden');
      }
    }
  }

  /** Auto-fill FROM/TO inputs from locked data file metadata. */
  function _autoFillCustomRange() {
    if (!_lockedFileMeta) return;

    const start = _lockedFileMeta.start;
    const end = _lockedFileMeta.end;

    if (start && els.timeStart) {
      els.timeStart.value = _utcToLocalInput(start);
    }
    if (end && els.timeEnd) {
      els.timeEnd.value = _utcToLocalInput(end);
    }

    // Show hint with full file range in UTC
    if (els.fileRangeHint && start && end) {
      els.fileRangeHint.textContent = `File range: ${_fmtDateRange(start, end)}`;
    }
  }

  /** Convert UTC ISO string to datetime-local input value in UTC (not local time). */
  function _utcToLocalInput(isoStr) {
    if (!isoStr) return '';
    try {
      const d = new Date(isoStr.replace(/\+00:00$/, 'Z'));
      if (isNaN(d.getTime())) return '';
      // Use UTC getters so inputs display UTC time, consistent with the file range hint
      const pad = (n) => String(n).padStart(2, '0');
      return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    } catch (e) { return ''; }
  }

  function _getSessionParams(fileEntry) {
    const mode = els.session ? els.session.value : 'full';

    if (mode === 'rth') {
      return { time_start: null, time_end: null, rth_only: true };
    }

    if (mode === 'overnight') {
      if (fileEntry && fileEntry.start) {
        const d = new Date(fileEntry.start);
        const dateStr = d.toISOString().substring(0, 10);
        return {
          time_start: `${dateStr}T22:00:00`,
          time_end:   null,
          rth_only:   false,
        };
      }
      return { time_start: null, time_end: null, rth_only: false };
    }

    if (mode === 'custom') {
      const ts = els.timeStart ? _localInputToUTC(els.timeStart.value) : null;
      const te = els.timeEnd   ? _localInputToUTC(els.timeEnd.value)   : null;
      return { time_start: ts, time_end: te, rth_only: false };
    }

    return { time_start: null, time_end: null, rth_only: false };
  }

  function _localInputToUTC(val) {
    if (!val) return null;
    // Inputs now store UTC values directly, so just append seconds if needed
    return val.length === 16 ? val + ':00' : val;
  }

  // ── Data files ────────────────────────────────────────────────────

  async function _loadDataFiles() {
    const files = await _apiCall('/api/data/files', 'GET');
    if (!files || !Array.isArray(files)) {
      _setStatus('DATA API ERROR', 'error');
      return;
    }

    _dataFiles = files;
    filesLoaded = files.length > 0;

    if (els.datafile) {
      els.datafile.innerHTML = '<option value="">Auto (match symbol)</option>';
      for (const f of files) {
        const opt = document.createElement('option');
        opt.value = JSON.stringify({ path: f.path, filter_symbol: f.filter_symbol });
        const dateRange = f.start ? _fmtDateRange(f.start, f.end) : '?';
        const label = f.multi_instrument
          ? `${f.symbol} — ${dateRange}`
          : `${f.symbol} — ${dateRange}`;
        const sizePart = ` (${f.size_mb}MB)`;
        opt.textContent = label + sizePart;
        els.datafile.appendChild(opt);
      }
    }

    if (filesLoaded && filesRetryTimer) {
      clearInterval(filesRetryTimer);
      filesRetryTimer = null;
    }

    if (els.status && !_feedRunning) {
      _setStatus(`READY (${files.length} FILES)`, 'paused');
    }
  }

  function _fmtDateShort(iso) {
    if (!iso) return '?';
    try {
      const d = new Date(iso.replace(/\+00:00$/, 'Z'));
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
    } catch (e) { return iso; }
  }

  function _fmtDateRange(start, end) {
    try {
      const norm = (s) => s ? s.replace(/\+00:00$/, 'Z') : null;
      const s = new Date(norm(start));
      const e = new Date(norm(end));
      if (isNaN(s.getTime()) || isNaN(e.getTime())) return (start || '') + ' \u2192 ' + (end || '');
      const tz = 'America/New_York';
      const dateOpts = { month: 'short', day: 'numeric', year: 'numeric', timeZone: tz };
      const timeOpts = { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: tz };
      const tzLabel = new Intl.DateTimeFormat('en-US', { timeZone: tz, timeZoneName: 'short' })
        .formatToParts(s).find(p => p.type === 'timeZoneName')?.value || 'ET';
      const sDate = s.toLocaleDateString('en-US', dateOpts);
      const eDate = e.toLocaleDateString('en-US', dateOpts);
      const t1 = s.toLocaleTimeString('en-US', timeOpts);
      const t2 = e.toLocaleTimeString('en-US', timeOpts);
      if (sDate === eDate) {
        return `${sDate} ${t1}\u2013${t2} ${tzLabel}`;
      } else {
        return `${sDate} ${t1} \u2192 ${eDate} ${t2} ${tzLabel}`;
      }
    } catch (ex) { return (start || '') + ' \u2192 ' + (end || ''); }
  }

  /** Look up start/end metadata from the _dataFiles list for a given dropdown value. */
  function _resolveFileMeta(dropdownValue) {
    if (!dropdownValue) return null;
    try {
      const parsed = JSON.parse(dropdownValue);
      const path = parsed.path;
      const filterSym = (parsed.filter_symbol || '').toUpperCase();
      const match = _dataFiles.find(f =>
        f.path === path && (f.filter_symbol || '').toUpperCase() === filterSym
      );
      if (match && (match.start || match.end)) {
        return { start: match.start, end: match.end };
      }
    } catch (e) { /* ignore */ }
    return null;
  }

  function _resolveFile(symLabel) {
    if (els.datafile && els.datafile.value) {
      try {
        return JSON.parse(els.datafile.value);
      } catch (e) {
        const f = _dataFiles.find(d => d.path === els.datafile.value);
        if (f) return { path: f.path, filter_symbol: f.filter_symbol || symLabel };
      }
    }

    const sym = symLabel.toUpperCase();
    const match = _dataFiles.find(d =>
      (d.symbol || '').toUpperCase() === sym ||
      (d.filter_symbol || '').toUpperCase() === sym
    );
    if (match) return { path: match.path, filter_symbol: match.filter_symbol || sym };
    return null;
  }

  // ── Feed control ──────────────────────────────────────────────────

  async function _startFeed() {
    const mode     = els.mode     ? els.mode.value               : 'replay';
    const symbol   = els.symbol   ? els.symbol.value             : 'ES.c.0';
    const interval = els.interval ? parseInt(els.interval.value) : 1000;
    const speed    = els.speed    ? parseInt(els.speed.value)    : 5;
    const symLabel = symbol.replace('.c.0', '').toUpperCase();

    await _apiCall('/api/feed/stop', 'POST');
    _resetFrontend();
    await _apiCall('/api/config', 'POST', { frame_rate_limit: speed });

    if (mode === 'live') {
      const level = els.level ? els.level.value : 'L1';
      _setStatus(`${symLabel} ${level} CONNECTING\u2026`, 'paused');
      if (els.start) els.start.disabled = true;
      const result = await _apiCall('/api/feed/live', 'POST', {
        symbol,
        dataset: 'GLBX.MDP3',
        snapshot_interval_ms: interval,
        level,
      }, 15000);
      if (els.start) els.start.disabled = false;
      if (result && result.status === 'live') {
        _feedRunning = true;
        _setStatus(`${symLabel} ${level} LIVE`, 'running');
        _updateReplayBar(null, 'live');
        if (result.capabilities) _applyCapabilityMask(result.capabilities);
      } else {
        _feedRunning = false;
        const detail = result && result.detail ? result.detail : 'Connection failed';
        _setStatus(`${level} ERR: ` + detail.substring(0, 26), 'error');
      }
      return;
    }

    // Replay mode — use the three-phase loader via ReplayControls.
    // START = reset playhead to candle 0 and begin playback.
    // Requires LOAD to have completed first (via the replay bar LOAD button).
    if (typeof ReplayControls !== 'undefined' && ReplayControls.isLoaded()) {
      _feedRunning = true;
      _setStatus(symLabel + ' REPLAY', 'running');
      ReplayControls.startFromBeginning();
    } else {
      _setStatus(symLabel + ' \u2014 LOAD a file first', 'error');
    }
  }

  function _sessionLabel(params) {
    if (params.rth_only) return '\u00b7 RTH';
    if (params.time_start && params.time_end) return '\u00b7 CUSTOM';
    if (params.time_start) return '\u00b7 FROM ' + params.time_start.substring(11, 16);
    return '';
  }

  async function _stopFeed() {
    const mode = els.mode ? els.mode.value : 'replay';
    if (mode === 'replay' && typeof ReplayControls !== 'undefined') {
      await ReplayControls.resetLoad();
    }
    await _apiCall('/api/feed/stop', 'POST');
    _feedRunning = false;
    if (els.status) els.status.title = '';
    _setStatus('PAUSED', 'paused');
    if (replayBar) replayBar.classList.add('hidden');
  }

  // ── Replay info bar ───────────────────────────────────────────────

  function _updateReplayBar(fileEntry, mode, sessionParams) {
    if (!replayBar) return;
    if (mode !== 'replay' || !fileEntry) {
      replayBar.classList.add('hidden');
      return;
    }
    replayBar.classList.remove('hidden');
    if (replayFileName) {
      replayFileName.textContent = fileEntry.file || '--';
    }
    if (replayDateRange) {
      let rangeText = fileEntry.start ? _fmtDateRange(fileEntry.start, fileEntry.end) : '--';
      if (sessionParams) {
        if (sessionParams.rth_only) rangeText += ' [RTH]';
        else if (sessionParams.time_start) rangeText = `${sessionParams.time_start} \u2192 ${sessionParams.time_end || 'end'}`;
      }
      replayDateRange.textContent = rangeText;
    }
    if (replayMatchedEl) replayMatchedEl.textContent = '0 matched';
    const fill = document.getElementById('replay-progress-fill');
    if (fill) fill.style.width = '0%';
  }

  // ── Status poll ───────────────────────────────────────────────────

  async function _pollStatus() {
    const data = await _apiCall('/api/status', 'GET');
    if (!data) return;

    const sym = (data.instrument || '').replace('.c.0', '').toUpperCase() || '--';

    if (data.live_feed_error) {
      const msg = data.live_feed_error;
      let label = msg.toLowerCase().includes('not authorized') ? 'NOT AUTHORIZED'
                : msg.toLowerCase().includes('closed') || msg.toLowerCase().includes('timed out') ? 'MARKET CLOSED'
                : 'LIVE ERROR';
      _setStatus(`${sym} ${label}`, 'error');
      if (els.status) els.status.title = msg;
      _feedRunning = false;
      return;
    }

    if (els.status) els.status.title = '';

    if (data.mode === 'live') {
      _feedRunning = true;
      const lv = data.live_level || 'L1';
      _setStatus(`${sym} ${lv} LIVE`, 'running');
      if (data.capabilities) _applyCapabilityMask(data.capabilities);
    } else if (data.mode === 'replay') {
      _feedRunning = true;
      const matched = data.records_matched != null ? data.records_matched.toLocaleString() : '\u2026';
      const scanned = data.records_scanned != null ? data.records_scanned.toLocaleString() : '\u2026';
      _setStatus(`${sym} REPLAY`, 'running');
      if (replayMatchedEl) {
        replayMatchedEl.textContent = `${matched} matched / ${scanned} scanned`;
      }
    } else {
      _feedRunning = false;
      _setStatus('PAUSED', 'paused');
      if (replayBar) replayBar.classList.add('hidden');
    }
  }

  // ── Helpers ───────────────────────────────────────────────────────

  function _onSpeedChange() {
    const speed = parseInt(els.speed.value);
    if (els.speedVal) els.speedVal.textContent = speed + ' fps';
  }

  function _onTimeframeChange() {
    const ms = els.timeframe ? parseInt(els.timeframe.value) : 1000;
    if (typeof ChartModule !== 'undefined') ChartModule.setTimeframe(ms);
  }

  function _resetFrontend() {
    if (typeof ChartModule !== 'undefined') ChartModule.reset();
    if (typeof TapeFeed !== 'undefined' && TapeFeed.reset) TapeFeed.reset();
  }

  async function _apiCall(url, method, body, timeoutMs) {
    try {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      if (timeoutMs) {
        const ctrl = new AbortController();
        opts.signal = ctrl.signal;
        setTimeout(() => ctrl.abort(), timeoutMs);
      }
      const resp = await fetch(url, opts);
      return await resp.json();
    } catch (e) {
      if (e.name === 'AbortError') {
        console.error('[Control] API timeout:', url);
        _setStatus('TIMEOUT', 'error');
      } else {
        console.error('[Control] API error:', e);
        _setStatus('ERROR', 'error');
      }
      return null;
    }
  }

  function _setStatus(text, cls) {
    if (els.status) {
      els.status.textContent = text;
      els.status.className = 'ctrl-status ' + cls;
    }
  }

  function updateReplayTimestamp(timestampMs) {
    if (!replayTimestamp || !timestampMs) return;
    try {
      const d = new Date(timestampMs);
      const etStr = d.toLocaleString('en-US', {
        timeZone: 'America/New_York',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false,
        month: 'short', day: 'numeric'
      });
      replayTimestamp.textContent = etStr + ' ET';
    } catch (e) { /* ignore */ }
  }

  /** Get the locked file metadata (start/end timestamps). */
  function getLockedFileMeta() {
    return _lockedFileMeta;
  }

  /** Get current lock state. */
  function getLocks() {
    return _locks;
  }

  function getSessionParams() {
    const fileEntry = _dataFiles.find(d => {
      const sym = _locks.mode && _locks.mode.value === 'replay'
        ? (d.filter_symbol || d.symbol || '').toUpperCase()
        : '';
      return sym && _locks.datafile && _locks.datafile.locked;
    });
    return _getSessionParams(fileEntry);
  }

  function onLoadComplete() {
    if (els.start) els.start.disabled = false;
  }

  return { init, updateReplayTimestamp, getLockedFileMeta, getLocks, getSessionParams, onLoadComplete };
})();
