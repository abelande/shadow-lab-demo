/**
 * websocket_client.js — WebSocket client with auto-reconnect.
 * Connects to ws://localhost:8420/ws, parses JSON frames,
 * dispatches to all registered handlers.
 */
const WebSocketClient = (() => {
  let ws = null;
  let url = (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws';
  let handlers = [];
  let replayHandlers = [];
  let tickHandlers = [];
  let reconnectDelay = 1000;
  const MAX_RECONNECT_DELAY = 30000;
  let intentionalClose = false;

  /**
   * Initialize and connect to the WebSocket server.
   * @param {string} wsUrl - WebSocket URL (optional, defaults to ws://localhost:8420/ws)
   */
  function init(wsUrl) {
    if (wsUrl) url = wsUrl;
    _connect();
  }

  /**
   * Register a handler function that receives each parsed frame (live mode).
   * @param {Function} fn - Callback receiving (frame: object)
   */
  function onFrame(fn) {
    if (typeof fn === 'function') handlers.push(fn);
  }

  /**
   * Register a handler for replay_frame messages specifically.
   * @param {Function} fn - Callback receiving (frame: object)
   */
  function onReplayFrame(fn) {
    if (typeof fn === 'function') replayHandlers.push(fn);
  }

  /**
   * Register a handler for price_tick messages (live mode fast-path).
   * @param {Function} fn - Callback receiving (tick: { price, bid, ask, ts })
   */
  function onPriceTick(fn) {
    if (typeof fn === 'function') tickHandlers.push(fn);
  }

  /** Establish WebSocket connection */
  function _connect() {
    intentionalClose = false;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      console.error('[WS] Failed to create WebSocket:', e);
      _scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      console.log('[WS] Connected to', url);
      reconnectDelay = 1000;
      StatsHeader.setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const frame = JSON.parse(event.data);
        // Route by message type
        if (frame && frame.type === 'replay_frame') {
          for (const fn of replayHandlers) {
            try { fn(frame); } catch (e) { console.error('[WS] Replay handler error:', e); }
          }
        } else if (frame && frame.type === 'price_tick') {
          for (const fn of tickHandlers) {
            try { fn(frame); } catch (e) { console.error('[WS] Tick handler error:', e); }
          }
        } else {
          for (const fn of handlers) {
            try { fn(frame); } catch (e) { console.error('[WS] Handler error:', e); }
          }
        }
      } catch (e) {
        console.warn('[WS] Failed to parse message:', e);
      }
    };

    ws.onclose = (event) => {
      console.log('[WS] Disconnected', event.code, event.reason);
      StatsHeader.setConnected(false);
      if (!intentionalClose) _scheduleReconnect();
    };

    ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      StatsHeader.setConnected(false);
    };
  }

  /** Schedule a reconnect with exponential backoff */
  function _scheduleReconnect() {
    console.log(`[WS] Reconnecting in ${reconnectDelay}ms...`);
    setTimeout(() => {
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
      _connect();
    }, reconnectDelay);
  }

  /**
   * Disconnect the WebSocket intentionally.
   */
  function disconnect() {
    intentionalClose = true;
    if (ws) ws.close();
  }

  /**
   * Check if currently connected.
   * @returns {boolean}
   */
  function isConnected() {
    return ws && ws.readyState === WebSocket.OPEN;
  }

  return { init, onFrame, onReplayFrame, onPriceTick, disconnect, isConnected };
})();
