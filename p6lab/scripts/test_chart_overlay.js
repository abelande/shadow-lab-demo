/**
 * Chart-level overlay E2E smoketest (headless, no browser needed).
 *
 * Loads correlation_feed.js + triple_view_panel.js under a minimal canvas
 * stub, fires synthetic triple_frame + correlation_match events through
 * the mock wsClient, and checks:
 *   - TripleViewPanel._onCorrelationMatch is wired to the correlation_match channel
 *   - matchOverlay populates as matches arrive
 *   - matches older than TTL are dropped
 *   - _renderCorrelationOverlay calls the expected canvas primitives
 *     (fillRect / strokeRect / fillText) without throwing
 *
 * Run:
 *   node scripts/test_chart_overlay.js
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..');
const TRIPLE_VIEW_SRC = path.join(ROOT, 'web/js/triple_view_panel.js');
const CORR_FEED_SRC   = path.join(ROOT, 'web/js/correlation_feed.js');

// --- 1) Parse both JS files (syntax check) -----------------------------
function parseOnly(file) {
  const src = fs.readFileSync(file, 'utf8');
  try {
    new vm.Script(src, { filename: file });
    return true;
  } catch (e) {
    console.error(`[FAIL] ${path.basename(file)} syntax: ${e.message}`);
    return false;
  }
}
if (!parseOnly(TRIPLE_VIEW_SRC) || !parseOnly(CORR_FEED_SRC)) process.exit(1);
console.log('  [OK] JS syntax parses (both files)');

// --- 2) Build a mock DOM/canvas sandbox and evaluate TripleViewPanel ----
const canvasStub = () => ({
  getContext: () => ({
    clearRect: () => {},
    fillRect: (...args) => { canvasStub.calls.fillRect++; },
    strokeRect: (...args) => { canvasStub.calls.strokeRect++; },
    fillText: (...args) => { canvasStub.calls.fillText++; },
    save: () => {}, restore: () => {},
    set fillStyle(_) {}, set strokeStyle(_) {},
    set globalAlpha(_) {}, set font(_) {}, set lineWidth(_) {},
  }),
  width: 600, height: 200,
  clientWidth: 600, clientHeight: 200,
  style: {},
  addEventListener: () => {},
});
canvasStub.calls = { fillRect: 0, strokeRect: 0, fillText: 0 };

const mockDoc = {
  createElement: (tag) => {
    const el = {
      tagName: tag, children: [], className: '', style: {},
      appendChild(c) { this.children.push(c); },
      append(...cs) { this.children.push(...cs); },
      addEventListener: () => {},
    };
    if (tag === 'canvas') return Object.assign(el, canvasStub());
    return el;
  },
};

// Minimal wsClient — records which channels got subscribed, lets us fire events
function makeWs() {
  const subs = {};
  return {
    on(channel, cb) { (subs[channel] = subs[channel] || []).push(cb); },
    off(channel, cb) { const a = subs[channel] || []; const i = a.indexOf(cb); if (i>=0) a.splice(i,1); },
    fire(channel, payload) { (subs[channel] || []).forEach(cb => cb(payload)); },
    subs,
  };
}

const sandbox = {
  document: mockDoc,
  window: { ResizeObserver: class { observe() {} } },
  ResizeObserver: class { observe() {} },
  Date: Date,
  console: console,
  module: { exports: {} },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(TRIPLE_VIEW_SRC, 'utf8'), sandbox, { filename: TRIPLE_VIEW_SRC });

const TripleViewPanel = sandbox.TripleViewPanel || sandbox.module.exports;
assert(typeof TripleViewPanel === 'function', 'TripleViewPanel class not exported to sandbox');
console.log('  [OK] TripleViewPanel class loaded into sandbox');

// --- 3) Construct the panel with a mock wsClient + verify subscriptions --
const ws = makeWs();
const panel = new TripleViewPanel({
  container: mockDoc.createElement('div'),
  wsClient: ws,
  replayControls: null,
  chart: { container: { style: {} } },
});
assert(Array.isArray(ws.subs.correlation_match) && ws.subs.correlation_match.length === 1,
       'panel did not subscribe to correlation_match');
console.log('  [OK] TripleViewPanel subscribed to correlation_match channel');

// --- 4) Fire a triple_frame so frameBuffer is non-empty, then a match ----
// `_onTripleFrame` short-circuits unless the panel is active, so set it first.
panel.active = true;
ws.fire('triple_frame', { timestamp_ms: 1000, l2_book_vector: new Array(40).fill(1) });
ws.fire('triple_frame', { timestamp_ms: 2000, l2_book_vector: new Array(40).fill(1) });
assert(panel.frameBuffer.length === 2, 'frameBuffer should have 2 entries');

const match = {
  tier: 'A',
  pattern_id: 'synthetic_bull',
  ensemble_score: 0.91,
  match_window_start_ms: 1000,
  match_window_end_ms: 2000,
};
const before = canvasStub.calls.fillRect;
ws.fire('correlation_match', match);

assert.strictEqual(panel.matchOverlay.length, 1, 'matchOverlay should have 1 entry');
assert(canvasStub.calls.fillRect > before, 'expected fillRect to be called during overlay render');
console.log(`  [OK] match reached overlay; canvas calls: fillRect=${canvasStub.calls.fillRect} ` +
            `strokeRect=${canvasStub.calls.strokeRect} fillText=${canvasStub.calls.fillText}`);

// --- 5) TTL expiry — simulate a match received 60s ago -------------------
panel.matchOverlay[0].received_at = Date.now() - 60_000;
ws.fire('correlation_match', { ...match, pattern_id: 'fresh' });
// after the new match fires, old one should be filtered out by TTL sweep
assert.strictEqual(panel.matchOverlay.length, 1,
                   `expected TTL sweep to drop aged entry, got ${panel.matchOverlay.length}`);
assert.strictEqual(panel.matchOverlay[0].pattern_id, 'fresh',
                   'only the fresh match should survive TTL sweep');
console.log('  [OK] TTL expiry drops aged matches');

// --- 6) CorrelationFeed class loads + exports without syntax errors ------
// We don't instantiate it here (the dock needs richer DOM primitives than the
// stub provides — querySelector, innerHTML parsing). Unit tests don't exist
// for the JS dock because it's purely browser-side; the instantiation check
// is meaningful enough as a smoketest.
vm.runInContext(fs.readFileSync(CORR_FEED_SRC, 'utf8'), sandbox, { filename: CORR_FEED_SRC });
const CorrelationFeed = sandbox.CorrelationFeed || sandbox.module.exports;
assert(typeof CorrelationFeed === 'function', 'CorrelationFeed class not exported');
assert(typeof CorrelationFeed.prototype._onMatch === 'function', 'dock missing _onMatch on prototype');
console.log('  [OK] CorrelationFeed class exports + _onMatch on prototype');

console.log('\nOK — chart overlay smoketest PASSED');
