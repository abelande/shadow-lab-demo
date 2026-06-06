/**
 * pattern_review_queue.js — §10.3 Pattern Review Queue
 *
 * Candidate workflow:
 * - load /api/patterns/candidates
 * - show one candidate at a time
 * - Accept / Reject / Next exemplar
 * - progress bar reviewed / total
 */

class PatternReviewQueue {
  constructor({ container, tripleView, replayControls }) {
    this.container = container;
    this.tripleView = tripleView;
    this.replayControls = replayControls;

    this.candidates = [];
    this.currentIdx = 0;
    this.exemplarIdx = 0;
    this.reviewed = 0;

    this._buildUI();
  }

  _buildUI() {
    this.root = document.createElement('div');
    this.root.className = 'pattern-review-root';

    this.header = document.createElement('div');
    this.header.innerHTML = '<b>Pattern Review Queue</b>';

    this.progress = document.createElement('div');
    this.progress.style.cssText = 'height:8px;background:#333;margin:6px 0;position:relative;';
    this.progressFill = document.createElement('div');
    this.progressFill.style.cssText = 'height:100%;width:0%;background:#4caf50;';
    this.progress.appendChild(this.progressFill);

    this.meta = document.createElement('pre');
    this.meta.style.cssText = 'font-size:12px;color:#ccc;max-height:220px;overflow:auto;';

    this.btnAccept = document.createElement('button');
    this.btnAccept.textContent = 'Accept';
    this.btnAccept.onclick = () => this.acceptCurrent();

    this.btnReject = document.createElement('button');
    this.btnReject.textContent = 'Reject';
    this.btnReject.onclick = () => this.rejectCurrent();

    this.btnNext = document.createElement('button');
    this.btnNext.textContent = 'Next Exemplar';
    this.btnNext.onclick = () => this.nextExemplar();

    this.root.append(this.header, this.progress, this.meta, this.btnAccept, this.btnReject, this.btnNext);
    this.container.appendChild(this.root);
  }

  async loadCandidates() {
    const resp = await fetch('/api/patterns/candidates');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    this.candidates = await resp.json();
    this.currentIdx = 0;
    this.exemplarIdx = 0;
    this.reviewed = 0;
    this._renderCurrent();
    this._updateProgress();
  }

  _current() {
    return this.candidates[this.currentIdx] || null;
  }

  async _renderCurrent() {
    const c = this._current();
    if (!c) {
      this.meta.textContent = 'No candidates pending.';
      return;
    }

    this.meta.textContent = JSON.stringify(c, null, 2);

    // auto-scroll replay to first/current exemplar
    const exemplars = c.exemplar_timestamps || [];
    const ts = exemplars[this.exemplarIdx] ?? exemplars[0];
    if (ts && this.replayControls?.seekTo) this.replayControls.seekTo(ts);

    // ensure triple view open for contextual review
    if (this.tripleView && !this.tripleView.active) this.tripleView.toggle();
  }

  _updateProgress() {
    const total = this.candidates.length || 1;
    const pct = Math.round((this.reviewed / total) * 100);
    this.progressFill.style.width = `${pct}%`;
    this.header.innerHTML = `<b>Pattern Review Queue</b> — reviewed ${this.reviewed} / ${this.candidates.length}`;
  }

  async acceptCurrent() {
    const c = this._current();
    if (!c) return;
    const resp = await fetch(`/api/patterns/candidate/${encodeURIComponent(c.id)}/accept`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer: 'human', decision_reason: 'approved in queue' })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    this._advanceAfterDecision();
  }

  async rejectCurrent() {
    const c = this._current();
    if (!c) return;
    const resp = await fetch(`/api/patterns/candidate/${encodeURIComponent(c.id)}/reject`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reviewer: 'human', decision_reason: 'rejected in queue' })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    this._advanceAfterDecision();
  }

  _advanceAfterDecision() {
    this.reviewed += 1;
    this.currentIdx += 1;
    this.exemplarIdx = 0;
    this._updateProgress();
    this._renderCurrent();
  }

  nextExemplar() {
    const c = this._current();
    if (!c) return;
    const exemplars = c.exemplar_timestamps || [];
    if (!exemplars.length) return;
    this.exemplarIdx = (this.exemplarIdx + 1) % exemplars.length;
    const ts = exemplars[this.exemplarIdx];
    if (this.replayControls?.seekTo) this.replayControls.seekTo(ts);
  }
}

if (typeof module !== 'undefined') module.exports = PatternReviewQueue;
