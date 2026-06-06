/**
 * dom_panel.js — Depth of Market (DOM) ladder panel.
 * Uses staircase bid/ask levels (10 each) when available.
 * Columns: VOL | #ORD | CUM | PRICE | CUM | #ORD | VOL
 * Fragility color-coding: SOLID=bright, MODERATE=dim, FRAGILE=pulsing red.
 */
const DomPanel = (() => {
  let tableBody = null;
  let imbalanceEl = null;

  // Previous volume snapshot for flash detection
  let prevVolMap = {};  // key: `${side}:${price}` -> volume

  /**
   * Initialize the DOM panel.
   * @param {HTMLElement} tbodyEl
   * @param {HTMLElement} imbEl
   */
  function init(tbodyEl, imbEl) {
    tableBody = tbodyEl;
    imbalanceEl = imbEl;
  }

  /**
   * Update from a new frame. Prefers staircase levels over dom_rows.
   * @param {object} frame - DepthIndicatorFrame
   */
  function updateFromFrame(frame) {
    if (!frame) return;

    const sc = frame.staircase;
    if (sc && sc.bid_levels && sc.bid_levels.length > 0 && sc.ask_levels && sc.ask_levels.length > 0) {
      _renderStaircase(sc);
    } else {
      _renderDomRows(frame.dom_rows || []);
    }
  }

  /**
   * Render full DOM ladder from staircase levels.
   * @param {object} sc - staircase object from frame
   */
  function _renderStaircase(sc) {
    if (!tableBody) return;

    const bids = sc.bid_levels || [];
    const asks = sc.ask_levels || [];

    // Build price maps
    const bidMap = {};
    for (const b of bids) if (b && b.price != null) bidMap[b.price] = b;
    const askMap = {};
    for (const a of asks) if (a && a.price != null) askMap[a.price] = a;

    // Compute cumulative volumes (inside-out order)
    const sortedBids = [...bids].filter(b => b && b.price != null).sort((a, b) => b.price - a.price);
    const sortedAsks = [...asks].filter(a => a && a.price != null).sort((a, b) => a.price - b.price);

    let bidCum = 0;
    for (const b of sortedBids) { bidCum += (b.volume || 0); b._cum = bidCum; }
    let askCum = 0;
    for (const a of sortedAsks) { askCum += (a.volume || 0); a._cum = askCum; }

    // All prices sorted descending: asks on top, bids below
    const allPrices = [...new Set([...sortedBids.map(b => b.price), ...sortedAsks.map(a => a.price)])]
      .sort((a, b) => b - a);

    const bestBid = sortedBids.length > 0 ? sortedBids[0].price : null;
    const bestAsk = sortedAsks.length > 0 ? sortedAsks[0].price : null;
    const midPrice = (bestBid !== null && bestAsk !== null) ? (bestBid + bestAsk) / 2 : null;

    // Max volume for proportional bar widths
    const maxBidVol = Math.max(...sortedBids.map(b => b.volume || 0), 1);
    const maxAskVol = Math.max(...sortedAsks.map(a => a.volume || 0), 1);
    const maxVol = Math.max(maxBidVol, maxAskVol);

    // Build new volume map for flash detection
    const newVolMap = {};
    for (const b of sortedBids) newVolMap[`BID:${b.price}`] = b.volume || 0;
    for (const a of sortedAsks) newVolMap[`ASK:${a.price}`] = a.volume || 0;

    let html = '';
    for (const price of allPrices) {
      const bid = bidMap[price];
      const ask = askMap[price];
      const isMid = midPrice !== null && Math.abs(price - midPrice) < 0.01;
      const isSpread = bestBid !== null && price === bestBid;
      const priceClass = isMid ? 'price-col current-price' : 'price-col';
      const rowClass = isSpread ? ' class="spread-row"' : '';

      html += `<tr${rowClass} data-price="${price.toFixed(2)}">`;

      if (bid) {
        const fragCls = _fragClass(bid.fragility);
        const barPct = Math.round((bid.volume || 0) / maxVol * 100);
        const flashCls = _flashClass('BID', price, bid.volume || 0);
        const aggrPct = bid.aggressive_ratio != null ? (bid.aggressive_ratio * 100).toFixed(0) : '--';
        const avgOrd = bid.avg_order_size != null ? _fmt(bid.avg_order_size) : '--';

        html += `<td class="bid-side vol-cell ${fragCls} ${flashCls}" ` +
          `title="Avg order: ${avgOrd} | Aggr: ${aggrPct}% | Fragility: ${bid.fragility || '--'}" ` +
          `style="--bar-pct:${barPct}%">${_fmt(bid.volume)}</td>`;
        html += `<td class="bid-side ${fragCls}">${_fmt(bid.order_count)}</td>`;
        html += `<td class="bid-side dim-col">${_fmt(bid._cum)}</td>`;
      } else {
        html += '<td class="empty-cell"></td><td class="empty-cell"></td><td class="empty-cell"></td>';
      }

      html += `<td class="${priceClass}">${_fmtPrice(price)}</td>`;

      if (ask) {
        const fragCls = _fragClass(ask.fragility);
        const barPct = Math.round((ask.volume || 0) / maxVol * 100);
        const flashCls = _flashClass('ASK', price, ask.volume || 0);
        const aggrPct = ask.aggressive_ratio != null ? (ask.aggressive_ratio * 100).toFixed(0) : '--';
        const avgOrd = ask.avg_order_size != null ? _fmt(ask.avg_order_size) : '--';

        html += `<td class="ask-side dim-col">${_fmt(ask._cum)}</td>`;
        html += `<td class="ask-side ${fragCls}">${_fmt(ask.order_count)}</td>`;
        html += `<td class="ask-side vol-cell ask-vol-cell ${fragCls} ${flashCls}" ` +
          `title="Avg order: ${avgOrd} | Aggr: ${aggrPct}% | Fragility: ${ask.fragility || '--'}" ` +
          `style="--bar-pct:${barPct}%">${_fmt(ask.volume)}</td>`;
      } else {
        html += '<td class="empty-cell"></td><td class="empty-cell"></td><td class="empty-cell"></td>';
      }

      html += '</tr>';
    }

    tableBody.innerHTML = html;
    prevVolMap = newVolMap;

    // Scroll the spread into view so the bid/ask is always centered
    if (bestBid !== null) {
      const spreadRow = tableBody.querySelector(`[data-price="${bestBid.toFixed(2)}"]`);
      if (spreadRow) spreadRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    _updateImbalanceStaircase(sc, midPrice);
  }

  /**
   * Fallback: render from dom_rows when staircase is unavailable.
   * @param {Array} rows
   */
  function _renderDomRows(rows) {
    if (!tableBody) return;

    const bids = {}, asks = {};
    for (const row of rows) {
      if (!row || row.price == null) continue;
      if (row.side === 'BID') bids[row.price] = row;
      else if (row.side === 'ASK') asks[row.price] = row;
    }

    const allPrices = [...new Set([...Object.keys(bids), ...Object.keys(asks)])]
      .map(Number).sort((a, b) => b - a);

    const bestBid = Object.keys(bids).length > 0 ? Math.max(...Object.keys(bids).map(Number)) : null;
    const bestAsk = Object.keys(asks).length > 0 ? Math.min(...Object.keys(asks).map(Number)) : null;
    const midPrice = (bestBid !== null && bestAsk !== null) ? (bestBid + bestAsk) / 2 : null;

    // Compute cumulative volumes inline (inside-out from spread) for fallback rows
    const sortedBidPrices = Object.keys(bids).map(Number).sort((a, b) => b - a);
    const sortedAskPrices = Object.keys(asks).map(Number).sort((a, b) => a - b);
    let bidCumFallback = 0;
    for (const p of sortedBidPrices) {
      bidCumFallback += (bids[p].volume || 0);
      bids[p]._cum = bidCumFallback;
    }
    let askCumFallback = 0;
    for (const p of sortedAskPrices) {
      askCumFallback += (asks[p].volume || 0);
      asks[p]._cum = askCumFallback;
    }

    let html = '';
    for (const price of allPrices) {
      const bid = bids[price], ask = asks[price];
      const isMid = midPrice !== null && Math.abs(price - midPrice) < 0.01;
      const isSpread = bestBid !== null && price === bestBid;
      const rowClass = isSpread ? ' class="spread-row"' : '';
      html += `<tr${rowClass} data-price="${price.toFixed(2)}">`;
      if (bid) {
        html += `<td class="bid-side">${_fmt(bid.volume)}</td>`;
        html += `<td class="bid-side">${_fmt(bid.order_count)}</td>`;
        html += `<td class="bid-side dim-col">${_fmt(bid._cum != null ? bid._cum : bid.cumulative_volume)}</td>`;
      } else {
        html += '<td class="empty-cell"></td><td class="empty-cell"></td><td class="empty-cell"></td>';
      }
      html += `<td class="${isMid ? 'price-col current-price' : 'price-col'}">${_fmtPrice(price)}</td>`;
      if (ask) {
        html += `<td class="ask-side dim-col">${_fmt(ask._cum != null ? ask._cum : ask.cumulative_volume)}</td>`;
        html += `<td class="ask-side">${_fmt(ask.order_count)}</td>`;
        html += `<td class="ask-side">${_fmt(ask.volume)}</td>`;
      } else {
        html += '<td class="empty-cell"></td><td class="empty-cell"></td><td class="empty-cell"></td>';
      }
      html += '</tr>';
    }
    tableBody.innerHTML = html;

    // Scroll spread into view
    if (bestBid !== null) {
      const spreadRow = tableBody.querySelector(`[data-price="${bestBid.toFixed(2)}"]`);
      if (spreadRow) spreadRow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }

    _updateImbalanceDom(bids, asks, midPrice);
  }

  /**
   * Update imbalance bar from staircase totals.
   */
  function _updateImbalanceStaircase(sc, midPrice) {
    if (!imbalanceEl) return;
    const bidVol = sc.bid_total_volume || 0;
    const askVol = sc.ask_total_volume || 0;
    const total = bidVol + askVol || 1;
    const bidPct = Math.round((bidVol / total) * 100);
    const askPct = 100 - bidPct;
    const mid = midPrice != null ? _fmtPrice(midPrice) : '--';

    // Color the imbalance bar based on which side dominates
    const dominated = bidPct > 60 ? 'dom-bid' : (askPct > 60 ? 'dom-ask' : '');
    imbalanceEl.className = `dom-imbalance ${dominated}`;
    imbalanceEl.innerHTML =
      `<span class="imbalance-bid">BID ${bidPct}%</span>` +
      `<span class="imbalance-bar"><span class="imbalance-fill" style="width:${bidPct}%"></span></span>` +
      `<span class="imbalance-mid">${mid}</span>` +
      `<span class="imbalance-ask">ASK ${askPct}%</span>`;
  }

  function _updateImbalanceDom(bids, asks, midPrice) {
    if (!imbalanceEl) return;
    let totalBid = 0, totalAsk = 0;
    for (const k in bids) totalBid += (bids[k].volume || 0);
    for (const k in asks) totalAsk += (asks[k].volume || 0);
    const total = totalBid + totalAsk || 1;
    const bidPct = Math.round((totalBid / total) * 100);
    const askPct = 100 - bidPct;
    const mid = midPrice != null ? _fmtPrice(midPrice) : '--';
    imbalanceEl.className = 'dom-imbalance';
    imbalanceEl.innerHTML =
      `<span class="imbalance-bid">BID ${bidPct}%</span>` +
      `<span class="imbalance-bar"><span class="imbalance-fill" style="width:${bidPct}%"></span></span>` +
      `<span class="imbalance-mid">${mid}</span>` +
      `<span class="imbalance-ask">ASK ${askPct}%</span>`;
  }

  /** Return CSS class for fragility level. */
  function _fragClass(fragility) {
    if (!fragility) return '';
    switch (fragility.toUpperCase()) {
      case 'SOLID':    return 'frag-solid';
      case 'MODERATE': return 'frag-moderate';
      case 'FRAGILE':  return 'frag-fragile';
      default:         return '';
    }
  }

  /** Return flash CSS class by comparing current volume to previous. */
  function _flashClass(side, price, vol) {
    const key = `${side}:${price}`;
    const prev = prevVolMap[key];
    if (prev == null) return '';
    if (vol > prev) return 'flash-green';
    if (vol < prev) return 'flash-red';
    return '';
  }

  function _fmt(val) {
    if (val == null) return '-';
    return Number(val).toLocaleString();
  }

  function _fmtPrice(val) {
    if (val == null) return '--';
    return Number(val).toFixed(2);
  }

  return { init, updateFromFrame };
})();
