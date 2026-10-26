(() => {
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (n, d = 2) => Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const pct = (n, d = 2) => (n >= 0 ? "+" : "") + (n * 100).toFixed(d) + "%";
  let lastData = null;

  /* ---------- tiny canvas line chart with band + hover tooltip ---------- */
  function lineChart(canvas, { labels, series, band, yFmt = (v) => fmt(v), splitAt = null }) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = parseInt(canvas.getAttribute("height"), 10);
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const pad = { l: 56, r: 14, t: 10, b: 24 };
    const n = labels.length;
    let all = series.flatMap((s) => s.data.filter((v) => v != null));
    if (band) all = all.concat(band.low.filter((v) => v != null), band.high.filter((v) => v != null));
    let min = Math.min(...all), max = Math.max(...all);
    const m = (max - min) * 0.08 || 1; min -= m; max += m;
    const x = (i) => pad.l + (i / Math.max(1, n - 1)) * (W - pad.l - pad.r);
    const y = (v) => pad.t + (1 - (v - min) / (max - min)) * (H - pad.t - pad.b);

    function draw(hover) {
      ctx.clearRect(0, 0, W, H);
      ctx.font = "11px system-ui,sans-serif";
      ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const v = min + ((max - min) * g) / 4, yy = y(v);
        ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(W - pad.r, yy); ctx.stroke();
        ctx.textAlign = "right"; ctx.fillText(yFmt(v), pad.l - 6, yy + 4);
      }
      ctx.textAlign = "center";
      [0, Math.floor(n / 2), n - 1].forEach((i) => ctx.fillText(labels[i], Math.min(Math.max(x(i), pad.l + 30), W - pad.r - 30), H - 6));
      if (band) {
        ctx.beginPath(); let started = false;
        band.high.forEach((v, i) => { if (v == null) return; started ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)); started = true; });
        for (let i = n - 1; i >= 0; i--) if (band.low[i] != null) ctx.lineTo(x(i), y(band.low[i]));
        ctx.closePath(); ctx.fillStyle = "rgba(18,184,134,.22)"; ctx.fill();
      }
      series.forEach((s) => {
        ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
        let started = false;
        s.data.forEach((v, i) => { if (v == null) { return; } started ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)); started = true; });
        ctx.stroke(); ctx.setLineDash([]);
      });
      if (splitAt != null) {
        ctx.strokeStyle = css("--line"); ctx.setLineDash([4, 4]); ctx.beginPath();
        ctx.moveTo(x(splitAt), pad.t); ctx.lineTo(x(splitAt), H - pad.b); ctx.stroke(); ctx.setLineDash([]);
      }
      if (hover != null) {
        ctx.strokeStyle = css("--muted"); ctx.beginPath(); ctx.moveTo(x(hover), pad.t); ctx.lineTo(x(hover), H - pad.b); ctx.stroke();
        const rows = series.filter((s) => s.data[hover] != null).map((s) => ({ c: s.color, t: `${s.name}: ${yFmt(s.data[hover])}` }));
        const head = labels[hover];
        const w = Math.max(...rows.map((r) => ctx.measureText(r.t).width), ctx.measureText(head).width) + 26;
        const h = 18 + rows.length * 16;
        let bx = x(hover) + 12; if (bx + w > W) bx = x(hover) - w - 12;
        ctx.fillStyle = css("--panel2"); ctx.strokeStyle = css("--line");
        ctx.beginPath(); ctx.roundRect(bx, pad.t + 4, w, h, 6); ctx.fill(); ctx.stroke();
        ctx.textAlign = "left"; ctx.fillStyle = css("--muted"); ctx.fillText(head, bx + 8, pad.t + 18);
        rows.forEach((r, k) => { ctx.fillStyle = r.c; ctx.fillRect(bx + 8, pad.t + 26 + k * 16, 8, 8); ctx.fillStyle = css("--text"); ctx.fillText(r.t, bx + 22, pad.t + 34 + k * 16); });
      }
    }
    canvas.onmousemove = (e) => {
      const r = canvas.getBoundingClientRect();
      const i = Math.round(((e.clientX - r.left - pad.l) / (W - pad.l - pad.r)) * (n - 1));
      draw(Math.min(n - 1, Math.max(0, i)));
    };
    canvas.onmouseleave = () => draw(null);
    draw(null);
  }

  /* ---------- rendering ---------- */
  function renderTable(el, head, rows) {
    el.innerHTML = "<thead><tr>" + head.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
      rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody>";
  }

  function render(d) {
    lastData = d;
    const p = d.prediction;
    $("k-last").textContent = fmt(p.last_close);
    $("k-pred-label").textContent = `Predicted price (${d.horizon}d)`;
    $("k-pred").textContent = fmt(p.predicted_price);
    const cls = p.predicted_return > 0.002 ? "up" : p.predicted_return < -0.002 ? "down" : "flat";
    $("k-ret").textContent = pct(p.predicted_return); $("k-ret").className = cls;
    $("k-range").textContent = `${fmt(p.low_price, 1)} – ${fmt(p.high_price, 1)}`;
    const sig = $("k-signal"); sig.textContent = p.signal; sig.className = "signal " + cls;

    const b = $("badge");
    b.textContent = d.source === "yahoo" ? "● Live data" : "Demo data (offline)";
    b.className = "badge" + (d.source === "yahoo" ? " live" : "");

    drawCharts();

    const bt = d.backtest;
    renderTable($("t-bt"), ["", "Total return", "Sharpe", "Max drawdown"], [
      ["Strategy", pct(bt.strategy.total_return, 1), bt.strategy.sharpe.toFixed(2), pct(bt.strategy.max_drawdown, 1)],
      ["Buy & hold", pct(bt.buy_and_hold.total_return, 1), bt.buy_and_hold.sharpe.toFixed(2), pct(bt.buy_and_hold.max_drawdown, 1)],
    ]);

    const maxImp = Math.max(...d.importance.map((i) => i.value));
    $("imp").innerHTML = d.importance.map((i) =>
      `<div class="bar"><span class="name" title="${i.feature}">${i.feature}</span><span class="track"><span class="fill" style="display:block;width:${(i.value / maxImp) * 100}%"></span></span><span class="val">${(i.value * 100).toFixed(1)}%</span></div>`).join("");

    const votes = Object.entries(d.per_model);
    const vmax = Math.max(...votes.map(([, v]) => Math.abs(v)), 0.001);
    $("votes").innerHTML = votes.map(([k, v]) => {
      const w = (Math.abs(v) / vmax) * 50;
      const bar = v >= 0
        ? `<span style="position:absolute;left:50%;width:${w}%;height:100%;background:var(--green);border-radius:0 5px 5px 0"></span>`
        : `<span style="position:absolute;right:50%;width:${w}%;height:100%;background:var(--red);border-radius:5px 0 0 5px"></span>`;
      return `<div class="bar"><span class="name">${k.replace("_", " ")}</span><span class="track">${bar}<span style="position:absolute;left:50%;width:1px;height:100%;background:var(--muted)"></span></span><span class="val ${v >= 0 ? "up" : "down"}">${pct(v)}</span></div>`;
    }).join("");

    const rows = Object.entries(d.metrics).map(([k, m]) => [
      k.replace(/_/g, " "),
      m.mae != null ? (m.mae * 100).toFixed(2) + "%" : "–",
      m.rmse != null ? (m.rmse * 100).toFixed(2) + "%" : "–",
      (m.directional_accuracy * 100).toFixed(1) + "%",
    ]);
    renderTable($("t-metrics"), ["Model", "MAE", "RMSE", "Direction accuracy"], rows);
  }

  function drawCharts() {
    if (!lastData) return;
    const d = lastData, h = d.history, f = d.forecast;
    const labels = h.dates.concat(f.dates), nh = h.dates.length;
    const closeS = h.close.concat(new Array(f.dates.length).fill(null));
    // connect forecast line to last close
    const fcS = new Array(nh - 1).fill(null).concat([h.close[nh - 1]], f.mid);
    const lab = h.dates.concat(f.dates);
    const L = fcS.length;
    lineChart($("c-price"), {
      labels: lab.slice(0, L),
      series: [
        { name: "Close", color: css("--blue"), data: closeS.slice(0, L) },
        { name: "Forecast", color: css("--green"), dash: [6, 4], data: fcS },
      ],
      band: {
        low: new Array(nh - 1).fill(null).concat([h.close[nh - 1]], f.low),
        high: new Array(nh - 1).fill(null).concat([h.close[nh - 1]], f.high),
      },
      splitAt: nh - 1,
    });
    const bt = d.backtest;
    lineChart($("c-eq"), {
      labels: bt.dates,
      yFmt: (v) => v.toFixed(2) + "x",
      series: [
        { name: "Strategy", color: css("--green"), data: bt.equity_strategy },
        { name: "Buy & hold", color: css("--blue"), data: bt.equity_buyhold },
      ],
    });
  }

  /* ---------- events ---------- */
  function setStatus(msg, kind) {
    const s = $("status");
    if (!msg) { s.className = "status hidden"; return; }
    s.textContent = msg; s.className = "status " + kind;
  }

  async function run(e) {
    if (e) e.preventDefault();
    const t = $("ticker").value.trim().toUpperCase();
    if (!t) return setStatus("Please enter a ticker symbol.", "error");
    const qs = new URLSearchParams({ ticker: t, period: $("period").value, horizon: $("horizon").value, days: $("days").value });
    $("go").disabled = true;
    setStatus(`Downloading data and training models for ${t}… (first run takes ~10–30s)`, "loading");
    try {
      const r = await fetch("/api/predict?" + qs);
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Request failed");
      $("results").classList.remove("hidden");
      setStatus(null);
      render(d);
    } catch (err) {
      setStatus("Error: " + err.message, "error");
    } finally {
      $("go").disabled = false;
    }
  }

  $("form").addEventListener("submit", run);
  $("chips").addEventListener("click", (e) => {
    if (e.target.dataset.t) { $("ticker").value = e.target.dataset.t; run(); }
  });
  window.addEventListener("resize", drawCharts);
})();
