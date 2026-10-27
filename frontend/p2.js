// Phase 2 — shared helpers: API calls, toasts, theme, and rendering of
// assistant "views" (kpis / bar / line / table) with Chart.js.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = 'HTTP ' + r.status;
    try { const j = await r.json(); msg = j.detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

function toast(msg, type = 'ok') {
  let box = document.querySelector('.toasts');
  if (!box) { box = document.createElement('div'); box.className = 'toasts'; document.body.appendChild(box); }
  const t = document.createElement('div');
  t.className = 'toast ' + type; t.textContent = msg; box.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(20px)'; }, 3500);
  setTimeout(() => t.remove(), 3900);
}

(function initTheme() {
  try { if (localStorage.getItem('mom-theme') === 'light') document.documentElement.setAttribute('data-theme', 'light'); } catch (e) {}
})();

function fmtClock(s) {
  s = Math.max(0, Math.floor(s || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = String(s % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

// ---------------------------------------------------------------- views
const SERIES_COLORS = ['#7c5cff', '#22d3ee', '#f472b6', '#fbbf24', '#34d399', '#60a5fa'];
const charts = new WeakMap();

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function fmtVal(v, unit) {
  if (v === null || v === undefined) return '';
  return unit === '%' ? `${(+v).toFixed(1)}%` : Math.round(v).toLocaleString('en-IN');
}

function chartDefaults(big) {
  const fg = cssVar('--fg-dim'), grid = cssVar('--chart-grid');
  return {
    responsive: true, maintainAspectRatio: false, animation: { duration: 350 },
    plugins: { legend: { labels: { color: fg, font: { size: big ? 15 : 11 } } } },
    scales: {
      x: { ticks: { color: fg, font: { size: big ? 14 : 11 } }, grid: { color: grid } },
      y: { ticks: { color: fg, font: { size: big ? 14 : 11 } }, grid: { color: grid } },
    },
  };
}

// Render a view into `el`. opts.big = room-screen sizing.
function renderView(el, view, opts = {}) {
  const old = charts.get(el); if (old) { old.destroy(); charts.delete(el); }
  el.innerHTML = '';
  el.classList.add('view');
  const head = document.createElement('div');
  head.className = 'view-head';
  head.innerHTML = `<h3>${esc(view.title)}</h3>${view.subtitle ? `<p>${esc(view.subtitle)}</p>` : ''}`;
  el.appendChild(head);

  if (view.kind === 'kpis') {
    const g = document.createElement('div'); g.className = 'kpis';
    g.innerHTML = view.kpis.map(k => `
      <div class="kpi"><div class="k-label">${esc(k.label)}</div>
        <div class="k-val">${esc(k.value)}</div>
        <div class="k-delta ${k.good ? 'good' : 'bad'}">${esc(k.delta)}</div>
        <div class="k-scheme">${esc(k.scheme)}</div></div>`).join('');
    el.appendChild(g);
  } else if (view.kind === 'bar' || view.kind === 'line') {
    const box = document.createElement('div'); box.className = 'chart-box';
    if (opts.height) box.style.height = opts.height + 'px';
    const cv = document.createElement('canvas'); box.appendChild(cv); el.appendChild(box);
    const cfg = chartDefaults(opts.big);
    let data;
    if (view.kind === 'bar') {
      const b = view.bar;
      const ds = [{ label: view.title.split(' — ')[0], data: b.values, backgroundColor: b.labels.map((_, i) => SERIES_COLORS[i % SERIES_COLORS.length] + 'cc'), borderRadius: 6 }];
      if (b.reference !== null && b.reference !== undefined) {
        ds.push({ type: 'line', label: b.reference_label || 'Reference', data: b.labels.map(() => b.reference),
          borderColor: cssVar('--warn'), borderDash: [6, 4], pointRadius: 0, borderWidth: 2 });
      }
      data = { labels: b.labels, datasets: ds };
      if (b.horizontal) cfg.indexAxis = 'y';
      cfg.plugins.legend.display = ds.length > 1;
      cfg.plugins.tooltip = { callbacks: { label: c => `${c.dataset.label}: ${fmtVal(c.raw, b.unit)}` } };
      const valAxis = b.horizontal ? 'x' : 'y';
      cfg.scales[valAxis].ticks.callback = v => fmtVal(v, b.unit);
      if (b.unit === '%') { cfg.scales[valAxis].suggestedMin = 0; cfg.scales[valAxis].suggestedMax = 100; }
    } else {
      const l = view.line;
      data = { labels: l.labels, datasets: l.series.map((s, i) => ({ label: s.name, data: s.values,
        borderColor: SERIES_COLORS[i % SERIES_COLORS.length], backgroundColor: SERIES_COLORS[i % SERIES_COLORS.length] + '33',
        tension: .3, pointRadius: opts.big ? 4 : 2, borderWidth: opts.big ? 3 : 2 })) };
      cfg.plugins.tooltip = { callbacks: { label: c => `${c.dataset.label}: ${fmtVal(c.raw, l.unit)}` } };
      cfg.scales.y.ticks.callback = v => fmtVal(v, l.unit);
    }
    charts.set(el, new Chart(cv, { type: view.kind, data, options: cfg }));
  } else if (view.kind === 'table') {
    const t = view.table;
    if (!t.rows.length) {
      el.insertAdjacentHTML('beforeend', `<div class="empty">Nothing unusual found.</div>`);
    } else {
      el.insertAdjacentHTML('beforeend', `<table class="data"><thead><tr>${t.columns.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
        <tbody>${t.rows.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
    }
  }
  const foot = document.createElement('div');
  foot.className = 'view-foot';
  foot.innerHTML = `<span>As of ${esc(view.as_of)}</span><span class="src">${esc(view.source)}</span>`;
  el.appendChild(foot);
}
