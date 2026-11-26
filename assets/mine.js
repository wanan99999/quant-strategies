/* 「我的组合策略」：首页卡片 + 组合详情页（combo-view.html?id=）。只展示，不编辑 */
(() => {
const SB = { url: "https://vsojozoqbyvxszgfxaop.supabase.co", key: "sb_publishable_ykBgwF4pxNq7hSnuYASSqw_AXEIF346" };
const SIG = "https://wanan99999.github.io/signal-monitor/";
let IDX = {}, ST = [];                 // 指数名、行情名：读引擎导出的 meta.json（setMeta）
function setMeta(m) { if (m) { IDX = m.index_names || {}; ST = Object.entries(m.states || {}); } }
const stName = st => (ST.find(x => x[0] === st) || [st, st])[1];   // meta.json 没读到时直接显示代号，不报错
const E = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
async function sb(path) {
  let last;
  for (let i = 0; i < 4; i++) {
    try { const r = await fetch(SB.url + "/rest/v1/" + path, { headers: { apikey: SB.key } }); if (!r.ok) throw new Error(r.statusText); return await r.json(); }
    catch (e) { last = e; await new Promise(r => setTimeout(r, 600 * (i + 1))); }
  }
  throw last;
}
const getJ = u => fetch(u, { cache: "no-store" }).then(r => r.ok ? r.json() : null).catch(() => null);
function states(cfg) { const t = cfg.rule.type; return t === "fixed" ? ["normal"] : t === "two_state" ? ["normal", "small"] : ["normal", "small", "both"]; }
const short = n => String(n || "").split("（")[0];
function rows(cfg) {
  return states(cfg).map(st => {
    const row = Object.entries(cfg.alloc[st] || {}).filter(([, v]) => v > 0);
    const tot = row.reduce((a, [, v]) => a + v, 0);
    const parts = row.map(([k, v]) => ({ name: cfg.sleeves[k].name, hold: cfg.sleeves[k].params.hold, v }));
    if (tot < 0.9999) parts.push({ name: "货币基金", hold: null, v: 1 - tot });
    return { st, label: stName(st), parts };
  });
}
function oneLine(cfg) { return rows(cfg).map(r => r.label + "：" + r.parts.map(p => short(p.name) + (r.parts.length > 1 ? " " + Math.round(p.v * 100) + "%" : "")).join(" + ")).join("；"); }
function ruleText(cfg) {
  const r = cfg.rule, out = [];
  if (r.type === "fixed") return ["不切换，一直按「平常」的分法。"];
  out.push(`${IDX[r.small.index] || r.small.index} 最近 ${r.small.er_n} 天净涨幅至少占来回波动的 ${Math.round(r.small.er_min * 100)}%，并且在 ${r.small.ma_n} 天均价上方 → 中小盘走强。`);
  if (r.type === "three_state") out.push(`${IDX[r.big.index] || r.big.index} 在 ${r.big.ma_n} 天均价上方、均价比 ${r.big.slope_lag} 天前高、${Math.round(r.big.breadth_min * 100)}% 以上申万一级行业站上 ${r.big.breadth_ma} 天均价 → 大盘上涨期。`);
  if (+r.confirm_days > 0) out.push(`新行情要连续 ${r.confirm_days} 天成立才切换。`);
  return out;
}
function perf(c, res) {
  const j = c.jq, r = res && res[c.id];
  if (j && j.ann != null) return { v: (j.ann > 0 ? "+" : "") + (+j.ann).toFixed(1) + "%", tag: "聚宽实测" };
  if (r && r.ann != null) return { v: (r.ann > 0 ? "+" : "") + (r.ann * 100).toFixed(1) + "%", tag: r.ran && Date.parse(c.updated_at) > Date.parse(r.ran.replace(" ", "T") + "+08:00") ? "配置已改 · 今晚重算" : "本地测算" };
  return { v: "待测算", tag: "新组合 · 待测算" };
}
const ACT = { "买入": "buy", "卖出": "sell", "调仓": "mix" };
const V14 = "_v14";
const pc = (x, d = 1) => x == null ? "—" : (x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%";

/* 本地回测：净值曲线（本组合 vs 原方案 V1.4）+ 分年收益 */
function btBlock(name, r, ref) {
  const W = 640, H = 240, L = 46, R = 14, T = 12, B = 30;
  const ser = [{ k: "me", nm: name, c: r.curve }];
  if (ref && ref.curve) ser.push({ k: "ref", nm: "原方案 V1.4", c: ref.curve });
  const all = ser.flatMap(s => s.c.v), lo = Math.min(...all, 1), hi = Math.max(...all, 1);
  const pad = (hi - lo) * 0.06 || 0.05, y0 = lo - pad, y1 = hi + pad;
  const D0 = Date.parse(r.curve.d[0]), D1 = Date.parse(r.curve.d[r.curve.d.length - 1]);
  const X = d => L + (Date.parse(d) - D0) / (D1 - D0 || 1) * (W - L - R), Y = v => T + (y1 - v) / (y1 - y0) * (H - T - B);
  const step = (y1 - y0) > 1.2 ? 0.5 : (y1 - y0) > 0.5 ? 0.2 : 0.1;
  let grid = "";
  for (let g = Math.ceil(y0 / step) * step; g <= y1; g += step)
    grid += `<line x1="${L}" x2="${W - R}" y1="${Y(g).toFixed(1)}" y2="${Y(g).toFixed(1)}" class="${Math.abs(g - 1) < 1e-9 ? "g1" : "g"}"/><text x="${L - 6}" y="${(Y(g) + 4).toFixed(1)}" class="ax" text-anchor="end">${g.toFixed(1)}</text>`;
  let xt = "";
  for (let y = new Date(D0).getFullYear() + 1; y <= new Date(D1).getFullYear(); y++) { const x = X(y + "-01-01"); xt += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" class="g"/><text x="${x}" y="${H - 9}" class="ax" text-anchor="middle">${y}</text>`; }
  const path = s => s.c.d.map((d, i) => (i ? "L" : "M") + X(d).toFixed(1) + "," + Y(s.c.v[i]).toFixed(1)).join("");
  const lines = ser.slice().reverse().map(s => `<path d="${path(s)}" class="ln ln-${s.k}"/>`).join("");
  const ends = ser.map(s => { const n = s.c.v.length - 1; return `<circle cx="${X(s.c.d[n])}" cy="${Y(s.c.v[n])}" r="3.5" class="dt dt-${s.k}"/>`; }).join("");
  const years = Object.keys(r.years);
  const yrow = (x, nm, cls) => `<tr><td><span class="sw sw-${cls}"></span>${E(nm)}</td>${years.map(y => `<td class="num ${x && x.years[y] < 0 ? "neg" : ""}">${pc(x && x.years[y])}</td>`).join("")}<td class="num">${pc(x && x.ann)}</td><td class="num">${pc(x && x.mdd)}</td></tr>`;
  return `<section class="blk bt"><h2>本地回测 <small>${E(r.start)} 起 · 截至 ${E(r.asof)} · 每周一个点 · 含手续费估算</small></h2>
    <div class="bt-leg">${ser.map(s => `<span><i class="sw sw-${s.k}"></i>${E(s.nm)} <b>${pc(s.k === "me" ? r.cum : ref.cum, 0)}</b></span>`).join("")}</div>
    <div class="bt-chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="净值曲线">${grid}${xt}${lines}${ends}<line class="cx" y1="${T}" y2="${H - B}" x1="-9" x2="-9"/></svg><div class="bt-tip" hidden></div></div>
    <div class="tblwrap"><table class="ctbl bt-tbl"><thead><tr><th>组合</th>${years.map(y => `<th>${y}${y === String(new Date(D1).getFullYear()) ? "*" : ""}</th>`).join("")}<th>年化</th><th>最大回撤</th></tr></thead>
      <tbody>${yrow(r, name, "me")}${ref ? yrow(ref, "原方案 V1.4", "ref") : ""}</tbody></table></div>
    <p class="hint">* 当年至今。本地电脑每晚收盘后重算。用当前上市公司名单和当前行业分类回看，会略高估；以聚宽实测为准。</p></section>`;
}
/* 每个子策略单独全程跑的成绩（夜间回测 r.parts） */
function partsBlock(r) {
  const ps = (r && r.parts) || [];
  if (!ps.length) return "";
  const years = Object.keys(r.years), lastY = years[years.length - 1];
  const use = u => Object.entries(u).map(([st, v]) => `${stName(st)} ${Math.round(v * 100)}%`).join("<br>");
  const row = (nm, x, u, cls) => `<tr class="${cls || ""}"><td>${nm}</td><td class="use">${u}</td>${years.map(y => `<td class="num ${x.years[y] < 0 ? "neg" : ""}">${pc(x.years[y])}</td>`).join("")}<td class="num"><b>${pc(x.ann)}</b></td><td class="num">${pc(x.mdd)}</td></tr>`;
  return `<section class="blk"><h2>每个策略单独跑 <small>${E(r.start)} 起 · 一直满仓用它，不切换</small></h2>
    <div class="tblwrap"><table class="ctbl bt-tbl parts-tbl"><thead><tr><th>策略</th><th>在组合里什么时候用</th>${years.map(y => `<th>${y}${y === lastY ? "*" : ""}</th>`).join("")}<th>年化</th><th>最大回撤</th></tr></thead>
    <tbody>${ps.map(p => row(`${E(short(p.name))}${p.hold ? `<small>（${p.hold} 只）</small>` : ""}`, p, use(p.use))).join("")}
    ${row("<b>组合（按行情切换）</b>", r, "—", "tot")}</tbody></table></div>
    <p class="hint">单独跑 = 整个期间都满仓交给这一个策略。组合只在对应行情时才用它，所以组合收益不等于这几行的平均。* 当年至今。</p></section>`;
}
function btHover(root, r, ref) {
  const svg = root.querySelector(".bt-chart svg"), tip = root.querySelector(".bt-tip"), cx = svg && svg.querySelector(".cx");
  if (!svg) return;
  const W = 640, L = 46, R = 14, d = r.curve.d, D0 = Date.parse(d[0]), D1 = Date.parse(d[d.length - 1]);
  const refMap = ref && ref.curve ? Object.fromEntries(ref.curve.d.map((x, i) => [x, ref.curve.v[i]])) : {};
  svg.addEventListener("pointermove", e => {
    const b = svg.getBoundingClientRect(), x = (e.clientX - b.left) / b.width * W;
    const t = D0 + (x - L) / (W - L - R) * (D1 - D0);
    let i = 0, best = 1e18; d.forEach((s, j) => { const k = Math.abs(Date.parse(s) - t); if (k < best) { best = k; i = j; } });
    const px = L + (Date.parse(d[i]) - D0) / (D1 - D0) * (W - L - R);
    cx.setAttribute("x1", px); cx.setAttribute("x2", px);
    tip.hidden = false;
    tip.innerHTML = `<div class="tt-d">${d[i]}</div><div><i class="sw sw-me"></i>净值 <b>${r.curve.v[i].toFixed(3)}</b></div>${refMap[d[i]] != null ? `<div><i class="sw sw-ref"></i>V1.4 <b>${refMap[d[i]].toFixed(3)}</b></div>` : ""}`;
    const left = px / W * b.width; tip.style.left = Math.min(Math.max(left - tip.offsetWidth / 2, 0), b.width - tip.offsetWidth) + "px";
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; cx.setAttribute("x1", -9); cx.setAttribute("x2", -9); });
}

/* 首页卡片 */
async function home(el) {
  try {
    const [list, res, sig, meta] = await Promise.all([sb("combos?select=id,name,note,config,jq,updated_at&order=created_at.asc"), getJ(SIG + "data/combos/results.json"), getJ(SIG + "data/signals.json"), getJ(SIG + "data/meta.json")]);
    setMeta(meta);
    const cnt = document.getElementById("mineCnt"); if (cnt) cnt.textContent = list.length;
    const nav = document.getElementById("mineNav"); if (nav) nav.textContent = list.length;
    const today = {}; for (const s of (sig && sig.strategies) || []) if (s.combo_id) { const x = s.starts[s.default_start] || []; today[s.combo_id] = x[0]; }
    el.innerHTML = list.map(c => {
      const p = perf(c, res), a = today[c.id];
      return `<a class="card" href="combo-view.html?id=${c.id}"><div class="chips"><span class="chip">${E(p.tag)}</span>${a ? `<span class="chip act-${ACT[a] || "idle"}">今天${E(a)}</span>` : ""}</div>
        <h3>${E(c.name)}</h3><p>${E(c.note || oneLine(c.config))}</p>
        <div class="cm"><span class="cmk">2023 年至今年化收益率</span><span class="cmv">${E(p.v)}</span></div>
        <div class="cnt">更新于 ${E(c.updated_at.slice(0, 10))}</div></a>`;
    }).join("") + `<a class="card card-new" href="combo.html"><h3>＋ 新建组合</h3><p>在组合配置页挑切换规则、给三个位置放策略和比例。</p></a>`;
  } catch (e) { el.innerHTML = `<p class="empty">读取失败：${E(e.message)}，请稍后刷新。</p>`; }
}

/* 详情页 */
async function view(el) {
  const id = new URLSearchParams(location.search).get("id");
  if (!id) { el.innerHTML = `<p class="empty">缺少组合编号。</p>`; return; }
  try {
    const [list, res, sig, meta] = await Promise.all([sb("combos?select=id,name,note,config,jq,created_at,updated_at&id=eq." + id), getJ(SIG + "data/combos/results.json"), getJ(SIG + "data/signals.json"), getJ(SIG + "data/meta.json")]);
    setMeta(meta);
    const c = list[0];
    if (!c) { el.innerHTML = `<p class="empty">这个组合不存在或已删除。</p>`; return; }
    document.title = c.name + " · 我的组合策略";
    const p = perf(c, res), r = res && res[c.id], j = c.jq, cfg = c.config;
    const s = sig && (sig.strategies || []).find(x => x.combo_id === c.id);
    const x = s ? s.starts[s.default_start] || [] : [];
    el.innerHTML = `
      <header class="shead"><div class="chips"><span class="chip c-mine">我的组合策略</span><span class="chip">${E(p.tag)}</span></div>
        <h1>${E(c.name)}</h1><p class="lede">${E(c.note || oneLine(cfg))}</p>
        <div class="vbtns"><a class="vbtn primary" href="${SIG}#/${s ? s.id : ""}">看今天的操作提示</a><a class="vbtn" href="combo.html?id=${c.id}">修改配置</a></div></header>
      <section class="kpis">
        <div class="kpi"><div class="k">本地测算 · 2023 年起</div><div class="v">${r ? (r.ann * 100).toFixed(1) + "%" : "待测算"}</div><div class="s">${r ? `年化；最多跌 ${(r.mdd * 100).toFixed(1)}% · 累计 ${(r.cum * 100).toFixed(0)}%${r.asof ? " · 截至 " + r.asof : ""}` : "保存后当晚本地电脑算"}</div></div>
        <div class="kpi"><div class="k">聚宽实测</div><div class="v">${j && j.ann != null ? (+j.ann).toFixed(1) + "%" : "未填"}</div><div class="s">${j && j.ann != null ? `年化${j.dd != null ? "；最多跌 " + (+j.dd).toFixed(1) + "%" : ""}${j.cum != null ? " · 累计 " + (+j.cum).toFixed(1) + "%" : ""}${j.sharpe != null ? " · 夏普 " + j.sharpe : ""}${j.range ? " · " + E(j.range) : ""}` : "跑完聚宽后在组合配置页填"}</div></div>
        <div class="kpi"><div class="k">今天（${E(sig ? sig.date : "")}）</div><div class="v">${E(x[0] || "—")}</div><div class="s">${s ? `买 ${x[1] || 0} · 卖 ${x[2] || 0} · 持有 ${x[3] || 0} 只（从 ${E(s.default_start)} 起跟）` : "保存后下一个交易日 14:30 起出提示"}</div></div>
      </section>
      ${r && r.curve ? btBlock(c.name, r, res[V14]) : ""}
      ${partsBlock(r)}
      <section class="blk"><h2>什么时候切换</h2><ul class="bul">${ruleText(cfg).map(t => `<li>${E(t)}</li>`).join("")}</ul></section>
      <section class="blk"><h2>每种行情下钱怎么分</h2>
        <div class="tblwrap"><table class="ctbl"><thead><tr><th>行情</th><th>交给哪些策略</th></tr></thead><tbody>
        ${rows(cfg).map(rw => `<tr><td>${E(rw.label)}</td><td>${rw.parts.map(q => `${E(q.name)} <b>${Math.round(q.v * 100)}%</b>${q.hold ? `（${q.hold} 只）` : ""}`).join("<br>")}</td></tr>`).join("")}
        </tbody></table></div></section>
      <section class="blk"><h2>说明</h2><ul class="bul">
        <li>每个策略在自己的模拟账户里按自己的规则买卖（风控、轮换、仓位都不变）；行情切换那天才按上表重新分钱。</li>
        <li>两个策略选中同一只股票时合并成一笔，各自记成本。</li>
        <li>最多同时持有 ${cfg.max_hold} 只。创建于 ${E(c.created_at.slice(0, 10))}，最后修改 ${E(c.updated_at.slice(0, 10))}。</li></ul></section>`;
    if (r && r.curve) btHover(el, r, res[V14]);
  } catch (e) { el.innerHTML = `<p class="empty">读取失败：${E(e.message)}，请稍后刷新。</p>`; }
}
const h = document.getElementById("mineCards"); if (h) home(h);
const v = document.getElementById("mineView"); if (v) view(v);
})();
