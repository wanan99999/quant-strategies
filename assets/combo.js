/* 组合配置页：编辑配置 → 检查 → 存到 Supabase（口令）。配置格式与 signal-monitor/engine/combo2.py 一致 */
(() => {
const SB = { url: "https://vsojozoqbyvxszgfxaop.supabase.co", key: "sb_publishable_ykBgwF4pxNq7hSnuYASSqw_AXEIF346" };
const SIG = "https://wanan99999.github.io/signal-monitor/";
let SLOTS = [], SLOTN = {};                             // 三个位置 [id, 名称, 什么时候用]：读引擎导出的 meta.json
let STATES = [];                                       // 行情名：读引擎导出的 meta.json
const $ = id => document.getElementById(id);
const E = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const clone = o => JSON.parse(JSON.stringify(o));
let P = null, MK = null, PL = {}, M = null, CUR = { id: null }, LIST = [];

/* ---------------- Supabase ---------------- */
async function sb(path, opt = {}) {
  let last;
  for (let i = 0; i < 4; i++) {
    try {
      const r = await fetch(SB.url + "/rest/v1/" + path, { ...opt, headers: { apikey: SB.key, "Content-Type": "application/json", ...(opt.headers || {}) } });
      const t = await r.text(); const j = t ? JSON.parse(t) : null;
      if (!r.ok) throw Object.assign(new Error((j && j.message) || r.statusText), { fatal: r.status < 500 });
      return j;
    } catch (e) { last = e; if (e.fatal) break; await new Promise(r => setTimeout(r, 600 * (i + 1))); }
  }
  throw last;
}
const rpc = (fn, body) => sb("rpc/" + fn, { method: "POST", body: JSON.stringify(body) });
let PASS = null;
try { PASS = sessionStorage.getItem("combo-pass"); } catch (e) {}
function askPass() {
  return new Promise(res => {
    if (PASS) return res(PASS);
    const d = $("passDlg"); $("passIn").value = ""; $("passErr").textContent = "";
    d.onclose = async () => {
      if (d.returnValue !== "ok") return res(null);
      const p = $("passIn").value;
      try {
        if (await rpc("check_pass", { p_pass: p })) { PASS = p; try { sessionStorage.setItem("combo-pass", p); } catch (e) {} return res(p); }
        toast("口令不对"); res(null);
      } catch (e) { toast("连接失败：" + e.message); res(null); }
    };
    d.showModal(); $("passIn").focus();
  });
}
function toast(t) { const x = $("toast"); x.textContent = t; x.hidden = false; clearTimeout(x._t); x._t = setTimeout(() => x.hidden = true, 3200); }

/* ---------------- 模型 ⇄ 引擎配置 ---------------- */
function defaults(pid) { const o = {}; for (const p of PL[pid].params) o[p.key] = p.default; return o; }
function defaultModel() {
  return {
    max_hold: 10, tol: 0.25, rule: clone(P.rule_default),
    slotAlloc: { normal: { steady: 100, stock: 0, sector: 0 }, small: { steady: 0, stock: 100, sector: 0 }, both: { steady: 0, stock: 50, sector: 50 } },
    slots: {
      steady: [{ plugin: "steady", share: 100, params: { ...defaults("steady"), hold: 10, buffer: 20, ind_max: 3 }, hold_both: null }],
      stock: [{ plugin: "smallcap", share: 100, params: { ...defaults("smallcap"), hold: 10, buffer: 20 }, hold_both: 5 }],
      sector: [{ plugin: "sector_etf", share: 100, params: { ...defaults("sector_etf"), hold: 3 }, hold_both: null }],
    },
  };
}
function activeStates() { return M.rule.type === "fixed" ? ["normal"] : M.rule.type === "two_state" ? ["normal", "small"] : ["normal", "small", "both"]; }
function holdOf(it, st) { return st === "both" && it.hold_both ? +it.hold_both : +it.params.hold; }
function toCfg() {
  const sleeves = {}, alloc = {};
  for (const st of ["normal", "small", "both"]) alloc[st] = {};
  for (const [slot] of SLOTS) M.slots[slot].forEach((it, i) => {
    const id = slot + i, nm = PL[it.plugin].name;
    sleeves[id] = { plugin: it.plugin, slot, name: nm, params: clone(it.params) };
    const bothDiff = it.hold_both && +it.hold_both !== +it.params.hold;
    if (bothDiff) sleeves[id + "b"] = { plugin: it.plugin, slot, name: nm + "（上涨期 " + it.hold_both + " 只）", params: { ...clone(it.params), hold: +it.hold_both } };
    for (const st of activeStates()) {
      const v = (M.slotAlloc[st][slot] || 0) / 100 * (it.share || 0) / 100;
      if (v > 0) alloc[st][st === "both" && bothDiff ? id + "b" : id] = +v.toFixed(4);
    }
  });
  return { max_hold: +M.max_hold, tol: M.tol, rule: clone(M.rule), sleeves, alloc, ui: clone(M) };
}

/* ---------------- 检查 ---------------- */
function checks() {
  const out = [];
  const used = new Set();
  for (const st of activeStates()) {
    const row = M.slotAlloc[st]; const tot = SLOTS.reduce((a, [s]) => a + (+row[s] || 0), 0);
    const nm = STATES.find(x => x[0] === st)[1];
    out.push([tot <= 100.0001, `「${nm}」三个位置合计 ${tot}%` + (tot < 100 ? `，其余 ${100 - tot}% 买货币基金` : "") + (tot > 100 ? "，超过 100%" : "")]);
    let n = 0;
    for (const [s] of SLOTS) if (+row[s] > 0) { used.add(s); for (const it of M.slots[s]) if (it.share > 0) n += holdOf(it, st); }
    out.push([n <= +M.max_hold, `「${nm}」最多可能同时持 ${n} 只（上限 ${M.max_hold}）`]);
  }
  for (const [s, nm] of SLOTS) {
    if (!used.has(s)) continue;
    const its = M.slots[s];
    if (!its.length) { out.push([false, `${nm}分到了钱，但没有放策略`]); continue; }
    const t = its.reduce((a, it) => a + (+it.share || 0), 0);
    out.push([Math.abs(t - 100) < 0.01, `${nm}里各策略比例合计 ${t}%（要正好 100%）`]);
  }
  for (const [s, nm] of SLOTS) for (const it of M.slots[s]) {
    if (!used.has(s)) continue;
    for (const p of PL[it.plugin].params) {
      const v = it.params[p.key];
      if (typeof p.default === "number" && (v === "" || isNaN(+v) || +v < p.min || +v > p.max)) out.push([false, `${nm} · ${PL[it.plugin].name}：「${p.label}」要在 ${p.min}–${p.max} 之间`]);
    }
  }
  const r = M.rule;
  if (r.type !== "fixed") {
    if (!(r.small.er_n >= 5 && r.small.er_n <= 120)) out.push([false, "中小盘判断的天数要在 5–120"]);
    if (!(r.small.ma_n >= 5 && r.small.ma_n <= 250)) out.push([false, "中小盘均价天数要在 5–250"]);
  }
  if (r.type === "three_state" && !(r.big.ma_n >= 20 && r.big.ma_n <= 250)) out.push([false, "大盘均价天数要在 20–250"]);
  return out;
}

/* ---------------- 切换规则统计（浏览器里直接算，2023 年起） ---------------- */
function erOk(x, n, th, ma) {
  if (x.length < Math.max(n + 1, ma)) return [false, 0];
  let path = 0; for (let i = x.length - n; i < x.length; i++) path += Math.abs(x[i] - x[i - 1]);
  const er = path > 0 ? (x[x.length - 1] - x[x.length - 1 - n]) / path : 0;
  let s = 0; for (let i = x.length - ma; i < x.length; i++) s += x[i];
  return [er > th && x[x.length - 1] > s / ma, er];
}
function mean(a, i0, i1) { let s = 0, n = 0; for (let i = i0; i < i1; i++) if (a[i] != null) { s += a[i]; n++; } return n ? s / n : NaN; }
function ruleSeries() {
  const r = M.rule, D = MK.dates, out = [];
  const xs = MK.index[r.small.index], hs = MK.index[r.big.index];
  const i0 = D.findIndex(d => d >= "2023-01-03");
  let state = null, pend = null, cnt = 0; const nconf = +r.confirm_days || 0;
  for (let i = i0; i < D.length; i++) {
    let raw = "normal";
    if (r.type !== "fixed") {
      const x = xs.slice(Math.max(0, i - 260), i + 1).filter(v => v != null);
      const g1 = erOk(x, +r.small.er_n, +r.small.er_min, +r.small.ma_n)[0];
      let up = false;
      if (r.type === "three_state" && g1) {
        const h = hs.slice(Math.max(0, i - 300), i).filter(v => v != null);
        const mn = +r.big.ma_n, lag = +r.big.slope_lag, bm = +r.big.breadth_ma;
        if (h.length >= mn + lag) {
          const now = mean(h, h.length - mn, h.length), old = mean(h, h.length - mn - lag, h.length - lag);
          let ok = 0, tot = 0;
          for (const row of MK.sw1) {
            let c = 0; for (let k = Math.max(0, i - bm - 70); k < i; k++) if (row[k] != null) c++;
            if (c < bm || row[i - 1] == null) continue;
            tot++; if (row[i - 1] > mean(row, i - bm, i)) ok++;
          }
          up = h[h.length - 1] > now && now > old && tot > 0 && ok / tot >= +r.big.breadth_min;
        }
      }
      raw = g1 && up ? "both" : g1 ? "small" : "normal";
    }
    if (state === null || nconf <= 0) state = raw;
    else if (raw === state) { pend = null; cnt = 0; }
    else if (raw === pend) { if (++cnt >= nconf) { state = raw; pend = null; cnt = 0; } }
    else { pend = raw; cnt = 1; if (cnt >= nconf) { state = raw; pend = null; cnt = 0; } }
    out.push([D[i], state]);
  }
  return out;
}
let statT = null;
function renderStats() {
  clearTimeout(statT);
  statT = setTimeout(() => {
    const s = ruleSeries(); if (!s.length) return;
    const n = s.length, cnt = {}, sw = {};
    let prev = null, chg = 0;
    for (const [d, st] of s) { cnt[st] = (cnt[st] || 0) + 1; if (prev && st !== prev) { chg++; const y = d.slice(0, 4); sw[y] = (sw[y] || 0) + 1; } prev = st; }
    const yrs = n / 244, cur = s[n - 1];
    const bar = activeStates().map(st => `<i class="s-${st}" style="flex:${cnt[st] || 0}"></i>`).join("");
    const strip = s.map(([, st]) => `<i class="s-${st}"></i>`).join("");
    $("ruleStats").innerHTML = `<h3>按这套规则，2023 年以来</h3>
      <div class="sbar">${bar}</div>
      <table class="tb"><tbody>${activeStates().map(st => `<tr><td><span class="sw s-${st}"></span>${STATES.find(x => x[0] === st)[1]}</td><td class="num">${((cnt[st] || 0) / n * 100).toFixed(0)}%</td><td class="num">${cnt[st] || 0} 天</td></tr>`).join("")}</tbody></table>
      <p class="kv">切换 <b class="num">${chg}</b> 次，平均每年 <b class="num">${(chg / yrs).toFixed(0)}</b> 次（${Object.entries(sw).map(([y, v]) => y + "：" + v).join("，")}）</p>
      <div class="strip" title="每一格是一个交易日">${strip}</div><div class="strip-ax"><span>${s[0][0]}</span><span>${cur[0]}</span></div>
      <p class="kv">最近一个交易日（${cur[0]}）是「${STATES.find(x => x[0] === cur[1])[1]}」</p>`;
  }, 120);
}

/* ---------------- 渲染 ---------------- */
function idxOpts(v) { return Object.entries(P.index_names).map(([k, n]) => `<option value="${k}"${k === v ? " selected" : ""}>${E(n)}</option>`).join(""); }
function num(path, v, opt = {}) { return `<input type="number" data-k="${path}" value="${v}" ${opt.step ? `step="${opt.step}"` : ""} ${opt.min != null ? `min="${opt.min}"` : ""} ${opt.max != null ? `max="${opt.max}"` : ""}>`; }
function renderRule() {
  const r = M.rule, t = r.type;
  $("ruleForm").innerHTML = `
    <div class="seg" role="radiogroup">${[["three_state", "原方案三段"], ["two_state", "只要两段（不要板块位）"], ["fixed", "不切换"]].map(([k, n]) => `<button class="${t === k ? "on" : ""}" data-type="${k}">${n}</button>`).join("")}</div>
    ${t === "fixed" ? `<p class="hint">不切换：一直按「平常」那一行分钱。</p>` : `
    <div class="rg"><h4>中小盘涨起来了吗（每天看，含当天价格）</h4>
      <p class="sent">看 <select data-k="rule.small.index">${idxOpts(r.small.index)}</select> 最近 ${num("rule.small.er_n", r.small.er_n, { min: 5, max: 120 })} 天，
      净涨幅至少占来回波动的 ${num("rule.small.er_min_pct", Math.round(r.small.er_min * 100), { min: 0, max: 100 })}%，
      并且价格高于最近 ${num("rule.small.ma_n", r.small.ma_n, { min: 5, max: 250 })} 天的平均价。</p></div>`}
    ${t === "three_state" ? `
    <div class="rg"><h4>大盘在上涨期吗（用前一天收盘）</h4>
      <p class="sent"><select data-k="rule.big.index">${idxOpts(r.big.index)}</select> 高于最近 ${num("rule.big.ma_n", r.big.ma_n, { min: 20, max: 250 })} 天平均价，
      这条平均价比 ${num("rule.big.slope_lag", r.big.slope_lag, { min: 1, max: 120 })} 天前高，
      而且至少 ${num("rule.big.breadth_min_pct", Math.round(r.big.breadth_min * 100), { min: 0, max: 100 })}% 的申万一级行业站在各自 ${num("rule.big.breadth_ma", r.big.breadth_ma, { min: 10, max: 120 })} 天平均价上方。</p></div>` : ""}
    ${t !== "fixed" ? `<div class="rg"><h4>防止来回切换</h4><p class="sent">新的行情要连续 ${num("rule.confirm_days", r.confirm_days, { min: 0, max: 20 })} 天都成立才切换（0 = 当天就切，原方案）。</p></div>` : ""}`;
}
function renderAlloc() {
  const sts = activeStates();
  $("allocTable").innerHTML = `<div class="tbw"><table class="tb alloc"><thead><tr><th>行情</th>${SLOTS.map(s => `<th>${s[1]}</th>`).join("")}<th>货币基金</th></tr></thead><tbody>
    ${sts.map(st => { const row = M.slotAlloc[st]; const tot = SLOTS.reduce((a, [s]) => a + (+row[s] || 0), 0);
      return `<tr><td><span class="sw s-${st}"></span>${STATES.find(x => x[0] === st)[1]}</td>${SLOTS.map(([s]) => `<td><span class="pct">${num(`alloc.${st}.${s}`, row[s] || 0, { min: 0, max: 100 })}%</span></td>`).join("")}<td class="num ${tot > 100 ? "bad" : ""}">${100 - tot}%</td></tr>`; }).join("")}
  </tbody></table></div>`;
}
function slotUsed(s) { return activeStates().some(st => +M.slotAlloc[st][s] > 0); }
function renderSlots() {
  $("slots").innerHTML = SLOTS.map(([s, nm, ds]) => {
    const used = slotUsed(s), inBoth = activeStates().includes("both") && +M.slotAlloc.both[s] > 0;
    return `<div class="panel slot ${used ? "" : "off"}"><div class="slot-h"><h3>${nm}</h3><span class="hint">${ds}${used ? "" : " · 现在没有分到钱"}</span></div>
      ${M.slots[s].map((it, i) => { const p = PL[it.plugin]; const odd = !p.slots.includes(s);
        return `<div class="item">
          <div class="item-top"><select data-k="slot.${s}.${i}.plugin">${P.plugins.map(x => `<option value="${x.id}"${x.id === it.plugin ? " selected" : ""}>${E(x.name)}</option>`).join("")}</select>
            <span class="pct">占 ${num(`slot.${s}.${i}.share`, it.share, { min: 0, max: 100 })}%</span>
            <button class="x" data-del="${s}.${i}" title="去掉这个策略">✕</button></div>
          <p class="hint">${E(p.desc)}${odd ? `　<em>（原本设计放在${p.slots.map(x => SLOTN[x]).join("/")}，放这里也可以）</em>` : ""}</p>
          <div class="kv-row"><label class="fld"><span>持股数</span>${num(`slot.${s}.${i}.params.hold`, it.params.hold, { min: p.params.find(q => q.key === "hold").min, max: p.params.find(q => q.key === "hold").max })}</label>
            ${inBoth ? `<label class="fld"><span>上涨期（两样都满足）时持几只</span><input type="number" data-k="slot.${s}.${i}.hold_both" value="${it.hold_both ?? ""}" placeholder="同上" min="1" max="50"></label>` : ""}</div>
          <details><summary>自带规则和其他参数</summary>
            <dl class="rules"><dt>风控</dt><dd>${E(p.rules.risk)}</dd><dt>轮换</dt><dd>${E(p.rules.rotate)}</dd><dt>仓位</dt><dd>${E(p.rules.size)}</dd>
              <dt>上场</dt><dd>${p.reselect_on_enter ? "分到钱当天重新选股再买" : "照它模拟账户当时的持仓买入"}；分不到钱当天卖出</dd></dl>
            <div class="params">${p.params.filter(q => q.key !== "hold").map(q => typeof q.default === "boolean"
              ? `<label class="fld chk"><input type="checkbox" data-k="slot.${s}.${i}.params.${q.key}"${it.params[q.key] ? " checked" : ""}><span>${E(q.label)}</span></label>`
              : `<label class="fld"><span>${E(q.label)}${q.max <= 1 ? "（%）" : ""}</span>${num(`slot.${s}.${i}.params.${q.key}` + (q.max <= 1 ? "_pct" : ""), q.max <= 1 ? +(it.params[q.key] * 100).toFixed(2) : it.params[q.key], { min: q.max <= 1 ? q.min * 100 : q.min, max: q.max <= 1 ? q.max * 100 : q.max, step: q.max <= 1 ? 0.5 : 1 })}</label>`).join("")}
              <button class="btn ghost sm" data-reset="${s}.${i}">参数恢复默认</button></div>
          </details></div>`; }).join("")}
      <button class="btn add" data-add="${s}">＋ 添加策略</button></div>`;
  }).join("");
}
function renderChecks() {
  const c = checks();
  $("checks").innerHTML = c.map(([ok, t]) => `<li class="${ok ? "ok" : "no"}"><span>${ok ? "✓" : "✗"}</span>${E(t)}</li>`).join("");
  const bad = c.some(x => !x[0]);
  $("btnSave").disabled = bad; $("btnSaveAs").disabled = bad;
  const r = M.rule, lines = [];
  if (r.type === "fixed") lines.push("不切换，一直按下面「平常」的分法。");
  else {
    lines.push(`每天下午看 ${P.index_names[r.small.index]}：最近 ${r.small.er_n} 天净涨幅至少占来回波动的 ${Math.round(r.small.er_min * 100)}%、并且在 ${r.small.ma_n} 天均价上方，就算「中小盘走强」。`);
    if (r.type === "three_state") lines.push(`同时 ${P.index_names[r.big.index]} 在 ${r.big.ma_n} 天均价上方、均价在上升、${Math.round(r.big.breadth_min * 100)}% 以上行业站上 ${r.big.breadth_ma} 天均价，就算「大盘上涨期」。`);
    if (+r.confirm_days > 0) lines.push(`新行情要连续 ${r.confirm_days} 天成立才切换。`);
  }
  for (const st of activeStates()) {
    const parts = [];
    for (const [s] of SLOTS) { const v = +M.slotAlloc[st][s]; if (!(v > 0)) continue;
      for (const it of M.slots[s]) if (it.share > 0) parts.push(`${PL[it.plugin].name} ${+(v * it.share / 100).toFixed(1)}%（${holdOf(it, st)} 只）`); }
    const tot = SLOTS.reduce((a, [s]) => a + (+M.slotAlloc[st][s] || 0), 0);
    if (tot < 100) parts.push(`货币基金 ${100 - tot}%`);
    lines.push(`<b>${STATES.find(x => x[0] === st)[1]}</b>：${parts.join(" ＋ ") || "全部买货币基金"}。`);
  }
  lines.push("每个策略按自己的规则在模拟账户里买卖；行情切换那天才重新分钱。两个策略选中同一只股票时合并成一笔，各自记成本。");
  $("summary").innerHTML = lines.map(l => `<p>${l}</p>`).join("");
}
function renderAll() { renderRule(); renderAlloc(); renderSlots(); renderChecks(); renderStats(); $("fMax").value = M.max_hold; }

/* ---------------- 交互 ---------------- */
function setPath(k, raw, el) {
  const v = el && el.type === "checkbox" ? el.checked : el && el.tagName === "SELECT" ? raw : raw === "" ? "" : +raw;
  const a = k.split(".");
  if (a[0] === "rule") {
    let o = M.rule; for (let i = 1; i < a.length - 1; i++) o = o[a[i]];
    let key = a[a.length - 1];
    if (key.endsWith("_pct")) { key = key.slice(0, -4); o[key] = v / 100; } else o[key] = v;
  } else if (a[0] === "alloc") M.slotAlloc[a[1]][a[2]] = v === "" ? 0 : v;
  else if (a[0] === "slot") {
    const it = M.slots[a[1]][+a[2]];
    if (a[3] === "plugin") { it.plugin = v; it.params = defaults(v); it.hold_both = null; return "slots"; }
    if (a[3] === "share") it.share = v;
    else if (a[3] === "hold_both") it.hold_both = v === "" ? null : v;
    else if (a[3] === "params") { let key = a[4]; if (key.endsWith("_pct")) { key = key.slice(0, -4); it.params[key] = v / 100; } else it.params[key] = v; }
  }
  return a[0];
}
document.addEventListener("input", e => {
  const k = e.target.dataset && e.target.dataset.k; if (!k) return;
  const what = setPath(k, e.target.value, e.target);
  if (what === "slots") renderSlots();
  if (what === "alloc") { renderSlots(); const tds = e.target.closest("tr"); if (tds) { const st = k.split(".")[1]; const tot = SLOTS.reduce((a, [s]) => a + (+M.slotAlloc[st][s] || 0), 0); const last = tds.lastElementChild; last.textContent = (100 - tot) + "%"; last.classList.toggle("bad", tot > 100); } }
  if (what === "rule") renderStats();
  renderChecks(); dirty(true);
});
document.addEventListener("change", e => { if (e.target.tagName === "SELECT" && e.target.dataset.k) { if (e.target.dataset.k.startsWith("rule")) renderStats(); } });
document.addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  if (b.dataset.type) { M.rule.type = b.dataset.type; renderAll(); dirty(true); }
  else if (b.dataset.add) { const s = b.dataset.add; const pid = P.plugins.find(x => x.slots.includes(s))?.id || P.plugins[0].id; M.slots[s].push({ plugin: pid, share: M.slots[s].length ? 0 : 100, params: defaults(pid), hold_both: null }); renderSlots(); renderChecks(); dirty(true); }
  else if (b.dataset.del) { const [s, i] = b.dataset.del.split("."); M.slots[s].splice(+i, 1); renderSlots(); renderChecks(); dirty(true); }
  else if (b.dataset.reset) { const [s, i] = b.dataset.reset.split("."); const it = M.slots[s][+i]; it.params = defaults(it.plugin); renderSlots(); renderChecks(); dirty(true); }
});
$("btnRuleReset").onclick = () => { M.rule = clone(P.rule_default); renderAll(); dirty(true); };
$("fMax").oninput = e => { M.max_hold = +e.target.value || 10; renderChecks(); dirty(true); };
$("btnNew").onclick = () => { if (!confirmLeave()) return; openModel(defaultModel(), { id: null, name: "", note: "" }); scrollTo({ top: $("editing").offsetTop - 70, behavior: "smooth" }); };
let DIRTY = false;
function dirty(v) { DIRTY = v; renderEditing(); }
function confirmLeave() { return !DIRTY || confirm("当前修改还没保存，放弃吗？"); }
window.addEventListener("beforeunload", e => { if (DIRTY) { e.preventDefault(); e.returnValue = ""; } });
function renderEditing() {
  $("editing").innerHTML = `<div class="panel edbar"><span>正在编辑：<b>${E(CUR.id ? CUR.name : "新组合（未保存）")}</b>${DIRTY ? ' <span class="chip warn">有未保存的修改</span>' : ""}</span>${CUR.id ? `<a href="${SIG}" target="_blank" rel="noopener">在信号监控里看 ↗</a>` : ""}</div>`;
}
function openModel(model, meta) {
  M = model; CUR = meta; $("fName").value = meta.name || ""; $("fNote").value = meta.note || ""; DIRTY = false; renderAll(); renderEditing();
}

async function save(asNew) {
  const name = $("fName").value.trim(); if (!name) { toast("先给组合起个名字"); $("fName").focus(); return; }
  const pass = await askPass(); if (!pass) return;
  $("saveMsg").textContent = "保存中…";
  try {
    const cfg = toCfg(); cfg.name = name;
    const id = await rpc("save_combo", { p_pass: pass, p_id: asNew ? null : CUR.id, p_name: name, p_config: cfg, p_note: $("fNote").value.trim() });
    CUR = { id, name, note: $("fNote").value.trim() }; DIRTY = false; renderEditing();
    $("saveMsg").textContent = "已保存，见页面最上方「已保存的组合」；收益率和操作提示在首页「我的组合策略」。本地电脑当晚跑 2023 年以来的回测，第二天 14:30 起出现在信号监控里。"; toast("已保存");
    await loadList();
    const card = document.querySelector(".mc.cur"); if (card) { scrollTo({ top: $("mine").offsetTop - 70, behavior: "smooth" }); card.classList.add("flash"); setTimeout(() => card.classList.remove("flash"), 2400); }
  } catch (e) {
    if (/口令/.test(e.message)) { PASS = null; try { sessionStorage.removeItem("combo-pass"); } catch (x) {} }
    $("saveMsg").textContent = "保存失败：" + e.message; toast("保存失败：" + e.message);
  }
}
$("btnSave").onclick = () => save(false);
$("btnSaveAs").onclick = () => save(true);

/* ---------------- 聚宽代码 ---------------- */
let TPL = null;
function cfgLines(cfg) {
  const r = cfg.rule, out = [], idx = c => P.index_names[c] || c;
  if (r.type === "fixed") out.push("切换规则：不切换，一直按「平常」的分法。");
  else {
    out.push(`中小盘走强：${idx(r.small.index)} 最近 ${r.small.er_n} 天净涨幅占来回波动 > ${Math.round(r.small.er_min * 100)}%，且在 ${r.small.ma_n} 天均价上方（14:50 现价）。`);
    if (r.type === "three_state") out.push(`大盘上涨期：${idx(r.big.index)} 在 ${r.big.ma_n} 天均价上方、均价比 ${r.big.slope_lag} 天前高、${Math.round(r.big.breadth_min * 100)}% 以上申万一级行业站上 ${r.big.breadth_ma} 天均价（前一日收盘）。`);
    if (+r.confirm_days > 0) out.push(`新行情要连续 ${r.confirm_days} 天成立才切换。`);
  }
  const SN = { normal: "平常", small: "中小盘走强", both: "中小盘走强 + 大盘上涨期" };
  const sts = r.type === "fixed" ? ["normal"] : r.type === "two_state" ? ["normal", "small"] : ["normal", "small", "both"];
  for (const st of sts) {
    const row = Object.entries(cfg.alloc[st] || {}).filter(([, v]) => v > 0);
    const tot = row.reduce((a, [, v]) => a + v, 0);
    const parts = row.map(([k, v]) => { const sl = cfg.sleeves[k]; const pm = Object.entries(sl.params || {}).map(([a, b]) => `${a}=${b}`).join(", ");
      return `${sl.name || PL[sl.plugin].name} ${Math.round(v * 100)}%（${pm}）`; });
    if (tot < 0.9999) parts.push(`货币 ETF ${Math.round((1 - tot) * 100)}%`);
    out.push(`「${SN[st]}」：${parts.join(" + ")}`);
  }
  out.push(`总持股上限 ${cfg.max_hold} 只；真实账户偏离目标 ${Math.round((cfg.tol ?? 0.25) * 100)}% 才调整。`);
  return out;
}
async function genCode(cfg, name) {
  if (!TPL) TPL = await fetch("combo/jq_template.py?v=" + (document.querySelector("meta[name=build]")?.content || Date.now())).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); });
  const c = clone(cfg); delete c.ui; c.name = name;
  const today = new Date().toISOString().slice(0, 10);
  const lines = cfgLines(c);
  const head = [`可配置组合「${name}」—— 聚宽回测代码（${today} 由量化策略库「组合配置」页生成）`, "", ...lines.map(l => "    " + l)].join("\n").replace(/\\/g, "\\\\").replace(/"""/g, '\\"\\"\\"');
  const code = TPL.replace("__HEADER__", head).replace("__CONFIG_JSON__", JSON.stringify(JSON.stringify(c)));
  return { code, lines };
}
async function showCode(cfg, name) {
  let r;
  try { r = await genCode(cfg, name); } catch (e) { toast("生成失败：" + e.message); return; }
  $("codeName").textContent = name; $("codePre").textContent = r.code;
  $("codeMeta").innerHTML = r.lines.map(l => `<li>${E(l)}</li>`).join("");
  const fn = "可配置组合_" + name.replace(/[\/:*?"<>|\s]+/g, "_") + "_聚宽.py";
  $("codeDl").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([r.code], { type: "text/x-python;charset=utf-8" })); a.download = fn; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); };
  $("codeCopy").onclick = async () => { try { await navigator.clipboard.writeText(r.code); toast("已复制，贴到聚宽新建策略里"); } catch (e) { const t = document.createRange(); t.selectNodeContents($("codePre")); const s = getSelection(); s.removeAllRanges(); s.addRange(t); toast("请按 Ctrl+C 复制已选中的代码"); } };
  $("codeDlg").showModal();
}
$("btnJq").onclick = () => {
  if (checks().some(x => !x[0])) { toast("先把「检查」里打 ✗ 的项改好"); return; }
  const name = $("fName").value.trim() || (CUR.id ? CUR.name : "未命名组合");
  showCode(toCfg(), name);
};

/* ---------------- 我的组合 ---------------- */
async function loadList() {
  try {
    LIST = await sb("combos?select=id,name,note,config,jq,created_at,updated_at&order=updated_at.desc");
    $("mineList").innerHTML = LIST.length ? LIST.map(c => {
      return `<div class="mc ${c.id === CUR.id ? "cur" : ""}"><div class="mc-t"><b>${E(c.name)}</b><span class="hint">${E(c.updated_at.slice(0, 10))}</span></div>
        ${c.note ? `<p class="hint">${E(c.note)}</p>` : ""}
        <div class="mc-a"><button class="btn sm" data-open="${c.id}">打开编辑</button><button class="btn sm" data-copy="${c.id}">复制一份</button><button class="btn sm" data-code="${c.id}">聚宽代码</button><button class="btn sm" data-jq="${c.id}">填聚宽成绩</button><a class="btn sm ghost" href="combo-view.html?id=${c.id}">看成绩</a><button class="btn sm ghost danger" data-rm="${c.id}">删除</button></div></div>`;
    }).join("") : `<p class="empty">还没有保存的组合。下面默认打开的是「原方案（V1.4 三合一）」，改完起个名字保存即可。保存后的组合和收益率显示在首页「我的组合策略」。</p>`;
  } catch (e) { $("mineList").innerHTML = `<p class="empty">读取失败：${E(e.message)}　<button class="btn sm" onclick="location.reload()">重试</button></p>`; }
}
$("mineList").addEventListener("click", async e => {
  const b = e.target.closest("button"); if (!b) return;
  const id = b.dataset.open || b.dataset.copy || b.dataset.jq || b.dataset.rm || b.dataset.code; const c = LIST.find(x => x.id === id); if (!c) return;
  if (b.dataset.code) { showCode(c.config, c.name); return; }
  if (b.dataset.open || b.dataset.copy) {
    if (!confirmLeave()) return;
    const model = c.config.ui ? clone(c.config.ui) : defaultModel();
    openModel(model, b.dataset.open ? { id: c.id, name: c.name, note: c.note } : { id: null, name: c.name + "（副本）", note: c.note });
    if (b.dataset.copy) dirty(true);
    loadList(); scrollTo({ top: $("editing").offsetTop - 70, behavior: "smooth" });
  } else if (b.dataset.rm) {
    if (!confirm(`删除「${c.name}」？信号监控里也不再显示它。`)) return;
    const pass = await askPass(); if (!pass) return;
    try { await rpc("delete_combo", { p_pass: pass, p_id: id }); toast("已删除"); if (CUR.id === id) CUR.id = null; loadList(); renderEditing(); } catch (x) { toast("删除失败：" + x.message); }
  } else if (b.dataset.jq) {
    const d = $("jqDlg"), j = c.jq || {};
    $("jqAnn").value = j.ann ?? ""; $("jqCum").value = j.cum ?? ""; $("jqDd").value = j.dd ?? ""; $("jqSh").value = j.sharpe ?? ""; $("jqRange").value = j.range ?? "";
    d.onclose = async () => {
      if (d.returnValue !== "ok") return;
      const pass = await askPass(); if (!pass) return;
      const v = x => x === "" ? null : +x;
      try { await rpc("set_jq", { p_pass: pass, p_id: id, p_jq: { ann: v($("jqAnn").value), cum: v($("jqCum").value), dd: v($("jqDd").value), sharpe: v($("jqSh").value), range: $("jqRange").value.trim() } }); toast("已保存聚宽成绩"); loadList(); }
      catch (x) { toast("保存失败：" + x.message); }
    };
    d.showModal();
  }
});

/* ---------------- 启动 ---------------- */
(async () => {
  try {
    const v = "?v=" + (document.querySelector("meta[name=build]")?.content || Date.now());
    [P, MK] = await Promise.all([fetch(SIG + "data/meta.json?t=" + Date.now(), { cache: "no-store" }).then(r => r.json()), fetch("combo/data/mkt.json" + v).then(r => r.json())]);
  } catch (e) { document.querySelector(".wrap").insertAdjacentHTML("afterbegin", `<p class="err">页面数据加载失败：${E(e.message)}</p>`); return; }
  for (const p of P.plugins) PL[p.id] = p;
  STATES = Object.entries(P.states);
  SLOTS = P.slots.map(x => [x.id, x.name, x.desc]); SLOTN = Object.fromEntries(SLOTS.map(x => [x[0], x[1]]));
  openModel(defaultModel(), { id: null, name: "", note: "" });
  await loadList();
  const want = new URLSearchParams(location.search).get("id");
  const hit = want && LIST.find(x => x.id === want);
  if (hit) { openModel(hit.config.ui ? clone(hit.config.ui) : defaultModel(), { id: hit.id, name: hit.name, note: hit.note }); loadList(); }
})();
})();
