/* 「我的组合策略」：首页卡片 + 组合详情页（combo-view.html?id=）。只展示，不编辑 */
(() => {
const SB = { url: "https://vsojozoqbyvxszgfxaop.supabase.co", key: "sb_publishable_ykBgwF4pxNq7hSnuYASSqw_AXEIF346" };
const SIG = "https://wanan99999.github.io/signal-monitor/";
const IDX = { sh000852: "中证1000", sh000905: "中证500", sz399006: "创业板指", sz399303: "国证2000", sh000300: "沪深300", sh000016: "上证50", sh000001: "上证指数" };
const ST = [["normal", "平常"], ["small", "中小盘走强"], ["both", "中小盘走强 + 大盘上涨期"]];
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
    return { st, label: ST.find(x => x[0] === st)[1], parts };
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
  if (r && r.ann != null) return { v: (r.ann > 0 ? "+" : "") + (r.ann * 100).toFixed(1) + "%", tag: "本地测算" };
  return { v: "待测算", tag: "新组合 · 待测算" };
}
const ACT = { "买入": "buy", "卖出": "sell", "调仓": "mix" };

/* 首页卡片 */
async function home(el) {
  try {
    const [list, res, sig] = await Promise.all([sb("combos?select=id,name,note,config,jq,updated_at&order=created_at.asc"), getJ(SIG + "data/combos/results.json"), getJ(SIG + "data/signals.json")]);
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
    const [list, res, sig] = await Promise.all([sb("combos?select=id,name,note,config,jq,created_at,updated_at&id=eq." + id), getJ(SIG + "data/combos/results.json"), getJ(SIG + "data/signals.json")]);
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
      <section class="blk"><h2>什么时候切换</h2><ul class="bul">${ruleText(cfg).map(t => `<li>${E(t)}</li>`).join("")}</ul></section>
      <section class="blk"><h2>每种行情下钱怎么分</h2>
        <div class="tblwrap"><table class="ctbl"><thead><tr><th>行情</th><th>交给哪些策略</th></tr></thead><tbody>
        ${rows(cfg).map(rw => `<tr><td>${E(rw.label)}</td><td>${rw.parts.map(q => `${E(q.name)} <b>${Math.round(q.v * 100)}%</b>${q.hold ? `（${q.hold} 只）` : ""}`).join("<br>")}</td></tr>`).join("")}
        </tbody></table></div></section>
      <section class="blk"><h2>说明</h2><ul class="bul">
        <li>每个策略在自己的模拟账户里按自己的规则买卖（风控、轮换、仓位都不变）；行情切换那天才按上表重新分钱。</li>
        <li>两个策略选中同一只股票时合并成一笔，各自记成本。</li>
        <li>最多同时持有 ${cfg.max_hold} 只。创建于 ${E(c.created_at.slice(0, 10))}，最后修改 ${E(c.updated_at.slice(0, 10))}。</li></ul></section>`;
  } catch (e) { el.innerHTML = `<p class="empty">读取失败：${E(e.message)}，请稍后刷新。</p>`; }
}
const h = document.getElementById("mineCards"); if (h) home(h);
const v = document.getElementById("mineView"); if (v) view(v);
})();
