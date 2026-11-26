# -*- coding: utf-8 -*-
"""
__HEADER__

—— 以下是「可配置组合」通用框架（量化策略库「组合配置」页生成，和本地信号监控的 combo2 引擎同一套规则）——
每个策略在自己的「影子账户」（100 万模拟钱）里按自己的规则买卖：选股、轮换、止损、仓位控制互不干扰；
组合层按「切换规则」判断今天是哪种行情，只在行情切换那天按比例重新分钱：
    放大倍数 k = 真实账户总资产 × 99% × 分到的比例 ÷ 影子账户总资产
    真实目标持仓 = Σ 各影子账户每只持仓市值 × k（两个策略选中同一只就合并）
真实账户和目标偏离超过 TOL 才下单；没分出去的钱买货币 ETF。
每天 14:50 交易（分钟回测）。建议回测 2023-01-01 起、初始资金 ≥ 100 万。
影子账户按 14:50 价格成交、单边按 0.15% 扣费，净值用前复权价逐日结算（含分红送转）。
"""
import builtins
import json
from jqdata import *
import numpy as np
import pandas as pd
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

CONFIG = json.loads(__CONFIG_JSON__)
VERSION = "可配置组合 · %s" % CONFIG.get("name", "")

TRADE_TIME = "14:50"
CASH_ETF = "511880.XSHG"         # 货币 ETF（银华日利）
CASH_BUFFER = 0.01
COST = 0.0015                    # 影子账户单边成本
STATES = {"normal": "平常", "small": "中小盘走强", "both": "中小盘走强 + 大盘上涨期"}


def jq_code(c):
    """sh600000 → 600000.XSHG；sz159915 → 159915.XSHE；已是聚宽代码的原样返回"""
    if "." in c:
        return c
    return c[2:] + (".XSHG" if c.startswith("sh") else ".XSHE")


# ======================================================================================
# 通用工具
# ======================================================================================
def can_buy(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price is not None and d.last_price > 0 and d.last_price < d.high_limit - 1e-6


def can_sell(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price is not None and d.last_price > d.low_limit + 1e-6


def er_ma_ok(s, er_n=20, er_min=0.20, ma_n=20):
    s = np.asarray(s, dtype=float)
    if len(s) < max(er_n + 1, ma_n) or np.isnan(s[-(er_n + 1):]).any():
        return False
    path = np.abs(np.diff(s[-(er_n + 1):])).sum()
    er = (s[-1] - s[-(er_n + 1)]) / path if path > 0 else 0.0
    return er > er_min and s[-1] > s[-ma_n:].mean()


def base_stocks(context, cd, list_days):
    day = context.previous_date
    secs = get_all_securities(["stock"], date=day)
    secs = secs[secs.start_date <= day - pd.Timedelta(days=list_days)]
    ss = [s for s in secs.index if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
    return [s for s in ss if (not cd[s].is_st) and (not cd[s].paused) and "退" not in cd[s].name]


def pivot_close(codes, day, count, fq="pre", field="close"):
    if not codes:
        return pd.DataFrame()
    p = get_price(list(codes), end_date=day, count=count, fields=[field], fq=fq, panel=False)
    if p is None or len(p) == 0:
        return pd.DataFrame()
    return p.pivot(index="time", columns="code", values=field)


def daycache(key, fn):
    """同一天里多个子账户共用的计算（选股排名、行业指数等）只算一次"""
    c = g.cache
    if c.get("_day") != g.today:
        c.clear(); c["_day"] = g.today
    if key not in c:
        c[key] = fn()
    return c[key]


# ======================================================================================
# 影子账户：按市值记账，14:50 成交
# ======================================================================================
class Shadow:
    def __init__(self, cash=1e6):
        self.cash = cash
        self.v = {}          # 代码 → 持仓市值
        self.cost = {}       # 代码 → 投入成本（加仓累加，减仓按比例减）
        self.today = []      # 今天的成交 [(代码, 方向, 原因)]

    def value(self):
        return self.cash + sum(self.v.values())

    def held(self):
        return [s for s, x in self.v.items() if x > 0]

    def pos(self, s):
        return self.v.get(s, 0.0)

    def mark(self, ratio):
        for s in list(self.v):
            r = ratio.get(s)
            if r is not None and np.isfinite(r) and r > 0:
                self.v[s] *= r

    def order(self, cd, s, target, tag=""):
        """把 s 调到目标市值；涨停买不进、跌停卖不出、停牌返回 False"""
        cur = self.v.get(s, 0.0)
        delta = max(target, 0.0) - cur
        if abs(delta) < 1:
            return True
        d = cd[s]
        if d.paused or d.last_price is None or d.last_price <= 0:
            return False
        if delta > 0:
            if d.last_price >= d.high_limit - 1e-6:
                return False
            delta = min(delta, self.cash / (1 + COST))
            if delta <= 1:
                return False
        elif d.last_price <= d.low_limit + 1e-6:
            return False
        self.cash -= delta + abs(delta) * COST
        nv = cur + delta
        if nv <= 1:
            self.v.pop(s, None); self.cost.pop(s, None)
            self.today.append((s, "卖出", tag))
        else:
            self.cost[s] = self.cost.get(s, 0.0) + delta if delta > 0 else self.cost.get(s, 0.0) * nv / cur
            self.v[s] = nv
            self.today.append((s, "买入" if cur <= 0 else ("加仓" if delta > 0 else "减仓"), tag))
        return True


# ======================================================================================
# 切换规则
# ======================================================================================
RULE_DEFAULT = dict(type="three_state",
                    small=dict(index="sh000852", er_n=20, er_min=0.2, ma_n=20),
                    big=dict(index="sh000300", ma_n=120, slope_lag=20, breadth_min=0.5, breadth_ma=60),
                    confirm_days=0)


def sw1_close(day, cal_days):
    codes = list(get_industries("sw_l1", date=day).index)
    start = day - pd.Timedelta(days=cal_days)
    out = []
    for i in range(0, len(codes), 25):
        q = query(finance.SW1_DAILY_PRICE.date, finance.SW1_DAILY_PRICE.code, finance.SW1_DAILY_PRICE.close
                  ).filter(finance.SW1_DAILY_PRICE.code.in_(codes[i:i + 25]),
                           finance.SW1_DAILY_PRICE.date >= start, finance.SW1_DAILY_PRICE.date <= day)
        out.append(finance.run_query(q))
    df = pd.concat(out) if out else pd.DataFrame()
    if len(df) == 0:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="code", values="close").sort_index()


class Rule:
    def __init__(self, cfg):
        c = json.loads(json.dumps(RULE_DEFAULT))
        for k, v in (cfg or {}).items():
            if isinstance(v, dict):
                c[k].update(v)
            else:
                c[k] = v
        self.c = c
        self.state, self.pending, self.cnt = None, None, 0
        self.g1 = self.up = False

    def raw(self, context):
        c = self.c
        if c["type"] == "fixed":
            return "normal"
        s = c["small"]; idx = jq_code(s["index"])
        n = max(s["er_n"] + 1, s["ma_n"])
        h = attribute_history(idx, n, "1d", ["close"], skip_paused=True, df=False)["close"]
        b = get_price(idx, end_date=context.current_dt, frequency="1m", fields=["close"], count=1)
        g1 = len(h) >= n and len(b) > 0 and er_ma_ok(np.append(h, float(b["close"].iloc[-1])), s["er_n"], s["er_min"], s["ma_n"])
        self.g1 = bool(g1)
        if c["type"] == "two_state":
            return "small" if g1 else "normal"
        up = False
        if g1:                                   # 大盘上涨期只在中小盘走强时才有用，省一次查询
            bg = c["big"]
            h = attribute_history(jq_code(bg["index"]), bg["ma_n"] + bg["slope_lag"], "1d", ["close"], skip_paused=True, df=False)["close"]
            if len(h) >= bg["ma_n"] + bg["slope_lag"]:
                ma_now = h[-bg["ma_n"]:].mean(); ma_old = h[-(bg["ma_n"] + bg["slope_lag"]):-bg["slope_lag"]].mean()
                if h[-1] > ma_now and ma_now > ma_old:
                    px = sw1_close(context.previous_date, int(bg["breadth_ma"] * 2.2) + 10)
                    px = px.loc[:, px.notna().sum() >= bg["breadth_ma"]] if len(px) else px
                    if px.shape[1] > 0:
                        up = (px.iloc[-1] > px.iloc[-bg["breadth_ma"]:].mean()).mean() >= bg["breadth_min"]
        self.up = bool(up)
        return "both" if (g1 and up) else ("small" if g1 else "normal")

    def step(self, context):
        r = self.raw(context)
        n = int(self.c.get("confirm_days") or 0)
        if self.state is None or n <= 0:
            self.state = r
            return r
        if r == self.state:
            self.pending, self.cnt = None, 0
        elif r == self.pending:
            self.cnt += 1
        else:
            self.pending, self.cnt = r, 1
        if self.pending is not None and self.cnt >= n:
            self.state, self.pending, self.cnt = r, None, 0
        return self.state


# ======================================================================================
# 插件（每个都在自己的影子账户里按自己的规则运行）
# ======================================================================================
class Plugin:
    DEFAULTS = {}
    reselect_on_enter = False      # True：没分到钱时不运行，分到钱那天重新选
    always_run = True

    def __init__(self, sid, spec):
        self.sid = sid
        self.name = spec.get("name") or sid
        self.p = dict(self.DEFAULTS)
        self.p.update({k: v for k, v in (spec.get("params") or {}).items() if v is not None})
        self.b = Shadow()

    def step(self, context, cd, force=False):
        raise NotImplementedError

    def after_close(self, context):
        pass


# ---------------- 大盘股熊市震荡市稳健 ----------------
def steady_rank(context, cd, pool_n=800, roe_min=0.05):
    def f():
        day = context.previous_date
        stocks = base_stocks(context, cd, 365)
        q = query(valuation.code, valuation.circulating_market_cap, indicator.roe, income.net_profit).filter(valuation.code.in_(stocks))
        df = get_fundamentals(q, date=day).dropna(subset=["circulating_market_cap"])
        df = df.sort_values("circulating_market_cap", ascending=False).head(pool_n)
        pool = list(df[(df.net_profit > 0) & (df.roe * 4 / 100 > roe_min)].code)
        if not pool:
            return 0, []
        px = pivot_close(pool, day, 249)
        if len(px) == 0:
            return 0, []
        cur = pd.Series({s: cd[s].last_price for s in px.columns})
        px = pd.concat([px, cur.to_frame().T]); px = px.loc[:, px.notna().sum() >= 200]
        sc = ((px.iloc[-1] / px.max()).rank(pct=True) + (-px.pct_change().iloc[-60:].std()).rank(pct=True)).dropna()
        return len(pool), list(sc.sort_values(ascending=False).index)
    return daycache(("steady", pool_n, roe_min), f)


class SteadyP(Plugin):
    DEFAULTS = dict(hold=10, buffer=20, ind_max=3, every=5, stop=0.10, vol_target=0.15, dd_half=0.15, dd_out=0.25)

    def __init__(self, sid, spec):
        Plugin.__init__(self, sid, spec)
        self.picks, self.sleeve, self.g_expo = [], [], 0.0

    def exposure(self):
        q = self.p
        r = np.array(self.sleeve[-40:])
        e = 1.0
        if len(r) >= 20:
            vol = np.std(r, ddof=1) * np.sqrt(252)
            e = 1.0 if vol <= 0 else min(1.0, q["vol_target"] / vol)
        if self.sleeve:
            nav = np.cumprod(1 + np.array(self.sleeve)); dd = nav[-1] / np.max(nav[-250:]) - 1
            if dd < -q["dd_out"]:
                e = 0.0
            elif dd < -q["dd_half"]:
                e *= 0.5
        return e

    def select(self, context, cd):
        q = self.p
        n_pool, codes = steady_rank(context, cd)
        if n_pool < q["hold"]:
            return self.picks
        rank = {s: i for i, s in enumerate(codes)}
        held = self.b.held()
        ind = get_industry(list(set(codes[:q["buffer"] + 60] + held)), date=context.previous_date)
        def ind_of(s): return ind.get(s, {}).get("sw_l2", {}).get("industry_name", "未知")
        cnt, pick = {}, []
        def add(s): pick.append(s); cnt[ind_of(s)] = cnt.get(ind_of(s), 0) + 1
        for s in sorted(held, key=lambda x: rank.get(x, 10 ** 9)):
            if rank.get(s, 10 ** 9) < q["buffer"] and len(pick) < q["hold"] and cnt.get(ind_of(s), 0) < q["ind_max"]:
                add(s)
        for s in codes:
            if len(pick) >= q["hold"]:
                break
            if s not in pick and cnt.get(ind_of(s), 0) < q["ind_max"]:
                add(s)
        return pick

    def _to(self, cd, picks, expo):
        b, q = self.b, self.p
        tv = b.value() * 0.99 * expo / q["hold"]
        for s in b.held():
            if s not in picks:
                b.order(cd, s, 0, "跌出前 %d 名" % q["buffer"])
            elif b.pos(s) > tv * 1.05:
                b.order(cd, s, tv, "仓位调低")
        for s in picks:
            if b.pos(s) < tv * 0.95:
                b.order(cd, s, tv, "新选入" if b.pos(s) == 0 else "仓位调高")

    def step(self, context, cd, force=False):
        b, q = self.b, self.p
        for s in b.held():
            if b.cost.get(s) and b.pos(s) / b.cost[s] - 1 < -q["stop"] and can_sell(cd, s):
                b.order(cd, s, 0, "亏损超过 %.0f%% 止损" % (q["stop"] * 100))
                if s in self.picks:
                    self.picks.remove(s)
        expo = self.exposure()
        if g.day % q["every"] == 0 or force:
            self.picks = self.select(context, cd)
            self._to(cd, self.picks if expo > 0 else [], expo); self.g_expo = expo
        elif abs(expo - self.g_expo) >= 0.10 or (expo == 0 and self.g_expo > 0):
            self._to(cd, [s for s in self.picks if s in b.v] if expo > 0 else [], expo); self.g_expo = expo

    def after_close(self, context):
        """影子净值（波动率控仓、回撤风控用）：选中股票当日等权收益"""
        r = g.day_ret
        x = [r[s] for s in self.picks if s in r]
        self.sleeve.append(float(np.mean(x)) if x else 0.0)
        self.sleeve = self.sleeve[-400:]


# ---------------- 银行股顺势 ----------------
class BankP(Plugin):
    DEFAULTS = dict(hold=4, keep_rank=6, chase=0.10, every=5, pool_n=20)

    def pool(self, context):
        y = context.current_dt.year; n = self.p["pool_n"]
        key = (y, n)
        if g.bank_pool.get("key") != key:
            d = get_trade_days(end_date="%d-12-31" % (y - 1), count=1)[0]
            banks = get_industry_stocks("801780", date=d)
            df = get_fundamentals(query(valuation.code, valuation.market_cap).filter(valuation.code.in_(banks)), date=d)
            g.bank_pool = dict(key=key, codes=list(df.sort_values("market_cap", ascending=False).head(n).code))
        return g.bank_pool["codes"]

    def signal(self, context, cd):
        pool = self.pool(context)
        def f():
            px = pivot_close(pool, context.previous_date, 249)
            rows = []
            for c in pool:
                if c not in px.columns:
                    continue
                s = px[c].values
                cur = cd[c].last_price
                if len(s) < 249 or np.isnan(s).any() or not cur or np.isnan(cur):
                    continue
                s = np.append(s, cur)
                rows.append(dict(code=c, dd=cur / s.max() - 1, ret=cur / s[-21] - 1))
            df = pd.DataFrame(rows, columns=["code", "dd", "ret"])
            return df.sort_values("dd", ascending=False).reset_index(drop=True)
        return daycache(("bank", tuple(pool)), f)

    def step(self, context, cd, force=False):
        b, q = self.b, self.p
        if not (g.day % q["every"] == 0 or force):
            return
        sig = self.signal(context, cd)
        rank = {c: k for k, c in enumerate(sig.code)}
        held = b.held()
        keep = [c for c in held if c in rank and rank[c] < q["keep_rank"]]
        for c in held:
            if c not in keep and not b.order(cd, c, 0, "不在股票池" if c not in rank else "排名掉到第 %d" % (rank[c] + 1)):
                keep.append(c)
        need, new = q["hold"] - len(keep), []
        for _, r in sig.iterrows():
            if need <= 0:
                break
            if r.code in keep or not can_buy(cd, r.code) or r.ret > q["chase"]:
                continue
            new.append(r.code); need -= 1
        tv = b.value() / q["hold"]
        for c in sorted(keep, key=lambda c: -b.pos(c)):
            b.order(cd, c, tv, "调回等权")
        for c in new:
            b.order(cd, c, min(tv, b.cash / (1 + COST)), "排名第 %d" % (rank[c] + 1))


# ---------------- 小市值反转（V1.4 个股块）----------------
def s_risk_drop(context, cd, stocks):
    """退市 / ST 风险过滤：股价<2 元、每股净资产≤0；1–4 月亏损；4/15–4/30 上一年年报未披露"""
    day = context.previous_date
    v = get_fundamentals(query(valuation.code, valuation.pe_ratio, valuation.pb_ratio).filter(valuation.code.in_(stocks)),
                         date=day).set_index("code")
    drop = set(s for s in stocks if cd[s].last_price is not None and cd[s].last_price < 2.0)
    drop |= set(v.index[v.pb_ratio <= 0])
    if day.month <= 4:
        drop |= set(v.index[v.pe_ratio < 0])
    if day.month == 4 and day.day >= 15:
        r = get_fundamentals(query(indicator.code, indicator.statDate).filter(indicator.code.in_(stocks)), date=day).set_index("code")
        done = set(r.index[pd.to_datetime(r.statDate) >= pd.Timestamp(day.year - 1, 12, 31)])
        drop |= set(stocks) - done
    return drop


def s_rank(context, cd, risk_filter):
    def f():
        day = context.previous_date
        stocks = base_stocks(context, cd, 365)
        amt = pivot_close(stocks, day, 20, fq=None, field="money").mean()
        amt = amt[amt > 0]; pool = list(amt[amt.rank(pct=True) > 0.30].index)
        if not pool:
            return []
        if risk_filter:
            bad = s_risk_drop(context, cd, pool)
            pool = [s for s in pool if s not in bad]
        cap = get_fundamentals(query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(pool)),
                               date=day).dropna().set_index("code")["circulating_market_cap"]
        cap = cap[cap > 0]
        px = pivot_close(list(cap.index), day, 20)
        if len(px) == 0 or len(cap) == 0:
            return []
        cur = pd.Series({s: cd[s].last_price for s in px.columns})
        ret = (cur / px.iloc[0] - 1).replace([np.inf, -np.inf], np.nan)
        df = pd.DataFrame({"cap": cap, "ret": ret}).dropna()
        sc = ((-np.log(df["cap"])).rank(pct=True) + (-df["ret"]).rank(pct=True)).sort_values(ascending=False)
        return list(sc.index)
    return daycache(("small", bool(risk_filter)), f)


class SmallCapP(Plugin):
    DEFAULTS = dict(hold=10, buffer=20, every=5, risk_filter=True)
    reselect_on_enter = True
    always_run = False

    def __init__(self, sid, spec):
        Plugin.__init__(self, sid, spec)
        self.picks = []

    def step(self, context, cd, force=False):
        b, q = self.b, self.p
        n = q["hold"]
        st_out = [s for s in self.picks if cd[s].is_st or "退" in cd[s].name]
        self.picks = [s for s in self.picks if s not in st_out]
        if g.day % q["every"] == 0 or force or len(self.picks) != n:
            codes = s_rank(context, cd, q["risk_filter"]); rk = {s: i for i, s in enumerate(codes)}
            pick = [s for s in sorted(self.picks, key=lambda x: rk.get(x, 10 ** 9)) if rk.get(s, 10 ** 9) < q["buffer"]][:n]
            for s in codes:
                if len(pick) >= n:
                    break
                if s not in pick:
                    pick.append(s)
            self.picks = pick
        tv = b.value() * 0.99 / n
        for s in b.held():
            if s not in self.picks:
                b.order(cd, s, 0, "被 ST，换掉" if s in st_out else "跌出前 %d 名" % q["buffer"])
            elif b.pos(s) > tv * 1.25:
                b.order(cd, s, tv, "仓位调低")
        for s in self.picks:
            if b.pos(s) < tv * 0.75:
                b.order(cd, s, tv, "新选入" if b.pos(s) == 0 else "仓位调高")


# ---------------- 行业 ETF（二级行业动量）----------------
ETF_EXCLUDE = ["300", "500", "1000", "2000", "50", "180", "100", "800", "200", "A股", "A50", "A100", "A500", "中证A", "创业", "科创50",
               "科创100", "科创200", "科创板", "科综", "双创", "深证", "上证", "深成", "综指", "中小", "国债", "债", "货币", "现金", "日利",
               "添益", "快线", "快钱", "财富宝", "添利", "天天", "日日", "短融", "黄金", "金ETF", "白银", "豆粕", "原油", "能源化工", "期货",
               "纳指", "纳斯达克", "标普", "日经", "日本", "德国", "法国", "恒生", "港股", "中概", "H股", "沪港深", "香港", "美国", "亚太",
               "东南亚", "沙特", "巴西", "MSCI", "道琼斯", "海外", "全球", "中韩", "AH", "等权", "央企", "国企", "民企", "ESG", "红利", "价值",
               "成长", "质量", "低波", "基本面", "动量", "现金流", "股息", "大盘", "中盘", "小盘", "微盘", "龙头", "核心", "宽基", "长三角",
               "大湾区", "粤港澳", "湾创", "湖北", "浙江", "浙商", "上海", "北京", "四川", "江苏", "福建", "山东", "安徽", "湖南", "广东",
               "京津冀", "成渝", "杭州湾", "张江", "G60", "长江保护", "中创400", "一带一路", "央视", "中关村", "新兴", "战略", "产业升级",
               "新经济", "高端装备", "科技100", "科技50", "TMT", "互联网+"]
L2_NEED = 185


def l2_close(context):
    """申万二级行业指数（截至前一交易日）：聚宽查不到二级行业行情，用成分股按流通市值加权的日收益合成。
    成分股每 20 个交易日更新；平时每天只追加一天，省查询。"""
    def f():
        day = context.previous_date
        L = g.l2
        if g.day - L.get("mem_day", -999) >= 20 or not L.get("mem"):
            inds = get_industries("sw_l2", date=day)
            inds = inds[pd.to_datetime(inds.start_date) <= pd.Timestamp(day)]
            L["names"] = dict(zip(inds.index, inds.name))
            L["mem"] = {c: [s for s in get_industry_stocks(c, date=day) if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
                        for c in inds.index}
            L["mem_day"] = g.day; L["px"] = None
        allst = sorted(set(s for v in L["mem"].values() for s in v))
        if not allst:
            return pd.DataFrame()
        last = L.get("px")
        full = last is None or len(last) == 0 or L.get("px_day") is None
        cnt = L2_NEED + 1 if full else 2
        if not full:
            gap = len(get_trade_days(start_date=L["px_day"], end_date=day)) - 1
            if gap <= 0:
                return last
            if gap > 5:
                full, cnt = True, L2_NEED + 1
            else:
                cnt = gap + 1
        p = pivot_close(allst, day, cnt)
        r = p.pct_change().iloc[1:]
        cap = get_fundamentals(query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(allst)),
                               date=day).dropna().set_index("code")["circulating_market_cap"]
        rows = {}
        for c, ss in L["mem"].items():
            ss = [s for s in ss if s in r.columns and s in cap.index]
            if len(ss) < 3:
                continue
            rr = r[ss]; ww = rr.notna().mul(cap[ss], axis=1)
            rows[c] = ((rr.fillna(0) * ww).sum(axis=1) / ww.sum(axis=1).replace(0, np.nan)).fillna(0)
        ret = pd.DataFrame(rows); ret.index = pd.to_datetime(ret.index)
        if full:
            px = (1 + ret).cumprod()
        else:
            base = last.iloc[-1].reindex(ret.columns).fillna(1.0)
            px = pd.concat([last, (1 + ret).cumprod().mul(base, axis=1)]).iloc[-(L2_NEED + 1):]
        L["px"], L["px_day"] = px, day
        return px
    return daycache(("l2",), f)


def etf_mapping(context, ind_px, corr_min):
    def f():
        day = context.previous_date
        df = get_all_securities(["etf"], date=day)
        df = df[df.start_date <= day - pd.Timedelta(days=120)]
        cand = [c for c, n in zip(df.index, df.display_name) if not any(k in n for k in ETF_EXCLUDE)]
        if not cand or len(ind_px) < 121:
            return {}
        m = pivot_close(cand, day, 20, fq=None, field="money").mean()
        etfs = list(m[m >= 3e7].index)
        if not etfs:
            return {}
        er = pivot_close(etfs, day, 121).pct_change().iloc[1:]
        er.index = pd.to_datetime(er.index)
        er = er.loc[:, er.notna().sum() >= 96]
        ir = ind_px.pct_change().iloc[-120:]
        ir, er = ir.align(er, join="inner", axis=0)
        if len(ir) < 96 or er.shape[1] == 0:
            return {}
        out = {}
        for c in ir.columns:
            cc = er.corrwith(ir[c]).dropna()
            if len(cc) and cc.max() >= corr_min:
                out[c] = (cc.idxmax(), float(cc.max()))
        log.info("行业→ETF 对应：%d 个二级行业找到相关度 ≥ %.2f 的 ETF" % (len(out), corr_min))
        return out
    return daycache(("map", corr_min), f)


def l2_mom(context):
    def f():
        px = l2_close(context)
        okset, mom = set(), {}
        for c in px.columns:
            s = px[c].dropna()
            if len(s) >= 61 and er_ma_ok(s.values):
                okset.add(c); mom[c] = s.iloc[-1] / s.iloc[-61] - 1
        return okset, mom
    return daycache(("l2mom",), f)


class SectorEtfP(Plugin):
    DEFAULTS = dict(hold=3, every=10, map_days=20, corr_min=0.85)
    reselect_on_enter = True
    always_run = False

    def __init__(self, sid, spec):
        Plugin.__init__(self, sid, spec)
        self.inds, self.etf, self.map, self.map_day = [], {}, {}, -999

    def step(self, context, cd, force=False):
        b, q = self.b, self.p
        okset, mom = l2_mom(context)
        weak = [c for c in self.inds if c not in okset]
        self.inds = [c for c in self.inds if c in okset]
        if g.day % q["every"] == 0 or force:
            if g.day - self.map_day >= q["map_days"] or not self.map:
                self.map = etf_mapping(context, l2_close(context), q["corr_min"]); self.map_day = g.day
            pick, used = [], set()
            for c in sorted(mom, key=lambda c: -mom[c]):
                if len(pick) >= q["hold"]:
                    break
                if c in self.map and self.map[c][0] not in used:
                    pick.append(c); used.add(self.map[c][0])
            self.inds = pick; self.etf = {c: self.map[c][0] for c in pick}
            nm = g.l2.get("names", {})
            log.info("%s：%s" % (self.name, "、".join("%s→%s" % (nm.get(c, c), get_security_info(self.etf[c]).display_name) for c in pick) or "无"))
        tgt = {}
        tv = b.value() * 0.99 / q["hold"]
        for c in self.inds:
            e = self.etf[c]; tgt[e] = tgt.get(e, 0) + tv
        for s in b.held():
            if s not in tgt:
                b.order(cd, s, 0, "行业走弱" if any(self.etf.get(c) == s for c in weak) else "跌出前 %d 个行业" % q["hold"])
            elif b.pos(s) > tgt[s] * 1.25:
                b.order(cd, s, tgt[s], "仓位调低")
        for s, v in tgt.items():
            if b.pos(s) < v * 0.75:
                b.order(cd, s, v, "行业动量")


PLUGINS = {"steady": SteadyP, "bank": BankP, "smallcap": SmallCapP, "sector_etf": SectorEtfP}


# ======================================================================================
# 主流程
# ======================================================================================
def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type="stock")
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type="fund")
    set_slippage(PriceRelatedSlippage(0.002), type="stock")
    set_slippage(PriceRelatedSlippage(0.0005), type="fund")
    log.set_level("order", "error")
    g.day = 0; g.today = None; g.cache = {}
    g.rule = Rule(CONFIG.get("rule"))
    g.sleeves = {sid: PLUGINS[s["plugin"]](sid, s) for sid, s in CONFIG["sleeves"].items()}
    g.k = {sid: 0.0 for sid in g.sleeves}
    g.alloc_prev = None
    g.tail = {}; g.p1450 = {}; g.day_ret = {}
    g.bank_pool = {}; g.l2 = {}
    run_daily(trade, time=TRADE_TIME)
    run_daily(after_close, time="after_close")
    log.info(VERSION)
    for st in ("normal", "small", "both"):
        row = CONFIG["alloc"].get(st, {})
        if row:
            log.info("「%s」：%s" % (STATES[st], " + ".join("%s %.0f%%" % (g.sleeves[k].name, v * 100) for k, v in row.items() if v > 0) or "货币 ETF"))


def mark_shadows(cd):
    """影子账户按前复权价结算到今天 14:50：昨天 14:50→收盘（after_close 记下）× 昨收→现价（前复权，含今天的除权）"""
    held = sorted(set(s for p in g.sleeves.values() for s in p.b.held()))
    if not held:
        return
    pre = history(1, unit="1d", field="close", security_list=held, df=False, skip_paused=False, fq="pre")
    ratio = {}
    for s in held:
        try:
            pc = float(pre[s][-1]); lp = cd[s].last_price
            if pc > 0 and lp and lp > 0:
                ratio[s] = g.tail.get(s, 1.0) * lp / pc
        except Exception:
            pass
    for p in g.sleeves.values():
        p.b.mark(ratio)


def place(context, cd, s, target_value):
    """带保护价的市价单 + 按手取整。target_value = 0 表示清仓"""
    d = cd[s]
    price = d.last_price
    if price is None or price <= 0 or d.paused:
        return
    pos = context.portfolio.positions
    cur = pos[s].total_amount if s in pos else 0
    if target_value <= 0:
        if cur > 0 and can_sell(cd, s):
            order_target(s, 0, MarketOrderStyle(max(price * 0.98, d.low_limit)))
        return
    tgt = int(target_value / price / 100) * 100
    if s.startswith("688") and cur == 0 and tgt < 200:
        return
    delta = tgt - cur
    if delta >= 100 and can_buy(cd, s):
        lim = min(price * 1.02, d.high_limit)
        afford = int(context.portfolio.available_cash * 0.995 / (lim * 1.0003) / 100) * 100
        delta = min(delta, afford)
        if s.startswith("688") and cur == 0 and delta < 200:
            return
        if delta >= 100:
            order(s, delta, MarketOrderStyle(lim))
    elif delta <= -100 and can_sell(cd, s):
        sell = min(-delta, pos[s].closeable_amount)
        sell = sell if sell == cur else int(sell / 100) * 100
        if sell > 0:
            order(s, -sell, MarketOrderStyle(max(price * 0.98, d.low_limit)))


def sell_cash_etf(context, cd, value):
    e = CASH_ETF
    pos = context.portfolio.positions
    if e not in pos or pos[e].closeable_amount <= 0 or cd[e].paused:
        return
    price = cd[e].last_price
    if price is None or price <= 0:
        return
    have = pos[e].closeable_amount
    n = int(np.ceil(value / price / 100.0)) * 100
    if have - n < 100:
        n = have
    if n > 0:
        order(e, -min(n, have))


def trade(context):
    cd = get_current_data()
    g.today = context.current_dt.date()
    pos = context.portfolio.positions
    mark_shadows(cd)
    for p in g.sleeves.values():
        p.b.today = []
    state = g.rule.step(context)
    a = {sid: float(CONFIG["alloc"].get(state, {}).get(sid, 0)) for sid in g.sleeves}
    prev = g.alloc_prev
    enter = set(sid for sid in g.sleeves if a[sid] > 0 and (prev is None or prev.get(sid, 0) == 0))
    for sid, p in g.sleeves.items():
        if p.always_run or a[sid] > 0:
            p.step(context, cd, force=(sid in enter and p.reselect_on_enter))
    if a != prev:                                      # 行情切换：按新比例重新分钱
        tot = context.portfolio.total_value * 0.99
        for sid, p in g.sleeves.items():
            v = p.b.value()
            g.k[sid] = tot * a[sid] / v if v > 0 and a[sid] > 0 else 0.0
        log.info("行情「%s」（中小盘走强=%s 大盘上涨期=%s）→ %s" % (STATES[state], g.rule.g1, g.rule.up,
                 " + ".join("%s %.0f%%" % (g.sleeves[k].name, v * 100) for k, v in a.items() if v > 0) or "全部货币 ETF"))
        g.alloc_prev = a
    tgt = {}
    for sid, p in g.sleeves.items():
        if g.k[sid] <= 0:
            continue
        for s, v in p.b.v.items():
            if v > 0:
                tgt[s] = tgt.get(s, 0.0) + g.k[sid] * v
    tol = float(CONFIG.get("tol", 0.25))
    # 1. 卖：不在目标里的清仓，超出目标太多的减仓
    for s in list(pos.keys()):
        if s == CASH_ETF or pos[s].total_amount == 0:
            continue
        if s not in tgt:
            place(context, cd, s, 0)
        elif pos[s].value > tgt[s] * (1 + tol):
            place(context, cd, s, tgt[s])
    # 2. 需要买入时先卖货币 ETF 腾钱
    need = sum(max(0.0, v - (pos[s].value if s in pos else 0)) for s, v in tgt.items()
               if (pos[s].value if s in pos else 0) < v * (1 - tol))
    pf = context.portfolio
    if need > 0 and CASH_ETF in pos and pos[CASH_ETF].total_amount > 0 and need > pf.available_cash:
        sell_cash_etf(context, cd, (need - pf.available_cash) * 1.02)
    # 3. 买
    for s, v in tgt.items():
        cur = pos[s].value if s in pos else 0
        if cur < v * (1 - tol):
            place(context, cd, s, v)
    # 4. 剩余现金买货币 ETF
    pf = context.portfolio
    spare = pf.available_cash - pf.total_value * CASH_BUFFER
    if spare > 20000 and not cd[CASH_ETF].paused:
        order_value(CASH_ETF, spare)
    # 记下 14:50 价格（收盘后算 14:50→收盘 的涨跌，第二天结算影子账户用）
    g.p1450 = {s: cd[s].last_price for p in g.sleeves.values() for s in p.b.held()}
    slot = {}
    for sid, v in a.items():
        sl = CONFIG["sleeves"][sid].get("slot", "other")
        slot[sl] = slot.get(sl, 0) + v * 100
    record(稳健位=slot.get("steady", 0), 个股位=slot.get("stock", 0), 板块位=slot.get("sector", 0))
    g.day += 1


def after_close(context):
    cd = get_current_data()
    g.tail = {}
    for s, p in g.p1450.items():
        try:
            c = cd[s].last_price
            if p and p > 0 and c and c > 0:
                g.tail[s] = c / p
        except Exception:
            pass
    # 稳健类插件的影子净值：选中股票今天的前复权收益
    picks = sorted(set(s for p in g.sleeves.values() for s in getattr(p, "picks", []) if isinstance(p, SteadyP)))
    g.day_ret = {}
    if picks:
        px = pivot_close(picks, context.current_dt.date(), 2)
        if len(px) >= 2:
            r = (px.iloc[-1] / px.iloc[-2] - 1).dropna()
            g.day_ret = r.to_dict()
    for p in g.sleeves.values():
        p.after_close(context)
