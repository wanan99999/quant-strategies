# -*- coding: utf-8 -*-
"""
全行情切换组合 · 板块+稳健 V1.1 —— 聚宽回测代码

V1.1 修改（相对 V1.0）：
    1. 同时持股不超过 10 只：稳健块 30 → 10 只（缓冲 60 → 20，同一行业最多 5 → 3 只）；个股块 20 → 10 只（缓冲 20）；
       板块块每个行业 5 → 3 只（3 个行业共 9 只）。「个股+板块+稳健」在上行期为 个股 4 只 + 板块 3 个行业各 2 只 = 10 只
    2. 修复货币 ETF（511880）减仓报错「平仓数量不能小于 100」：减仓数量按 100 份向上取整，剩余不足 100 份就全部卖出
    3. 修复「资金不足 / 开仓数量不能小于 100」：买入前按当前可用资金封顶并按手取整，不够 1 手就不下单

一个账户里跑三块资金，按行情开关分配（每天 14:50）：
    稳健块：「大盘股熊市震荡市稳健策略 V1.1」规则，持股改为 10 只（大盘 800 盈利股、距一年高点 + 低波动前 10、波动率控仓、回撤风控、单只止损）
    个股块：「上行期个股·小市值反转 V1.0」（流动性池 小市值排名 + 20 日反转排名 前 10，每 5 日调仓）
    板块块：「上行期板块·申万二级行业动量 V1.0」（自身趋势成立的二级行业里 60 日涨幅前 3，每行业流通市值前 3 只，每 10 日调仓）

开关：
    开关1「中证1000趋势」：中证1000 ER20 > 0.2 且 当前价 > MA20
           ER20 = (当前价 − 20 日前收盘) ÷ 20 天每日涨跌绝对值之和
    开关2「上行期」：开关1 成立，且 沪深300 > MA120、MA120 高于 20 日前、申万一级站上 60 日线 ≥ 50%（后三项用前一日收盘）
本版本资金分配（MODE = "板块+稳健"）：
    开关2 成立 → 100% 板块块（3 个行业各 3 只）；否则 → 100% 稳健块（10 只）（2023 年以来开关2 约 12% 的交易日成立）

本地测算（后复权日线，单边 0.15%，当日收盘成交；切换时稳健块另计 0.15%；主要看 2023 起）：
    V1.1（持股 ≤10）2023 起：年化 +33.9%、最大回撤 -22.1%、夏普 1.54；上行期日子 +75% · 震荡期 +27% · 下行期 +13%
    逐年：2023 +28.2%、2024 +50.7%、2025 +11.4%、2026 至今 +37.3%；2015 起（参考）+19.8% / -22.4%
    V1.0（稳健 30 只）：2023 起 +34.8% / -19.5%；本地板块块按行业指数测算，聚宽每行业 3 只的跟踪误差未计入
与本地口径的差异：
    本地三块各自满额回测再按当日资金比例拼接；这里在一个账户里按目标市值合并下单，同一只股票两块都要时合并持有
    板块块本地直接持有行业指数，这里用每个行业流通市值前 3 只（与个股块同时持有时 2 只）；行业排名用前一日收盘
    稳健块在两次调仓之间只在总仓位变动 ≥10% 或开关切换时调整
建议回测：2019-01-01 起（看 2023 起为主），分钟频率，初始资金 ≥ 100 万（同时持股不超过 10 只，切换当天新旧两块交接、跌停卖不出时可能短暂多于 10 只）
"""
import builtins
from jqdata import *
import numpy as np
import pandas as pd
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "全行情切换组合·板块+稳健 V1.1"
MODE = "板块+稳健"     # "个股+稳健" / "板块+稳健" / "个股+板块+稳健"


class Cfg:
    TRADE_TIME = "14:50"
    CASH_ETF = "511880.XSHG"     # 货币 ETF（银华日利）
    CASH_BUFFER = 0.01
    TOL = 0.25                   # 已持有股票偏离目标市值超过 25% 才调整
    # ---- 开关 ----
    GATE_INDEX = "000852.XSHG"
    REG_INDEX = "000300.XSHG"
    ER_N, ER_MIN, MA_N = 20, 0.20, 20
    REG_MA, REG_SLOPE, BR_MA, BR_MIN = 120, 20, 60, 0.50
    # ---- 稳健块（V1.1）----
    A_POOL = 800; A_LIST_DAYS = 365; A_ROE_MIN = 0.05; A_HIGH_N = 250; A_VOL_N = 60
    A_HOLD = 10; A_BUFFER = 20; A_REBAL = 5; A_IND_MAX = 3; A_STOP = 0.10
    A_VOL_TARGET = 0.15; A_VOL_WIN = 40; A_EXPO_BAND = 0.10; A_DD_HALF = 0.15; A_DD_OUT = 0.25; A_DD_LOOK = 250
    # ---- 个股块 ----
    S_LIST_DAYS = 365; S_AMT_N = 20; S_AMT_DROP = 0.30; S_REV_N = 20; S_HOLD = 10; S_HOLD_MIX = 4; S_BUFFER = 20; S_REBAL = 5
    # ---- 板块块 ----
    B_MOM_N = 60; B_IND_N = 3; B_PER_IND = 3; B_PER_IND_MIX = 2; B_REBAL = 10


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
    g.day = 0
    # 稳健块
    g.a_picks = []; g.a_sleeve = []; g.a_expo = 0.0; g.a_tv = {}
    # 个股块
    g.s_on = False; g.s_picks = []
    # 板块块
    g.b_on = False; g.b_inds = []; g.b_stocks = {}
    g.last_alloc = None
    run_daily(trade, time=Cfg.TRADE_TIME)
    run_daily(after_close, time="after_close")
    log.info(VERSION)


# ================= 通用工具 =================
def can_buy(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price < d.high_limit - 1e-6


def can_sell(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price > d.low_limit + 1e-6


def place(context, cd, s, target_value):
    """带保护价的市价单 + 按手取整。target_value = 0 表示清仓。"""
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
        afford = int(context.portfolio.available_cash * 0.995 / (lim * 1.0003) / 100) * 100   # 按可用资金封顶（含佣金余量）
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


def er_ma_ok(s):
    s = np.asarray(s, dtype=float)
    if len(s) < max(Cfg.ER_N + 1, Cfg.MA_N) or np.isnan(s[-(Cfg.ER_N + 1):]).any():
        return False
    path = np.abs(np.diff(s[-(Cfg.ER_N + 1):])).sum()
    er = (s[-1] - s[-(Cfg.ER_N + 1)]) / path if path > 0 else 0.0
    return er > Cfg.ER_MIN and s[-1] > s[-Cfg.MA_N:].mean()


def sw_close(codes, day, cal_days):
    """申万指数日收盘（截至 day）DataFrame[日期 × 代码]，分批查询"""
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


def base_stocks(context, cd, list_days):
    day = context.previous_date
    secs = get_all_securities(["stock"], date=day)
    secs = secs[secs.start_date <= day - pd.Timedelta(days=list_days)]
    ss = [s for s in secs.index if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
    return [s for s in ss if (not cd[s].is_st) and (not cd[s].paused) and "退" not in cd[s].name]


def sell_cash_etf(context, cd, value):
    """卖出约 value 元的货币 ETF：按 100 份向上取整；卖完剩余不足 100 份就全部卖出"""
    e = Cfg.CASH_ETF
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


# ================= 开关 =================
def gate_1000(context):
    n = max(Cfg.ER_N + 1, Cfg.MA_N)
    h = attribute_history(Cfg.GATE_INDEX, n, "1d", ["close"], skip_paused=True, df=False)["close"]
    b = get_price(Cfg.GATE_INDEX, end_date=context.current_dt, frequency="1m", fields=["close"], count=1)
    if len(h) < n or len(b) == 0:
        return False
    return er_ma_ok(np.append(h, float(b["close"].iloc[-1])))


def uptrend(context):
    day = context.previous_date
    c = attribute_history(Cfg.REG_INDEX, Cfg.REG_MA + Cfg.REG_SLOPE, "1d", ["close"], skip_paused=True, df=False)["close"]
    if len(c) < Cfg.REG_MA + Cfg.REG_SLOPE:
        return False
    ma_now = c[-Cfg.REG_MA:].mean(); ma_old = c[-(Cfg.REG_MA + Cfg.REG_SLOPE):-Cfg.REG_SLOPE].mean()
    if not (c[-1] > ma_now and ma_now > ma_old):
        return False
    px = sw_close(list(get_industries("sw_l1", date=day).index), day, 130)
    px = px.loc[:, px.notna().sum() >= Cfg.BR_MA] if len(px) else px
    if px.shape[1] == 0:
        return False
    return (px.iloc[-1] > px.iloc[-Cfg.BR_MA:].mean()).mean() >= Cfg.BR_MIN


def allocation(g1, g2):
    """返回 (稳健块, 个股块, 板块块) 资金占比"""
    if MODE == "个股+稳健":
        return (0.0, 1.0, 0.0) if g1 else (1.0, 0.0, 0.0)
    if MODE == "板块+稳健":
        return (0.0, 0.0, 1.0) if g2 else (1.0, 0.0, 0.0)
    if g2:
        return (0.0, 0.5, 0.5)
    return (0.0, 1.0, 0.0) if g1 else (1.0, 0.0, 0.0)


# ================= 稳健块（V1.1 规则）=================
def a_exposure():
    r = np.array(g.a_sleeve[-Cfg.A_VOL_WIN:])
    if len(r) < 20:
        e = 1.0
    else:
        vol = np.std(r, ddof=1) * np.sqrt(252)
        e = 1.0 if vol <= 0 else min(1.0, Cfg.A_VOL_TARGET / vol)
    if len(g.a_sleeve) > 0:
        nav = np.cumprod(1 + np.array(g.a_sleeve)); dd = nav[-1] / np.max(nav[-Cfg.A_DD_LOOK:]) - 1
        if dd < -Cfg.A_DD_OUT:
            e = 0.0
        elif dd < -Cfg.A_DD_HALF:
            e *= 0.5
    return e


def a_select(context, cd):
    day = context.previous_date
    stocks = base_stocks(context, cd, Cfg.A_LIST_DAYS)
    q = query(valuation.code, valuation.circulating_market_cap, indicator.roe, income.net_profit).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=day).dropna(subset=["circulating_market_cap"])
    df = df.sort_values("circulating_market_cap", ascending=False).head(Cfg.A_POOL)
    pool = list(df[(df.net_profit > 0) & (df.roe * 4 / 100 > Cfg.A_ROE_MIN)].code)
    if len(pool) < Cfg.A_HOLD:
        return g.a_picks
    px = get_price(pool, end_date=day, count=Cfg.A_HIGH_N - 1, fields=["close"], fq="pre", panel=False
                   ).pivot(index="time", columns="code", values="close")
    cur = pd.Series({s: cd[s].last_price for s in px.columns})
    px = pd.concat([px, cur.to_frame().T]); px = px.loc[:, px.notna().sum() >= 200]
    sc = ((px.iloc[-1] / px.max()).rank(pct=True) + (-px.pct_change().iloc[-Cfg.A_VOL_N:].std()).rank(pct=True)).dropna()
    codes = list(sc.sort_values(ascending=False).index); rank = {s: i for i, s in enumerate(codes)}
    ind = get_industry(codes + [s for s in g.a_picks if s not in rank], date=day)
    def ind_of(s): return ind.get(s, {}).get("sw_l2", {}).get("industry_name", "未知")
    cnt, pick = {}, []
    def add(s): pick.append(s); cnt[ind_of(s)] = cnt.get(ind_of(s), 0) + 1
    for s in sorted(g.a_picks, key=lambda x: rank.get(x, 10 ** 9)):
        if rank.get(s, 10 ** 9) < Cfg.A_BUFFER and len(pick) < Cfg.A_HOLD and cnt.get(ind_of(s), 0) < Cfg.A_IND_MAX:
            add(s)
    for s in codes:
        if len(pick) >= Cfg.A_HOLD:
            break
        if s not in pick and cnt.get(ind_of(s), 0) < Cfg.A_IND_MAX:
            add(s)
    return pick


def a_update(context, cd, share):
    """维护稳健块：止损、调仓日选股、总仓位；返回该块每只股票的目标市值"""
    pos = context.portfolio.positions
    # 单只止损（只对稳健块持有、且不在上行块目标里的股票）
    up = set(g.s_picks) | set(s for v in g.b_stocks.values() for s in v)
    for s in list(g.a_picks):
        if s in pos and pos[s].total_amount > 0 and s not in up and pos[s].avg_cost > 0 \
                and cd[s].last_price / pos[s].avg_cost - 1 < -Cfg.A_STOP and can_sell(cd, s):
            g.a_picks.remove(s); g.a_tv.pop(s, None)
            log.info("稳健块止损 %s %s" % (s, cd[s].name))
    expo = a_exposure()
    rebal = g.day % Cfg.A_REBAL == 0
    if rebal:
        g.a_picks = a_select(context, cd)          # 不持仓时也照常选股，影子净值连续
    base = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) * share
    if rebal or abs(expo - g.a_expo) >= Cfg.A_EXPO_BAND or (expo == 0 and g.a_expo > 0) \
            or share != getattr(g, "a_share", None):
        g.a_expo = expo; g.a_share = share
        tv = base * expo / Cfg.A_HOLD
        g.a_tv = {s: tv for s in g.a_picks} if share > 0 and expo > 0 else {}
    return dict(g.a_tv)


# ================= 个股块 =================
def s_select(context, cd, n):
    day = context.previous_date
    stocks = base_stocks(context, cd, Cfg.S_LIST_DAYS)
    amt = get_price(stocks, end_date=day, count=Cfg.S_AMT_N, fields=["money"], panel=False
                    ).pivot(index="time", columns="code", values="money").mean()
    amt = amt[amt > 0]; pool = list(amt[amt.rank(pct=True) > Cfg.S_AMT_DROP].index)
    cap = get_fundamentals(query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(pool)),
                           date=day).dropna().set_index("code")["circulating_market_cap"]
    cap = cap[cap > 0]
    px = get_price(list(cap.index), end_date=day, count=Cfg.S_REV_N, fields=["close"], fq="pre", panel=False
                   ).pivot(index="time", columns="code", values="close")
    cur = pd.Series({s: cd[s].last_price for s in px.columns})
    ret = (cur / px.iloc[0] - 1).replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({"cap": cap, "ret": ret}).dropna()
    sc = ((-np.log(df["cap"])).rank(pct=True) + (-df["ret"]).rank(pct=True)).sort_values(ascending=False)
    codes = list(sc.index); rank = {s: i for i, s in enumerate(codes)}
    pick = [s for s in sorted(g.s_picks, key=lambda x: rank.get(x, 10 ** 9)) if rank.get(s, 10 ** 9) < Cfg.S_BUFFER][:n]
    for s in codes:
        if len(pick) >= n:
            break
        if s not in pick:
            pick.append(s)
    return pick


def s_update(context, cd, share, n):
    if share <= 0:
        g.s_on = False; g.s_picks = []
        return {}
    if (not g.s_on) or g.day % Cfg.S_REBAL == 0 or not g.s_picks or len(g.s_picks) != n:
        g.s_picks = s_select(context, cd, n)
    g.s_on = True
    tv = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) * share / n
    return {s: tv for s in g.s_picks}


# ================= 板块块 =================
def b_rank(context):
    day = context.previous_date
    inds = get_industries("sw_l2", date=day); inds = inds[inds.start_date <= day]
    px = sw_close(list(inds.index), day, 130)
    ok, mom = set(), {}
    for code in px.columns:
        s = px[code].dropna()
        if len(s) >= Cfg.B_MOM_N + 1 and er_ma_ok(s.values):
            ok.add(code); mom[code] = s.iloc[-1] / s.iloc[-(Cfg.B_MOM_N + 1)] - 1
    return sorted(mom, key=lambda k: -mom[k]), ok


def b_stocks_of(context, cd, ind, k):
    day = context.previous_date
    ss = [s for s in get_industry_stocks(ind, date=day)
          if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ") and (not cd[s].is_st) and (not cd[s].paused)
          and "退" not in cd[s].name and get_security_info(s).start_date <= day - pd.Timedelta(days=60)]
    if not ss:
        return []
    df = get_fundamentals(query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(ss)),
                          date=day).dropna().sort_values("circulating_market_cap", ascending=False)
    return list(df.code[:k])


def b_update(context, cd, share, k):
    if share <= 0:
        g.b_on = False; g.b_inds = []; g.b_stocks = {}
        return {}
    order_, ok = b_rank(context)
    for ind in list(g.b_inds):                 # 行业自身趋势失效当天卖出
        if ind not in ok:
            g.b_inds.remove(ind); g.b_stocks.pop(ind, None)
    if any(len(v) > k for v in g.b_stocks.values()):          # 每行业只数变少时立即截断
        g.b_stocks = {i: v[:k] for i, v in g.b_stocks.items()}
    if (not g.b_on) or g.day % Cfg.B_REBAL == 0:
        new = order_[:Cfg.B_IND_N]
        for ind in list(g.b_inds):
            if ind not in new:
                g.b_inds.remove(ind); g.b_stocks.pop(ind, None)
        for ind in new:
            if ind not in g.b_inds:
                g.b_inds.append(ind); g.b_stocks[ind] = b_stocks_of(context, cd, ind, k)
    g.b_on = True
    tv = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) * share / (Cfg.B_IND_N * k)
    return {s: tv for i in g.b_inds for s in g.b_stocks.get(i, [])}


# ================= 主流程 =================
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    g1 = gate_1000(context)
    g2 = g1 and MODE != "个股+稳健" and uptrend(context)
    a_sh, s_sh, b_sh = allocation(g1, g2)
    if (a_sh, s_sh, b_sh) != g.last_alloc:
        log.info("资金分配 → 稳健 %.0f%% · 个股 %.0f%% · 板块 %.0f%%（中证1000趋势=%s 上行期=%s）"
                 % (a_sh * 100, s_sh * 100, b_sh * 100, g1, g2))
        g.last_alloc = (a_sh, s_sh, b_sh)
    tgt = {}
    mix = s_sh > 0 and b_sh > 0                     # 个股、板块同时持有时各自减少只数，合计 ≤ 10
    n_s = Cfg.S_HOLD_MIX if mix else Cfg.S_HOLD
    k_b = Cfg.B_PER_IND_MIX if mix else Cfg.B_PER_IND
    for part in (s_update(context, cd, s_sh, n_s), b_update(context, cd, b_sh, k_b), a_update(context, cd, a_sh)):
        for s, v in part.items():
            tgt[s] = tgt.get(s, 0.0) + v
    # 1. 卖：不在目标里的清仓，超出目标太多的减仓
    for s in list(pos.keys()):
        if s == Cfg.CASH_ETF or pos[s].total_amount == 0:
            continue
        if s not in tgt:
            place(context, cd, s, 0)
        elif pos[s].value > tgt[s] * (1 + Cfg.TOL):
            place(context, cd, s, tgt[s])
    # 2. 需要买入时先卖货币 ETF 腾钱
    need = sum(max(0.0, v - (pos[s].value if s in pos else 0)) for s, v in tgt.items()
               if (pos[s].value if s in pos else 0) < v * (1 - Cfg.TOL))
    e = Cfg.CASH_ETF
    if need > 0 and e in pos and pos[e].total_amount > 0 and not cd[e].paused:
        pf = context.portfolio
        if need > pf.available_cash:
            sell_cash_etf(context, cd, (need - pf.available_cash) * 1.02)
    # 3. 买
    for s, v in tgt.items():
        cur = pos[s].value if s in pos else 0
        if cur < v * (1 - Cfg.TOL):
            place(context, cd, s, v)
    # 4. 剩余现金买货币 ETF
    pf = context.portfolio
    spare = pf.available_cash - pf.total_value * Cfg.CASH_BUFFER
    if spare > 20000 and not cd[e].paused:
        order_value(e, spare)
    record(稳健块=a_sh * 100, 个股块=s_sh * 100, 板块块=b_sh * 100)
    g.day += 1


def after_close(context):
    """稳健块影子净值：选股组合当日等权收益（不管当天是否实际持有）"""
    if not g.a_picks:
        g.a_sleeve.append(0.0); return
    px = get_price(g.a_picks, end_date=context.current_dt.date(), count=2, fields=["close"], fq="pre", panel=False
                   ).pivot(index="time", columns="code", values="close")
    if len(px) < 2:
        g.a_sleeve.append(0.0); return
    r = (px.iloc[-1] / px.iloc[-2] - 1).dropna()
    g.a_sleeve.append(float(r.mean()) if len(r) else 0.0)
    g.a_sleeve = g.a_sleeve[-400:]
