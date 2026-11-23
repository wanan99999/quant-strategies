# -*- coding: utf-8 -*-
"""
上行期板块 · 申万二级行业动量 V1.0 —— 聚宽回测代码（上行期研究第二轮「板块方向」）

思路：行业动量只在真正的上行期有效。2023 年以来不择时做申万二级行业轮动，扣成本后年化 -5% ~ -30%
     （一周一调，换手 20–45 倍）；但只在「中证1000 趋势成立 且 大盘处于上行期」时持有 60 日最强的行业，
     上行期那些日子年化可达 +62%，其余时间空仓。

① 入场开关（每天 14:50）两个条件都满足才持仓，否则当天清仓、资金买货币 ETF（511880）
    a. 中证1000：ER20 > 0.2 且 当前价 > MA20（ER20 = (当前价 − 20 日前收盘) ÷ 20 天每日涨跌绝对值之和）
    b. 上行期（与研究里的行情状态同一口径，用前一日收盘）：
       沪深300 > MA120，且 MA120 高于 20 个交易日前，且 申万一级行业里站上 60 日线的占比 ≥ 50%
② 选行业（用前一日收盘的申万二级行业指数）
    行业自身趋势：行业指数 ER20 > 0.2 且 > MA20，不满足的行业不选，已持有的当天卖掉
    在自身趋势成立的行业里取 60 日涨幅最高的 3 个；每 10 个交易日换一次，开关打开当天立即建仓
③ 每个行业怎么买
    行业成分股里流通市值最大的 5 只（非 ST、非停牌、上市满 60 天），每只 1/15 仓位
    涨停买不进、跌停/停牌卖不出

本地测算（申万二级行业指数，单边 0.15%，当日收盘成交，3 份资金错开取平均；主要看 2023 起）：
    2023 起：年化 +13.8%、最大回撤 -15.2%、夏普 0.85；上行期日子年化 +62%；开关打开约 12% 的交易日
    行业排名和行业趋势改用前一日收盘（本代码的做法）：2023 起 +11.7%
    2015 起（参考，当时只有 69 个二级行业有数据）：+13.1% / -15.2%
    参数附近（开关 3 种 × 20/60 日动量 × 持 3/5 个 × 5/10 日调仓）：只看此开关 12 组 2023 起年化中位 +8.7%（+5.5% ~ +12.9%）
    与「大盘股熊市震荡市稳健策略」切换（开关开时资金跑本策略，关时跑稳健策略）：
        2023 起 +34.8% / -19.5% / 夏普 1.58（参数中位 +28.8%）；稳健策略单独 +19.0% / -10.5%
与本地口径的差异：本地直接持有行业指数；这里用每个行业市值最大的 5 只股票代替，会有跟踪误差
弱点：只在上行期标签确认后才进场，急涨的头几天抓不到；单独运行时 88% 的时间空仓
建议回测：2019-01-01 起，分钟频率，初始资金 ≥ 100 万
"""
import builtins
from jqdata import *
import numpy as np
import pandas as pd
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "上行期板块·申万二级行业动量 V1.0"


class Cfg:
    GATE_INDEX = "000852.XSHG"   # 中证1000
    REG_INDEX = "000300.XSHG"    # 沪深300（上行期判断）
    ER_N = 20
    ER_MIN = 0.20
    MA_N = 20
    REG_MA = 120
    REG_SLOPE = 20
    BR_MA = 60
    BR_MIN = 0.50
    MOM_N = 60                   # 行业动量窗口
    IND_N = 3                    # 持有几个行业
    STK_PER_IND = 5              # 每个行业买几只
    REBAL_DAYS = 10
    TRADE_TIME = "14:50"
    CASH_ETF = "511880.XSHG"
    CASH_BUFFER = 0.01


def initialize(context):
    set_benchmark("000852.XSHG")
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
    g.gate = False
    g.inds = []          # 当前持有的行业代码
    g.stocks = {}        # 行业代码 -> 该行业买入的股票
    run_daily(trade, time=Cfg.TRADE_TIME)
    log.info(VERSION)


# ---------------- 工具 ----------------
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
        order(s, delta, MarketOrderStyle(min(price * 1.02, d.high_limit)))
    elif delta <= -100 and can_sell(cd, s):
        sell = min(-delta, pos[s].closeable_amount)
        sell = sell if sell == cur else int(sell / 100) * 100
        if sell > 0:
            order(s, -sell, MarketOrderStyle(max(price * 0.98, d.low_limit)))


def er_ma_ok(s):
    """s: 收盘序列（最后一个是最新价）→ ER20 > 0.2 且 > MA20"""
    s = np.asarray(s, dtype=float)
    if len(s) < max(Cfg.ER_N + 1, Cfg.MA_N) or np.isnan(s[-(Cfg.ER_N + 1):]).any():
        return False
    path = np.abs(np.diff(s[-(Cfg.ER_N + 1):])).sum()
    er = (s[-1] - s[-(Cfg.ER_N + 1)]) / path if path > 0 else 0.0
    return er > Cfg.ER_MIN and s[-1] > s[-Cfg.MA_N:].mean()


def sw_close(codes, day, cal_days):
    """申万指数日收盘（截至 day），返回 DataFrame[日期 × 代码]；分批查询避免单次行数上限"""
    start = day - pd.Timedelta(days=cal_days)
    out = []
    for i in range(0, len(codes), 25):
        q = query(finance.SW1_DAILY_PRICE.date, finance.SW1_DAILY_PRICE.code, finance.SW1_DAILY_PRICE.close
                  ).filter(finance.SW1_DAILY_PRICE.code.in_(codes[i:i + 25]),
                           finance.SW1_DAILY_PRICE.date >= start, finance.SW1_DAILY_PRICE.date <= day)
        out.append(finance.run_query(q))
    df = pd.concat(out) if out else pd.DataFrame(columns=["date", "code", "close"])
    if len(df) == 0:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="code", values="close").sort_index()


def gate_1000(context):
    n = max(Cfg.ER_N + 1, Cfg.MA_N)
    h = attribute_history(Cfg.GATE_INDEX, n, "1d", ["close"], skip_paused=True, df=False)["close"]
    b = get_price(Cfg.GATE_INDEX, end_date=context.current_dt, frequency="1m", fields=["close"], count=1)
    if len(h) < n or len(b) == 0:
        return False
    return er_ma_ok(np.append(h, float(b["close"].iloc[-1])))


def uptrend(context):
    """上行期：沪深300 > MA120 且 MA120 高于 20 日前 且 申万一级站上 60 日线 ≥ 50%（前一日收盘）"""
    day = context.previous_date
    c = attribute_history(Cfg.REG_INDEX, Cfg.REG_MA + Cfg.REG_SLOPE, "1d", ["close"], skip_paused=True, df=False)["close"]
    if len(c) < Cfg.REG_MA + Cfg.REG_SLOPE:
        return False
    ma_now = c[-Cfg.REG_MA:].mean()
    ma_old = c[-(Cfg.REG_MA + Cfg.REG_SLOPE):-Cfg.REG_SLOPE].mean()
    if not (c[-1] > ma_now and ma_now > ma_old):
        return False
    l1 = list(get_industries("sw_l1", date=day).index)
    px = sw_close(l1, day, 130)
    px = px.loc[:, px.notna().sum() >= Cfg.BR_MA]
    if px.shape[1] == 0:
        return False
    above = (px.iloc[-1] > px.iloc[-Cfg.BR_MA:].mean())
    return above.mean() >= Cfg.BR_MIN


def rank_industries(context):
    """返回 (自身趋势成立的行业按 60 日涨幅排序的列表, 自身趋势成立的行业集合)"""
    day = context.previous_date
    inds = get_industries("sw_l2", date=day)
    inds = inds[inds.start_date <= day]
    px = sw_close(list(inds.index), day, 130)
    ok, mom = set(), {}
    for code in px.columns:
        s = px[code].dropna()
        if len(s) < Cfg.MOM_N + 1:
            continue
        if er_ma_ok(s.values):
            ok.add(code)
            mom[code] = s.iloc[-1] / s.iloc[-(Cfg.MOM_N + 1)] - 1
    order_ = sorted(mom, key=lambda k: -mom[k])
    return order_, ok


def stocks_of(context, cd, ind):
    day = context.previous_date
    ss = get_industry_stocks(ind, date=day)
    ss = [s for s in ss if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")
          and (not cd[s].is_st) and (not cd[s].paused) and "退" not in cd[s].name
          and get_security_info(s).start_date <= day - pd.Timedelta(days=60)]
    if not ss:
        return []
    q = query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(ss))
    df = get_fundamentals(q, date=day).dropna().sort_values("circulating_market_cap", ascending=False)
    return list(df.code[:Cfg.STK_PER_IND])


def clear_all(context, cd):
    pos = context.portfolio.positions
    for s in list(pos.keys()):
        if s != Cfg.CASH_ETF and pos[s].total_amount > 0:
            place(context, cd, s, 0)
    g.inds, g.stocks = [], {}


def park_cash(context, cd):
    e = Cfg.CASH_ETF
    pf = context.portfolio
    if cd[e].paused:
        return
    spare = pf.available_cash - pf.total_value * Cfg.CASH_BUFFER
    if spare > 10000:
        order_value(e, spare)


# ---------------- 主流程 ----------------
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    on = gate_1000(context) and uptrend(context)
    if not on:
        if g.gate:
            log.info("开关关闭，清仓")
        clear_all(context, cd)
        park_cash(context, cd)
        g.gate = False
        g.day += 1
        record(开关=0)
        return
    just_on = not g.gate
    order_, ok = rank_industries(context)
    # 已持有行业自身趋势失效 → 当天卖掉，空位留到下次调仓
    for ind in list(g.inds):
        if ind not in ok:
            for s in g.stocks.get(ind, []):
                place(context, cd, s, 0)
            g.inds.remove(ind); g.stocks.pop(ind, None)
            log.info("行业 %s 趋势失效，卖出" % ind)
    if just_on or g.day % Cfg.REBAL_DAYS == 0:
        new = [i for i in order_[:Cfg.IND_N]]
        for ind in list(g.inds):
            if ind not in new:
                for s in g.stocks.get(ind, []):
                    place(context, cd, s, 0)
                g.inds.remove(ind); g.stocks.pop(ind, None)
        for ind in new:
            if ind not in g.inds:
                g.inds.append(ind); g.stocks[ind] = stocks_of(context, cd, ind)
        # 货币 ETF 腾钱
        e = Cfg.CASH_ETF
        if e in pos and pos[e].total_amount > 0 and not cd[e].paused:
            order_target(e, 0)
        names = get_industries("sw_l2", date=context.previous_date)
        log.info("持有行业：%s" % "、".join(names.loc[i, "name"] if i in names.index else i for i in g.inds))
    # 按目标买入 / 调整（每只 = 总资产 / (行业数 × 每行业只数)）
    want = set(s for i in g.inds for s in g.stocks.get(i, []))
    for s in list(pos.keys()):
        if s != Cfg.CASH_ETF and pos[s].total_amount > 0 and s not in want:
            place(context, cd, s, 0)
    tv = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) / (Cfg.IND_N * Cfg.STK_PER_IND)
    for s in want:
        cur = pos[s].value if s in pos else 0
        if cur < tv * 0.8:
            place(context, cd, s, tv)
    park_cash(context, cd)
    g.gate = True
    g.day += 1
    record(开关=100, 行业数=len(g.inds))
