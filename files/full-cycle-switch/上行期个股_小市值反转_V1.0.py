# -*- coding: utf-8 -*-
"""
上行期个股 · 小市值反转 V1.0 —— 聚宽回测代码（上行期研究第二轮「个股方向」）

思路：只在中小盘指数走出趋势时持股；持股时买「市值小、近 20 日跌得多」的股票。
     A 股个股短期反转很强：追涨（20/60 日动量、放量突破、创新高、涨停多、高换手）的前 30 只在
     2023 年以来年化 -30% ~ -60%（扣成本前就为负）。上行期真正能吃到的，是「趋势成立时低位、弹性大」的小票。

① 入场开关（每天 14:50）
    中证1000 指数：路径效率 ER20 = (当前价 − 20 日前收盘) ÷ 20 天每日涨跌绝对值之和
    ER20 > 0.2 且 当前价 > MA20 → 开；否则关 → 当天清仓，资金买货币 ETF（511880）
② 股票池（调仓时重算）
    上市满 365 天、非 ST、非停牌、非北交所、名称不含「退」；去掉近 20 日日均成交额后 30% 的股票
③ 打分
    小市值 = 流通市值越小越好 → 截面百分位排名；反转 = 近 20 日涨幅越低越好 → 截面百分位排名；两者相加
④ 选股与轮换
    取前 20 只等权；每 5 个交易日调仓，开关由关转开的当天也立即建仓
    已持有的股票排名还在前 40 就继续拿（缓冲区）；涨停买不进、跌停/停牌卖不出
不加个股止损（本地测过：止损在反转型选股里会砍在低点）。

本地测算（后复权日线，单边 0.15%，当日收盘成交，3 份资金错开调仓取平均；主要看 2023 起）：
    2023 起：年化 +19.9%、最大回撤 -12.2%、夏普 1.10；上行期日子年化 +26%；开关打开约 24% 的交易日
    2015 起（参考）：年化 +13.7%、最大回撤 -33%
    参数附近（开关 ER10~30 / 阈值 0.1~0.3、反转窗口 20~60 日、持股 10~40 只）45 组：2023 起年化中位 +16.4%（10%~90%：+4.5% ~ +22.9%）
    与「大盘股熊市震荡市稳健策略」切换（开关开时资金跑本策略，关时跑稳健策略）：
        2023 起 +35.1% / -13.1% / 夏普 1.55（参数 45 组中位 +29.4%）；稳健策略单独 +19.0% / -10.5%
弱点：
    开关晚一天执行，2023 起年化降到 +12.3%（入场当天的涨幅很重要，务必 14:50 下单）
    开关窗口改成 30 日只有 +5.4%；单边成本 0.3% 时 +15.2%
    重仓小市值，微盘股流动性危机（如 2024 年 1–2 月）时只能靠开关关掉保护
建议回测：2019-01-01 起，分钟频率，初始资金 ≥ 100 万（20 只每只至少一手）
"""
import builtins
from jqdata import *
import numpy as np
import pandas as pd
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "上行期个股·小市值反转 V1.0"


class Cfg:
    GATE_INDEX = "000852.XSHG"   # 中证1000
    ER_N = 20                    # 路径效率窗口
    ER_MIN = 0.20
    MA_N = 20
    MIN_LIST_DAYS = 365
    AMT_N = 20                   # 成交额窗口
    AMT_DROP = 0.30              # 去掉成交额后 30%
    REV_N = 20                   # 反转窗口
    HOLD_N = 20
    BUFFER = 40
    REBAL_DAYS = 5
    TRADE_TIME = "14:50"
    CASH_ETF = "511880.XSHG"     # 货币 ETF（银华日利）
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
    g.picks = []
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
    if s.startswith("688") and cur == 0 and tgt < 200:   # 科创板首次买入至少 200 股
        return
    delta = tgt - cur
    if delta >= 100 and can_buy(cd, s):
        order(s, delta, MarketOrderStyle(min(price * 1.02, d.high_limit)))
    elif delta <= -100 and can_sell(cd, s):
        sell = min(-delta, pos[s].closeable_amount)
        sell = sell if sell == cur else int(sell / 100) * 100
        if sell > 0:
            order(s, -sell, MarketOrderStyle(max(price * 0.98, d.low_limit)))


def gate_on(context):
    """① 中证1000：ER20 > 0.2 且 当前价 > MA20"""
    n = max(Cfg.ER_N, Cfg.MA_N)
    h = attribute_history(Cfg.GATE_INDEX, n, "1d", ["close"], skip_paused=True, df=False)["close"]
    b = get_price(Cfg.GATE_INDEX, end_date=context.current_dt, frequency="1m", fields=["close"], count=1)
    if len(h) < n or len(b) == 0:
        return False
    s = np.append(h, float(b["close"].iloc[-1]))
    path = np.abs(np.diff(s[-(Cfg.ER_N + 1):])).sum()
    er = (s[-1] - s[-(Cfg.ER_N + 1)]) / path if path > 0 else 0.0
    ma = s[-Cfg.MA_N:].mean()
    return er > Cfg.ER_MIN and s[-1] > ma


def build_pool(context, cd):
    """② 股票池"""
    day = context.previous_date
    secs = get_all_securities(["stock"], date=day)
    secs = secs[secs.start_date <= day - pd.Timedelta(days=Cfg.MIN_LIST_DAYS)]
    stocks = [s for s in secs.index if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
    stocks = [s for s in stocks if (not cd[s].is_st) and (not cd[s].paused) and "退" not in cd[s].name]
    amt = get_price(stocks, end_date=day, count=Cfg.AMT_N, fields=["money"], panel=False)
    amt = amt.pivot(index="time", columns="code", values="money").mean()
    amt = amt[amt > 0]
    keep = amt[amt.rank(pct=True) > Cfg.AMT_DROP].index
    return list(keep)


def score_pool(context, cd, pool):
    """③ 小市值排名 + 反转排名"""
    day = context.previous_date
    q = query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(pool))
    cap = get_fundamentals(q, date=day).dropna().set_index("code")["circulating_market_cap"]
    cap = cap[cap > 0]
    codes = list(cap.index)
    px = get_price(codes, end_date=day, count=Cfg.REV_N, fields=["close"], fq="pre", panel=False)
    px = px.pivot(index="time", columns="code", values="close")
    base = px.iloc[0]                                       # 20 个交易日前的收盘（今天算第 20 天）
    cur = pd.Series({s: cd[s].last_price for s in px.columns})
    ret = (cur / base - 1).replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({"cap": cap, "ret": ret}).dropna()
    sc = (-np.log(df["cap"])).rank(pct=True) + (-df["ret"]).rank(pct=True)
    return sc.sort_values(ascending=False)


def select(context, sc):
    """④ 前 20 只，已持有的排名在前 40 继续拿"""
    codes = list(sc.index)
    rank = {s: i for i, s in enumerate(codes)}
    pos = context.portfolio.positions
    held = [s for s in pos if pos[s].total_amount > 0 and s != Cfg.CASH_ETF]
    pick = [s for s in sorted(held, key=lambda x: rank.get(x, 10 ** 9)) if rank.get(s, 10 ** 9) < Cfg.BUFFER][:Cfg.HOLD_N]
    for s in codes:
        if len(pick) >= Cfg.HOLD_N:
            break
        if s not in pick:
            pick.append(s)
    return pick


def cash_etf(context, cd, keep_for_stocks):
    """股票以外的资金放货币 ETF；keep_for_stocks=True 时先把货币 ETF 卖掉腾钱"""
    e = Cfg.CASH_ETF
    if cd[e].paused:
        return
    pf = context.portfolio
    if keep_for_stocks:
        if e in pf.positions and pf.positions[e].total_amount > 0:
            order_target(e, 0)
    else:
        spare = pf.available_cash - pf.total_value * Cfg.CASH_BUFFER
        if spare > 10000:
            order_value(e, spare)


# ---------------- 主流程 ----------------
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    on = gate_on(context)
    if not on:
        if g.gate:
            log.info("开关关闭：中证1000 趋势结束，清仓")
        for s in list(pos.keys()):
            if s != Cfg.CASH_ETF and pos[s].total_amount > 0:
                place(context, cd, s, 0)
        g.picks = []
        cash_etf(context, cd, keep_for_stocks=False)
    else:
        just_on = not g.gate
        if just_on or g.day % Cfg.REBAL_DAYS == 0:
            if just_on:
                log.info("开关打开：中证1000 趋势成立，建仓")
            cash_etf(context, cd, keep_for_stocks=True)
            pool = build_pool(context, cd)
            g.picks = select(context, score_pool(context, cd, pool))
            # 先卖
            for s in list(pos.keys()):
                if s != Cfg.CASH_ETF and pos[s].total_amount > 0 and s not in g.picks:
                    place(context, cd, s, 0)
            # 再买：货币 ETF 当天卖出的钱可以立即用于买入
            total = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER)
            tv = total / Cfg.HOLD_N
            for s in g.picks:
                cur = pos[s].value if s in pos else 0
                if cur < tv * 0.9:
                    place(context, cd, s, tv)
            record(持股数=len([s for s in pos if pos[s].total_amount > 0 and s != Cfg.CASH_ETF]))
    record(开关=100 if on else 0)
    g.gate = on
    g.day += 1
