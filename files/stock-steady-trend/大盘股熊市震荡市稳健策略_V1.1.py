# -*- coding: utf-8 -*-
"""
大盘股熊市震荡市稳健策略 V1.1 —— 聚宽回测代码（原名「个股稳健趋势完整版」，方向 A：选股 + 轮换 + 仓位 + 风控）

V1.1 修改（相对 V1.0）：
    1. 所有委托改为带保护价的市价单：买入保护价 = 当前价 × 1.02（不超过涨停价），卖出 = 当前价 × 0.98（不低于跌停价）。
       修复科创板（688）报错「科创板市价单需要指定保护限价」，之前科创板股票一直买不进。
    2. 下单前按 100 股取整：调整量不足 1 手就不下单（清仓卖出除外）。修复贵州茅台这类高价股
       小幅加仓时报错「开仓数量不能小于 100」。
    3. 策略更名为「大盘股熊市震荡市稳健策略」，规则和参数不变。

① 股票池（每次调仓时重算）
    上市满 365 天、非 ST、非停牌、非北交所 → 流通市值前 800（≈中证800）
    → 最新报告期 净利润 > 0 且 年化 ROE > 5%
② 选股打分
    距一年高点 = 当前价 / 近 250 日最高收盘价（含当前价）  → 截面百分位排名
    低波动     = 近 60 日日收益率标准差，越低越好          → 截面百分位排名
    两个排名相加，取前 30 只，等权
③ 轮换
    每 5 个交易日调仓一次；已持有的股票只要排名还在前 60 就不卖（缓冲区，降换手）
    同一申万二级行业最多 5 只；当前价涨停不买、跌停/停牌不卖
④ 总仓位（每天 14:50 检查）
    波动率目标：总仓位 = min(1, 15% / 选股组合近 40 日年化波动)；与当前仓位相差 ≥10% 才调整
⑤ 风控（每天 14:50 检查）
    单只：较持仓成本亏损 > 10% 当天卖出，空出的仓位留现金到下次调仓
    组合：选股组合「满仓影子净值」较近 250 日高点回撤 > 15% → 总仓位再乘 0.5；> 25% → 空仓
          （用影子净值而不是账户净值判断，空仓期间影子净值照样更新，行情恢复后能自动回来）

本地测算（后复权日线 2015-01 至 2026-11，单边成本 0.15%，当日收盘成交，3 份资金错开调仓取平均）：
    完整版：年化 +13.1%，最大回撤 -21.6%，夏普 0.90
    样本内 2015–2022：+10.4% / -21.6%；样本外 2023 起：+18.7% / -10.5%
    基线（只有打分、满仓）：+10.6% / -42.5%；同期沪深300 价格指数 +0.8% / -46.7%
    参数附近（波动率目标 12–20%、窗口 20–60 日、持股 20–30 只、缓冲 60–100、晚一天成交）都在 12.6%–14.0%
已验证无效、所以没加的：跌破 MA20/MA60 止损（年化掉到 2–6%）、指数择时（2023 年后收益几乎清零）
与本地口径的差异：
    聚宽 indicator.roe 是单季 ROE，这里 ×4 年化；本地用的是累计 ROE 按季数年化，结果会略有不同
    行业：本地用东方财富二级行业，这里用申万二级行业
建议回测：2015-01-01 起，日频或分钟频率均可（14:50 下单），初始资金 ≥ 100 万（30 只股票每只至少一手）
"""
import builtins
from jqdata import *
import numpy as np
import pandas as pd
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "大盘股熊市震荡市稳健策略 V1.1"


class Cfg:
    POOL_SIZE = 800         # 流通市值前 N
    MIN_LIST_DAYS = 365     # 上市满多少天
    ROE_MIN = 0.05          # 年化 ROE 门槛
    HIGH_N = 250            # 一年高点窗口
    VOL_N = 60              # 低波动窗口
    HOLD_N = 30             # 持股数
    BUFFER = 60             # 已持有股票排名在前 BUFFER 内就继续拿
    REBAL_DAYS = 5          # 每几个交易日调仓
    IND_MAX = 5             # 同一申万二级行业最多几只
    STOP_LOSS = 0.10        # 单只亏损超过 10% 卖出
    VOL_TARGET = 0.15       # 目标年化波动
    VOL_WIN = 40            # 估算组合波动的天数
    EXPO_BAND = 0.10        # 总仓位变动 ≥ 10% 才调整
    DD_HALF = 0.15          # 影子净值回撤超过 15% 半仓
    DD_OUT = 0.25           # 超过 25% 空仓
    DD_LOOK = 250           # 回撤按近 250 日高点算
    TRADE_TIME = "14:50"
    CASH_BUFFER = 0.01      # 留 1% 现金应付手续费


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type="stock")
    set_slippage(PriceRelatedSlippage(0.002), type="stock")
    log.set_level("order", "error")
    g.day = 0              # 交易日计数
    g.picks = []           # 当前选股组合（目标持仓）
    g.sleeve = []          # 选股组合每日等权收益（满仓口径），用于波动率目标和影子净值
    g.expo = 0.0           # 当前执行中的总仓位
    run_daily(trade, time=Cfg.TRADE_TIME)
    run_daily(after_close, time="after_close")
    log.info(VERSION)


# ---------------- 工具 ----------------
def can_buy(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price < d.high_limit - 1e-6


def can_sell(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price > d.low_limit + 1e-6


def target_exposure():
    """④ 波动率目标 × ⑤ 组合回撤控制"""
    r = np.array(g.sleeve[-Cfg.VOL_WIN:])
    if len(r) < 20:
        e = 1.0
    else:
        vol = np.std(r, ddof=1) * np.sqrt(252)
        e = 1.0 if vol <= 0 else min(1.0, Cfg.VOL_TARGET / vol)
    if len(g.sleeve) > 0:
        nav = np.cumprod(1 + np.array(g.sleeve))
        peak = np.max(nav[-Cfg.DD_LOOK:])
        dd = nav[-1] / peak - 1
        if dd < -Cfg.DD_OUT:
            e = 0.0
        elif dd < -Cfg.DD_HALF:
            e *= 0.5
    return e


def build_pool(context, cd):
    """① 股票池"""
    day = context.previous_date
    secs = get_all_securities(["stock"], date=day)
    secs = secs[(secs.start_date <= day - pd.Timedelta(days=Cfg.MIN_LIST_DAYS))]
    stocks = [s for s in secs.index if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
    stocks = [s for s in stocks if (not cd[s].is_st) and (not cd[s].paused) and "退" not in cd[s].name]
    q = query(valuation.code, valuation.circulating_market_cap, indicator.roe, income.net_profit
              ).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=day).dropna(subset=["circulating_market_cap"])
    df = df.sort_values("circulating_market_cap", ascending=False).head(Cfg.POOL_SIZE)
    df = df[(df.net_profit > 0) & (df.roe * 4 / 100 > Cfg.ROE_MIN)]
    return list(df.code)


def score_pool(context, cd, pool):
    """② 打分：距一年高点排名 + 低波动排名"""
    px = get_price(pool, end_date=context.previous_date, count=Cfg.HIGH_N - 1, fields=["close"],
                   fq="pre", panel=False).pivot(index="time", columns="code", values="close")
    cur = pd.Series({s: cd[s].last_price for s in px.columns})
    px = pd.concat([px, cur.to_frame().T])
    px = px.loc[:, px.notna().sum() >= 200]
    hi = px.iloc[-1] / px.max()
    vol = px.pct_change().iloc[-Cfg.VOL_N:].std()
    sc = hi.rank(pct=True) + (-vol).rank(pct=True)
    return sc.dropna().sort_values(ascending=False)


def select(context, sc):
    """③ 轮换：缓冲区 + 行业上限"""
    codes = list(sc.index)
    rank = {s: i for i, s in enumerate(codes)}
    ind = get_industry(codes + [s for s in g.picks if s not in rank], date=context.previous_date)
    def ind_of(s):
        return ind.get(s, {}).get("sw_l2", {}).get("industry_name", "未知")
    cnt, pick = {}, []
    def add(s):
        pick.append(s); k = ind_of(s); cnt[k] = cnt.get(k, 0) + 1
    held = [s for s in context.portfolio.positions if context.portfolio.positions[s].total_amount > 0]
    for s in sorted(held, key=lambda x: rank.get(x, 10 ** 9)):
        if rank.get(s, 10 ** 9) < Cfg.BUFFER and len(pick) < Cfg.HOLD_N and cnt.get(ind_of(s), 0) < Cfg.IND_MAX:
            add(s)
    for s in codes:
        if len(pick) >= Cfg.HOLD_N:
            break
        if s not in pick and cnt.get(ind_of(s), 0) < Cfg.IND_MAX:
            add(s)
    return pick


def place(context, cd, s, target_value):
    """V1.1：带保护价的市价单 + 按手取整。target_value = 0 表示清仓。"""
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


def rebalance_to(context, cd, picks, expo):
    total = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER)
    tv = total * expo / Cfg.HOLD_N
    pos = context.portfolio.positions
    # 先卖
    for s in list(pos.keys()):
        if pos[s].total_amount == 0:
            continue
        if s not in picks:
            place(context, cd, s, 0)
        elif pos[s].value > tv * 1.05:
            place(context, cd, s, tv)
    # 再买
    for s in picks:
        cur = pos[s].value if s in pos else 0
        if cur < tv * 0.95:
            place(context, cd, s, tv)


# ---------------- 主流程 ----------------
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    # ⑤ 单只止损
    for s in list(pos.keys()):
        p = pos[s]
        if p.total_amount > 0 and p.avg_cost > 0 and cd[s].last_price / p.avg_cost - 1 < -Cfg.STOP_LOSS:
            if can_sell(cd, s):
                place(context, cd, s, 0)
                if s in g.picks:
                    g.picks.remove(s)
                log.info("止损 %s %s 亏损 %.1f%%" % (s, cd[s].name, (cd[s].last_price / p.avg_cost - 1) * 100))
    expo = target_exposure()
    if g.day % Cfg.REBAL_DAYS == 0:
        pool = build_pool(context, cd)
        if len(pool) >= Cfg.HOLD_N:
            g.picks = select(context, score_pool(context, cd, pool))
        rebalance_to(context, cd, g.picks if expo > 0 else [], expo)
        g.expo = expo
        record(总仓位=expo * 100, 持股数=len([s for s in pos if pos[s].total_amount > 0]))
    elif abs(expo - g.expo) >= Cfg.EXPO_BAND or (expo == 0 and g.expo > 0):
        # 只调整总仓位：按新比例缩放已有目标持仓（止损卖掉的不补）
        rebalance_to(context, cd, [s for s in g.picks if s in pos and pos[s].total_amount > 0] if expo > 0 else [], expo)
        g.expo = expo
        log.info("总仓位调整为 %.0f%%" % (expo * 100))
    g.day += 1


def after_close(context):
    """记录选股组合当日等权收益（满仓影子口径）"""
    if not g.picks:
        g.sleeve.append(0.0)
        return
    px = get_price(g.picks, end_date=context.current_dt.date(), count=2, fields=["close"], fq="pre", panel=False)
    px = px.pivot(index="time", columns="code", values="close")
    if len(px) < 2:
        g.sleeve.append(0.0)
        return
    r = (px.iloc[-1] / px.iloc[-2] - 1).dropna()
    g.sleeve.append(float(r.mean()) if len(r) else 0.0)
    g.sleeve = g.sleeve[-400:]
