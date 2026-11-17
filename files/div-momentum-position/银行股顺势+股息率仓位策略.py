# -*- coding: utf-8 -*-
"""
银行股顺势 + 股息率定仓位策略 —— 聚宽回测代码

在「银行股顺势策略（最终版，5 日调仓）」上加一层仓位控制：买哪几只由顺势决定，买多少由股息率决定。

股票池：每年第一个交易日，取「上一年最后一个交易日」总市值前 20 的银行股（申万一级 银行 801780），全年不变
仓位：  池平均股息率 ≤ 3.0% 空仓，≥ 5.0% 满仓，中间按 0.2 个百分点一档、每档 10% 线性加减
        个股股息率 = 最近 240 个交易日（不含今天）已除息的每股现金分红合计 ÷ 昨日收盘价
        （数据：finance.STK_XR_XD；送转股按除权后股本折算；当年未分红的计为 0）
        池平均股息率 = 20 只个股股息率的简单平均
选股：  距 250 日高点 = 当前价 ÷ 最近 250 个交易日（含今天）最高收盘价 - 1，越接近 0 越强
        持 4 只；持仓掉出前 6 名才卖；新买入须「近 20 个交易日涨幅 ≤ 10%」；涨停、停牌跳过
        即使仓位为 0，也照常维护「选中的 4 只」名单，仓位恢复时直接买入名单上的股票
调仓：  每 5 个交易日一次，14:50 用当前价近似收盘价成交；每只目标市值 = 总资产 × 仓位 ÷ 4
空仓资金：默认不做任何处理；可把 Cfg.CASH_ETF 设为货币 ETF（如 "511880.XSHG" 银华日利）让闲置资金吃货币利率

本地测算（2008-01~2026-11，后复权含分红，5 个调仓起始日中位数，空仓资金不计息）：
  全程年化 +10.6%，最大回撤 -17%（同规则满仓：+5.5% / -69%；池内 20 家等权：+4.6% / -69%）
  2008-2016 +7.6%（满仓 -2.8%）；2017 年起 +13.0%（满仓 +13.9%）；2019 年起 +16.4%（满仓 +17.3%）；2023 年起 +26.1%（满仓 +25.9%）
  闲置资金按年化 2% 计息：全程 +11.8%，2019 年起 +17.6%
  阈值敏感度：2%空→4%满 全程 +11.6%/-25%；2.5%→4.5% +11.2%/-20%；3.5%→5.5% +9.1%/-16%
  主要作用是在股息率低（估值贵）时空仓，躲开 2008、2015-16、2018 年的下跌；代价是错过 2009、2017、2019 年的上涨。
注意：需用「分钟」频率回测（14:50 下单）。可改 REBAL_OFFSET（0~4）多跑几次看区间。
      聚宽的分红数据（STK_XR_XD）与本地推算口径略有差异，股息率在阈值附近时仓位可能差一档。
"""
import builtins
from jqdata import *
import pandas as pd
import numpy as np
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "银行股顺势+股息率仓位策略 V1.0"


class Cfg:
    INDUSTRY = "801780"     # 申万一级：银行
    POOL_SIZE = 20          # 上年末总市值前 N
    LOOKBACK = 250          # 距 N 日高点
    HOLD = 4                # 持股只数
    KEEP_RANK = 6           # 持仓掉出前 N 名才卖
    CHASE_DAYS = 20         # 防追高：近 N 日涨幅
    CHASE_MAX = 0.10        # 防追高：涨幅上限
    REBAL_DAYS = 5          # 每 N 个交易日调仓
    REBAL_OFFSET = 0        # 第几个交易日开始第一次调仓（0 = 回测第一天）
    TRADE_TIME = "14:50"
    # —— 股息率仓位 ——
    YIELD_DAYS = 240        # 股息率统计窗口（交易日）
    YIELD_EMPTY = 0.030     # 池平均股息率 ≤ 此值：空仓
    YIELD_FULL = 0.050      # 池平均股息率 ≥ 此值：满仓
    POS_STEP = 0.10         # 仓位档位（10% 一档）
    CASH_ETF = None         # 闲置资金买入的货币 ETF，如 "511880.XSHG"；None = 不处理


def initialize(context):
    set_benchmark("399986.XSHE")            # 中证银行；也可改 000300.XSHG
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025,
                             close_commission=0.00025, min_commission=5), type="stock")
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0, close_commission=0, min_commission=0), type="fund")
    set_slippage(PriceRelatedSlippage(0.001), type="stock")
    log.set_level("order", "error")
    g.pool, g.pool_year, g.n, g.picks, g.w = [], None, 0, [], None
    run_daily(update_pool, time="before_open")
    run_daily(trade, time=Cfg.TRADE_TIME)
    log.info(VERSION)


def update_pool(context):
    y = context.current_dt.year
    if g.pool_year == y and g.pool:
        return
    d = get_trade_days(end_date="%d-12-31" % (y - 1), count=1)[0]
    banks = get_industry_stocks(Cfg.INDUSTRY, date=d)
    df = get_fundamentals(query(valuation.code, valuation.market_cap).filter(valuation.code.in_(banks)), date=d)
    df = df.sort_values("market_cap", ascending=False).head(Cfg.POOL_SIZE)
    g.pool, g.pool_year = list(df.code), y
    log.info("%d 年股票池（%s 总市值前 %d）：%s" % (y, d, Cfg.POOL_SIZE,
             " ".join("%s%.0f亿" % (get_security_info(c).display_name, m) for c, m in zip(df.code, df.market_cap))))


# ---------------- 股息率 → 仓位 ----------------
def pool_yield(context):
    """返回 (池平均股息率, {code: 个股股息率})。窗口 = 截至昨日的最近 YIELD_DAYS 个交易日。"""
    days = get_trade_days(end_date=context.previous_date, count=Cfg.YIELD_DAYS)
    d0, d1 = pd.Timestamp(days[0]), pd.Timestamp(days[-1])
    q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.a_xr_date, finance.STK_XR_XD.bonus_ratio_rmb,
              finance.STK_XR_XD.dividend_ratio, finance.STK_XR_XD.transfer_ratio).filter(
        finance.STK_XR_XD.code.in_(g.pool), finance.STK_XR_XD.a_xr_date != None,
        finance.STK_XR_XD.a_xr_date >= d0.strftime("%Y-%m-%d"))
    dv = finance.run_query(q)
    if len(dv):
        dv["xr"] = pd.to_datetime(dv["a_xr_date"])
        dv = dv[dv.xr <= d1].copy()
        for c in ["bonus_ratio_rmb", "dividend_ratio", "transfer_ratio"]:
            dv[c] = dv[c].fillna(0).astype(float)
        dv["bonus"] = (dv["dividend_ratio"] + dv["transfer_ratio"]) / 10.0
    ys = {}
    for c in g.pool:
        h = attribute_history(c, 1, "1d", ["close"], skip_paused=False, df=False, fq=None)
        px = float(h["close"][-1]) if len(h["close"]) else np.nan
        if not px or np.isnan(px):
            continue                                         # 尚未上市
        dps = 0.0
        if len(dv):
            x = dv[dv.code == c].sort_values("xr")
            for _, r in x.iterrows():
                if r.bonus_ratio_rmb <= 0:
                    continue
                factor = 1.0                                 # 本次及之后的送转，把每股分红折算到当前股本
                for b in x[(x.xr >= r.xr) & (x.bonus > 0)].bonus:
                    factor *= 1 + b
                dps += r.bonus_ratio_rmb / 10.0 / factor
        ys[c] = dps / px
    return (float(np.mean(list(ys.values()))) if ys else np.nan), ys


def target_weight(y):
    if y is None or np.isnan(y):
        return 1.0
    w = (y - Cfg.YIELD_EMPTY) / (Cfg.YIELD_FULL - Cfg.YIELD_EMPTY)
    w = min(1.0, max(0.0, w))
    return round(w / Cfg.POS_STEP) * Cfg.POS_STEP


# ---------------- 顺势排序 ----------------
def signals(context):
    """返回 DataFrame：code, dd(距高点), ret(近20日涨幅), paused, up_limit；按 dd 从高到低排序"""
    cd = get_current_data(); rows = []
    for c in g.pool:
        h = attribute_history(c, Cfg.LOOKBACK - 1, "1d", ["close"], skip_paused=False, df=True, fq="pre")
        px = h["close"].values
        if len(px) < Cfg.LOOKBACK - 1 or np.isnan(px).any():
            continue                                         # 上市不足 250 个交易日
        cur = cd[c].last_price
        if not cur or np.isnan(cur):
            continue
        s = np.append(px, cur)
        rows.append(dict(code=c, name=cd[c].name, dd=cur / s.max() - 1, ret=cur / s[-1 - Cfg.CHASE_DAYS] - 1,
                         paused=cd[c].paused, up_limit=cur >= cd[c].high_limit))
    df = pd.DataFrame(rows)
    return df.sort_values("dd", ascending=False).reset_index(drop=True) if len(df) else df


def trade(context):
    g.n += 1
    if (g.n - 1 - Cfg.REBAL_OFFSET) % Cfg.REBAL_DAYS != 0 or g.n - 1 < Cfg.REBAL_OFFSET:
        return
    sig = signals(context)
    if sig.empty:
        return
    cd = get_current_data()
    rank = {c: i for i, c in enumerate(sig.code)}
    # 1. 维护「选中名单」（与仓位无关）
    keep = [c for c in g.picks if c in rank and rank[c] < Cfg.KEEP_RANK]
    held_now = {c for c, p in context.portfolio.positions.items() if p.total_amount > 0 and c != Cfg.CASH_ETF}
    for c in g.picks:                                        # 停牌的股票卖不掉，留在名单里
        if c not in keep and c in held_now and cd[c].paused:
            keep.append(c); log.info("停牌无法卖出，继续持有 %s" % c)
    dropped = [c for c in g.picks if c not in keep]
    need = Cfg.HOLD - len(keep); new = []
    for _, r in sig.iterrows():
        if need <= 0:
            break
        if r.code in keep or r.paused or r.up_limit:
            continue
        if r.ret > Cfg.CHASE_MAX:
            log.info("跳过 %s %s：近 %d 日涨幅 %.1f%% > %.0f%%" % (r.code, r.name, Cfg.CHASE_DAYS, r.ret * 100, Cfg.CHASE_MAX * 100)); continue
        new.append(r.code); need -= 1
    g.picks = keep + new
    # 2. 仓位
    y, ys = pool_yield(context)
    w = target_weight(y)
    if g.w is None or abs(w - g.w) > 1e-9:
        log.info("池平均股息率 %.2f%% → 仓位 %.0f%%（原 %s）" % (y * 100, w * 100, "-" if g.w is None else "%.0f%%" % (g.w * 100)))
    g.w = w
    # 3. 下单：先卖（不在名单的 / 货币 ETF 腾资金），再按目标市值调整
    for c in held_now:
        if c not in g.picks and not cd[c].paused:
            why = "不在股票池" if c not in rank else "排名第 %d" % (rank[c] + 1)
            order_target_value(c, 0); log.info("卖出 %s %s（%s）" % (c, cd[c].name, why))
    tv = context.portfolio.total_value * w / Cfg.HOLD
    if Cfg.CASH_ETF and Cfg.CASH_ETF in context.portfolio.positions:
        need_cash = sum(max(0, tv - (context.portfolio.positions[c].value if c in context.portfolio.positions else 0)) for c in g.picks)
        etf = context.portfolio.positions[Cfg.CASH_ETF].value
        if need_cash > context.portfolio.available_cash and etf > 0:
            order_target_value(Cfg.CASH_ETF, max(0, etf - (need_cash - context.portfolio.available_cash)))
    # 先减后加
    for c in sorted(g.picks, key=lambda c: -(context.portfolio.positions[c].value if c in context.portfolio.positions else 0)):
        if cd[c].paused:
            continue
        cur = context.portfolio.positions[c].value if c in context.portfolio.positions else 0
        if cur > tv:
            order_target_value(c, tv)
    for c in g.picks:
        if cd[c].paused:
            continue
        cur = context.portfolio.positions[c].value if c in context.portfolio.positions else 0
        if cur < tv:
            v = min(tv, cur + context.portfolio.available_cash)
            if v > cur:
                order_target_value(c, v)
                if c in new:
                    log.info("买入 %s %s 排名第 %d 距高点 %.1f%% 近20日 %+.1f%% 股息率 %.2f%%" % (c, cd[c].name, rank[c] + 1,
                             sig.dd[rank[c]] * 100, sig.ret[rank[c]] * 100, ys.get(c, 0) * 100))
    if Cfg.CASH_ETF and context.portfolio.available_cash > 10000 and not cd[Cfg.CASH_ETF].paused:
        order_value(Cfg.CASH_ETF, context.portfolio.available_cash * 0.99)
    top = " ".join("%d.%s(%.1f%%)" % (i + 1, sig.name[i], sig.dd[i] * 100) for i in range(min(8, len(sig))))
    names = " ".join(cd[c].name for c in g.picks)
    log.info("调仓 #%d | 股息率 %.2f%% 仓位 %.0f%% | 选中：%s | 排名：%s" % ((g.n - 1 - Cfg.REBAL_OFFSET) // Cfg.REBAL_DAYS + 1,
             y * 100, w * 100, names, top))
