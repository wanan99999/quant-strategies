# -*- coding: utf-8 -*-
"""
石油石化板块顺势策略 —— 聚宽回测代码（由银行股顺势策略移植）

股票池：每年第一个交易日，取「上一年最后一个交易日」总市值前 20 的股票（申万一级 石油石化 801960），全年不变
排序：  距 250 日高点 = 当前价 ÷ 最近 250 个交易日（含今天）最高收盘价 - 1，越接近 0 越强
        （复权价计算，除息不会被误算成下跌；上市不足 250 个交易日的不参与）
调仓：  每 5 个交易日一次，14:50 用当前价近似收盘价成交
卖出：  持仓掉出前 6 名才卖出；两次调仓之间不止损
        持仓不在当年股票池（年初换池被剔除）也在调仓日卖出
买入：  按排名依次补满 4 只；新买入必须满足「近 20 个交易日涨幅 ≤ 10%」，不满足的跳过看下一名
        涨停、停牌、ST 的跳过；满足条件的不足时，剩余仓位空着等下次调仓
仓位：  每只 1/4；调仓日把继续持有的股票也调回 1/4（REBALANCE_KEPT=True）

本地测算（2008-01~2026-11，后复权含分红，每 5 日调仓，5 个调仓起始日中位数）：
  19 年年化 +2.6%；2008-2016 年化 -4.3%（无效）；2017 年起 +11.5%；2019 年起 +18.6%（5 个起始日 +13.6%~+21.4%，最大回撤 -41%）；2023 年起 +2.9%（跑输等权）
  同期池内 20 只等权持有：2017 年起 +2.3%，2019 年起 +8.6%，2023 年起 +7.1%
与银行版规则相同（持 4 只、掉出前 6 名才卖、20 日涨幅 ≤10%）；起始日之间差异较大，建议改 REBAL_OFFSET 多跑几次
成份股按申万当前成份回测（有幸存者偏差）；聚宽按历史日期取行业成份，结果会有差异。
注意：结果对「从哪一天开始数 5 个交易日」仍有一定敏感，可改 REBAL_OFFSET（0~4）多跑几次看区间。
（原 20 日调仓版本对起始日更敏感：如石油石化 2023 年起 20 个起始日结果在 -2%~+78% 之间。）
需要用「分钟」频率回测（14:50 下单）；建议回测区间 2019-01-01 起。
"""
import builtins
from jqdata import *
import pandas as pd
import numpy as np
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "石油石化板块顺势策略"


class Cfg:
    INDUSTRY = "801960"     # 申万一级 石油石化 801960
    POOL_SIZE = 20          # 上年末总市值前 N
    LOOKBACK = 250          # 距 N 日高点
    HOLD = 4                # 持股只数
    KEEP_RANK = 6           # 持仓掉出前 N 名才卖
    CHASE_DAYS = 20         # 防追高：近 N 日涨幅
    CHASE_MAX = 0.10        # 防追高：涨幅上限
    REBAL_DAYS = 5          # 每 N 个交易日调仓（原为 20）
    REBAL_OFFSET = 0        # 第几个交易日开始第一次调仓（0 = 回测第一天）
    REBALANCE_KEPT = True   # 调仓日是否把继续持有的股票调回 1/HOLD
    TRADE_TIME = "14:50"


def initialize(context):
    set_benchmark("000300.XSHG")            # 沪深300
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025,
                             close_commission=0.00025, min_commission=5), type="stock")
    set_slippage(PriceRelatedSlippage(0.001), type="stock")
    log.set_level("order", "error")
    g.pool, g.pool_year, g.n = [], None, 0
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
                         paused=cd[c].paused, up_limit=cur >= cd[c].high_limit, st=cd[c].is_st))
    df = pd.DataFrame(rows)
    return df.sort_values("dd", ascending=False).reset_index(drop=True) if len(df) else df


def trade(context):
    g.n += 1
    if (g.n - 1 - Cfg.REBAL_OFFSET) % Cfg.REBAL_DAYS != 0 or g.n - 1 < Cfg.REBAL_OFFSET:
        return
    sig = signals(context)
    if sig.empty:
        return
    rank = {c: i for i, c in enumerate(sig.code)}
    held = [c for c, p in context.portfolio.positions.items() if p.total_amount > 0]
    keep = [c for c in held if c in rank and rank[c] < Cfg.KEEP_RANK]
    cd = get_current_data()
    # 卖出
    for c in held:
        if c in keep:
            continue
        if cd[c].paused:
            keep.append(c); log.info("停牌无法卖出，继续持有 %s" % c); continue
        why = "不在股票池" if c not in rank else "排名第 %d" % (rank[c] + 1)
        order_target_value(c, 0); log.info("卖出 %s %s（%s）" % (c, cd[c].name, why))
    # 买入
    need = Cfg.HOLD - len(keep); new = []
    for _, r in sig.iterrows():
        if need <= 0:
            break
        if r.code in keep or r.paused or r.up_limit or r.st:
            continue
        if r.ret > Cfg.CHASE_MAX:
            log.info("跳过 %s %s：近 %d 日涨幅 %.1f%% > %.0f%%" % (r.code, r.name, Cfg.CHASE_DAYS, r.ret * 100, Cfg.CHASE_MAX * 100)); continue
        new.append(r.code); need -= 1
    tv = context.portfolio.total_value / Cfg.HOLD
    if Cfg.REBALANCE_KEPT:                                   # 先减仓超配的，再补低配的
        for c in sorted(keep, key=lambda c: -context.portfolio.positions[c].value):
            if not cd[c].paused:
                order_target_value(c, tv)
    for c in new:
        v = min(tv, context.portfolio.available_cash)
        if v > 0:
            order_target_value(c, v)
            log.info("买入 %s %s 排名第 %d 距高点 %.1f%% 近20日 %+.1f%%" % (c, cd[c].name, rank[c] + 1,
                     sig.dd[rank[c]] * 100, sig.ret[rank[c]] * 100))
    top = " ".join("%d.%s(%.1f%%)" % (i + 1, sig.name[i], sig.dd[i] * 100) for i in range(min(8, len(sig))))
    hold = [cd[c].name for c, p in context.portfolio.positions.items() if p.total_amount > 0]
    log.info("调仓 #%d | 排名：%s | 持仓：%s" % ((g.n - 1 - Cfg.REBAL_OFFSET) // Cfg.REBAL_DAYS + 1, top, " ".join(hold)))
