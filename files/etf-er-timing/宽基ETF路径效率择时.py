# -*- coding: utf-8 -*-
"""
宽基 ETF 路径效率择时 V1.0 —— 聚宽回测代码

四只宽基 ETF，资金各占 1/4，各自按自己跟踪的指数决定持有还是空仓：
    沪深300  510300.XSHG  ← 000300.XSHG
    中证500  510500.XSHG  ← 000905.XSHG
    中证1000 512100.XSHG  ← 000852.XSHG（2016-11 上市，之前这一份资金空仓）
    创业板   159915.XSHE  ← 399006.XSHE

信号（每个交易日 14:50 用指数当前价计算，当场下单）：
    路径效率 ER = (当前价 − 20 个交易日前收盘) ÷ 这 20 天每日涨跌幅绝对值之和（今天用当前价）
    20 日均线 MA20 = 前 19 个交易日收盘 + 当前价 的平均
    ER > 0.2 且 当前价 > MA20  → 持有这只 ETF（目标市值 = 总资产 × 1/4）
    否则                       → 卖出，这份资金空仓
空仓资金：默认买货币 ETF（银华日利 511880）吃货币利率；设 Cfg.CASH_ETF = None 则放现金。

本地测算（指数全收益，2011-01 至 2026-11，当日收盘成交，单边成本 0.05%，空仓计隔夜 Shibor）：
    四指数等权：年化 +10.7%，最大回撤 -16.8%，夏普 0.75（四指数买入持有：+5.0% / -62%）
    2023 年起（样本外）：年化 +9.2%，最大回撤 -13.6%
    单个指数年化：沪深300 +5.3%、中证500 +9.2%、中证1000 +11.1%、创业板 +15.9%；平均约 28% 时间持有
    参数附近（ER 窗口 10–20、阈值 0–0.3、均线 20/40）都有效；晚一天成交每年少 2–4 个点，所以必须当日 14:50 下单
注意：需用「分钟」频率回测；ETF 上市较晚，建议回测从 2017-01-01 开始（四只都已上市）。
      聚宽 ETF 价格不含分红再投资、有跟踪误差和申赎溢价，结果会和本地指数测算有差异。
"""
import builtins
from jqdata import *
import numpy as np
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "宽基ETF路径效率择时 V1.0"


class Cfg:
    PAIRS = [  # (ETF, 信号指数, 名称)
        ("510300.XSHG", "000300.XSHG", "沪深300"),
        ("510500.XSHG", "000905.XSHG", "中证500"),
        ("512100.XSHG", "000852.XSHG", "中证1000"),
        ("159915.XSHE", "399006.XSHE", "创业板"),
    ]
    ER_N = 20               # 路径效率窗口（交易日）
    ER_MIN = 0.20           # 路径效率阈值
    MA_N = 20               # 均线天数
    TRADE_TIME = "14:50"
    REBAL_TOL = 0.20        # 已持有的 ETF 偏离目标市值超过 20% 才调整，减少零碎交易
    CASH_ETF = "511880.XSHG"  # 空仓资金买入的货币 ETF；None = 不处理
    CASH_BUFFER = 0.01      # 留 1% 现金应付手续费


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type="fund")
    set_slippage(PriceRelatedSlippage(0.0005), type="fund")
    log.set_level("order", "error")
    g.sig = {}
    run_daily(trade, time=Cfg.TRADE_TIME)
    log.info(VERSION)


def index_now(idx, context):
    """指数当前价：取截至当前的最后一根 1 分钟 K 线收盘价。"""
    b = get_price(idx, end_date=context.current_dt, frequency="1m", fields=["close"], count=1, skip_paused=True)
    return float(b["close"].iloc[-1]) if len(b) else np.nan


def signal(idx, context):
    """返回 (是否持有, ER, 当前价, MA20)。数据不足返回 (None, …)。"""
    n = max(Cfg.ER_N, Cfg.MA_N)
    h = attribute_history(idx, n, "1d", ["close"], skip_paused=True, df=False)["close"]
    cur = index_now(idx, context)
    if len(h) < n or np.isnan(cur) or np.isnan(h).any():
        return None, np.nan, cur, np.nan
    s = np.append(h, cur)                                   # 最后一个元素 = 今天当前价
    path = np.abs(np.diff(s[-(Cfg.ER_N + 1):])).sum()
    er = (s[-1] - s[-(Cfg.ER_N + 1)]) / path if path > 0 else 0.0
    ma = s[-Cfg.MA_N:].mean()
    return (er > Cfg.ER_MIN and cur > ma), er, cur, ma


def listed(code, context):
    info = get_security_info(code)
    return info is not None and info.start_date <= context.current_dt.date()


def trade(context):
    cd = get_current_data(); pf = context.portfolio
    slot = pf.total_value / len(Cfg.PAIRS)
    targets, msgs = {}, []
    for etf, idx, name in Cfg.PAIRS:
        if not listed(etf, context):
            targets[etf] = 0.0; msgs.append(f"{name}:未上市"); continue
        hold, er, cur, ma = signal(idx, context)
        if hold is None:
            targets[etf] = pf.positions[etf].value if etf in pf.positions else 0.0
            msgs.append(f"{name}:数据不足"); continue
        targets[etf] = slot if hold else 0.0
        if g.sig.get(etf) != hold:
            log.info("%s %s：ER %.2f，现价 %.2f vs MA%d %.2f → %s" % (name, etf, er, cur, Cfg.MA_N, ma, "持有" if hold else "空仓"))
        g.sig[etf] = hold
        msgs.append(f"{name}:{'持' if hold else '空'}")
    # 1. 先卖：信号转空、或超配
    for etf, tv in targets.items():
        cur_v = pf.positions[etf].value if etf in pf.positions else 0.0
        if cd[etf].paused:
            continue
        if tv == 0 and cur_v > 0:
            order_target_value(etf, 0)
        elif tv > 0 and cur_v > tv * (1 + Cfg.REBAL_TOL):
            order_target_value(etf, tv)
    # 2. 货币 ETF：给要买入的部分腾出资金，剩余闲置资金买入
    need = sum(max(0.0, tv - (pf.positions[e].value if e in pf.positions else 0.0)) for e, tv in targets.items()
               if tv > 0 and not cd[e].paused and ((pf.positions[e].value if e in pf.positions else 0.0) < tv * (1 - Cfg.REBAL_TOL)))
    if Cfg.CASH_ETF and listed(Cfg.CASH_ETF, context) and not cd[Cfg.CASH_ETF].paused:
        invest = sum(tv for tv in targets.values())
        cash_target = max(0.0, pf.total_value - invest - pf.total_value * Cfg.CASH_BUFFER)
        cur_cash_etf = pf.positions[Cfg.CASH_ETF].value if Cfg.CASH_ETF in pf.positions else 0.0
        if cur_cash_etf > cash_target + 2000 or need > pf.available_cash:
            order_target_value(Cfg.CASH_ETF, max(0.0, min(cash_target, cur_cash_etf - max(0.0, need - pf.available_cash))))
    # 3. 再买：信号转多、或低配
    for etf, tv in targets.items():
        if tv <= 0 or cd[etf].paused:
            continue
        cur_v = pf.positions[etf].value if etf in pf.positions else 0.0
        if cur_v < tv * (1 - Cfg.REBAL_TOL):
            v = min(tv, cur_v + pf.available_cash)
            if v > cur_v + 1000:
                order_target_value(etf, v)
    # 4. 剩余闲置现金买货币 ETF
    if Cfg.CASH_ETF and listed(Cfg.CASH_ETF, context) and not cd[Cfg.CASH_ETF].paused:
        spare = pf.available_cash - pf.total_value * Cfg.CASH_BUFFER
        if spare > 10000:
            order_value(Cfg.CASH_ETF, spare)
    hold = [n for (e, i, n) in Cfg.PAIRS if e in pf.positions and pf.positions[e].total_amount > 0]
    log.info("信号 %s | 持有：%s" % (" ".join(msgs), "、".join(hold) or "无"))
