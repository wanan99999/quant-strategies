# -*- coding: utf-8 -*-
"""
股息率区间轮动策略 V3.1（回测期末数据：2023-12-29 市值前 20 家银行 + 2023 年全年 TTM 最常见区间）

V3.1：对照试验，故意使用回测区间「结束时」的数据选股和定区间，检验收益差异是否来自数据时点。
      与 V2.1 回测同一时间段：V2.1 用的是回测开始前的数据，V3.1 用的是回测结束时的数据。
      股票池 = 2023-12-29 收盘市值前 20 的 A 股银行（市值 = 收盘价 × 当日总股本，含 H 股）；
      区间 = 2023 年全年 每日 TTM 股息率按 0.5% 分档、天数最多的一档（卖出线=下限，买入线=上限）；
      空仓起步，无初始持仓；交易规则、TTM 算法、仓位(2只×50%)、换仓门槛与 V1.2 相同。
      建议回测区间：2023-01-01 ~ 2023-12-31。

V1.2：股息率改为 TTM（滚动 12 个月）口径；其余同 V1.1。
V1.1：换仓门槛 SWITCH_MARGIN 默认 0.005（0.5个百分点），减少来回换仓；其余同 V1.0。

买入：股息率 > 该股「最常见区间」上限（买入线）
卖出：股息率 < 该股「最常见区间」下限（卖出线）
仓位：每只占当前总资产 50%，最多持有 2 只
排序：按「距卖出线距离 = 股息率 - 卖出线」从大到小；
      持仓股(未触发卖出) + 买入候选 一起排序取前 2 名，
      满仓时若候选比持仓更远，则卖出较近的持仓、换入候选。
股息率：TTM 股息率 = 最近 12 个月报告期对应的每股现金分红 ÷ 当前价
        （数据：finance.STK_XR_XD；送转股按除权后股本折算）
        做法：以已除息分红中最新的报告期 R 为终点，取 (R-12个月, R] 这一年；
        每笔分红覆盖「上一笔分红的报告期 ~ 本报告期」，按落在这一年里的比例计入。
        例：工行 2026-10 = 2025 年末期(覆盖 2025H2) + 2026 年中期(覆盖 2026H1)。
        不用「过去 365 天除息」：除息日每年前后漂移，会出现同年分红算两次(股息率翻倍)
        或某笔分红暂时掉出窗口(股息率腰斩)，产生假的买卖信号；
        也避免首次实施中期分红的年份，上年全年末期 + 本年中期被叠加。
成交：分钟回测，每天 14:55 用当前价近似收盘价，计算信号并下单。
"""
import builtins
from jqdata import *
import pandas as pd
# jqdata 会覆盖内置函数，恢复
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "股息率区间轮动策略 V3.1"


class Cfg:
    # 代码: (名称, 卖出线, 买入线)  —— 2023-12-29 市值前 20；2023 年全年 TTM 股息率按0.5%分档的最常见区间
    BANDS = {
        "601398.XSHG": ("工商银行", 0.060, 0.065),   #  1. 市值 17,036 亿，最常见档占 54%
        "601939.XSHG": ("建设银行", 0.060, 0.065),   #  2. 市值 16,276 亿，最常见档占 55%
        "601288.XSHG": ("农业银行", 0.060, 0.065),   #  3. 市值 12,739 亿，最常见档占 50%
        "601988.XSHG": ("中国银行", 0.055, 0.060),   #  4. 市值 11,746 亿，最常见档占 37%
        "600036.XSHG": ("招商银行", 0.040, 0.045),   #  5. 市值 7,016 亿，最常见档占 23%
        "601658.XSHG": ("邮储银行", 0.050, 0.055),   #  6. 市值 4,314 亿，最常见档占 59%
        "601328.XSHG": ("交通银行", 0.060, 0.065),   #  7. 市值 4,263 亿，最常见档占 43%
        "601166.XSHG": ("兴业银行", 0.060, 0.065),   #  8. 市值 3,368 亿，最常见档占 34%
        "601998.XSHG": ("中信银行", 0.055, 0.060),   #  9. 市值 2,590 亿，最常见档占 40%
        "600000.XSHG": ("浦发银行", 0.055, 0.060),   # 10. 市值 1,943 亿，最常见档占 44%
        "000001.XSHE": ("平安银行", 0.015, 0.020),   # 11. 市值 1,822 亿，最常见档占 44%
        "601818.XSHG": ("光大银行", 0.060, 0.065),   # 12. 市值 1,713 亿，最常见档占 54%
        "600016.XSHG": ("民生银行", 0.055, 0.060),   # 13. 市值 1,637 亿，最常见档占 45%
        "002142.XSHE": ("宁波银行", 0.015, 0.020),   # 14. 市值 1,328 亿，最常见档占 69%
        "600919.XSHG": ("江苏银行", 0.050, 0.055),   # 15. 市值 1,228 亿，最常见档占 36%
        "601169.XSHG": ("北京银行", 0.065, 0.070),   # 16. 市值 958 亿，最常见档占 67%
        "600015.XSHG": ("华夏银行", 0.065, 0.070),   # 17. 市值 894 亿，最常见档占 56%
        "601229.XSHG": ("上海银行", 0.065, 0.070),   # 18. 市值 848 亿，最常见档占 76%
        "601009.XSHG": ("南京银行", 0.065, 0.070),   # 19. 市值 763 亿，最常见档占 32%
        "601916.XSHG": ("浙商银行", 0.040, 0.045),   # 20. 市值 692 亿，最常见档占 43%
    }
    MAX_HOLD = 2            # 最多持有只数
    WEIGHT = 0.5            # 每只占当前总资产比例
    RANK_MODE = "abs"       # 距卖出线距离: "abs"=股息率-卖出线(百分点)；"rel"=股息率/卖出线-1
    SWITCH_MARGIN = 0.005   # 换仓门槛：候选距离需比被换持仓多出 0.5 个百分点才换(0=只要更远就换)
    TRADE_TIME = "14:55"
    DEBUG = True


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025,
                             close_commission=0.00025, close_today_commission=0, min_commission=5), type="stock")
    set_slippage(FixedSlippage(0))
    g.div = None
    run_daily(load_dividends, time="before_open")
    run_daily(trade, time=Cfg.TRADE_TIME)
    log.info(VERSION)


# ---------------- 分红数据与股息率 ----------------
def load_dividends(context):
    """每天盘前读取20只股票的分红送转记录（只保留已除权除息的）。"""
    codes = list(Cfg.BANDS.keys())
    q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.report_date, finance.STK_XR_XD.a_xr_date,
              finance.STK_XR_XD.bonus_ratio_rmb, finance.STK_XR_XD.dividend_ratio,
              finance.STK_XR_XD.transfer_ratio).filter(
        finance.STK_XR_XD.code.in_(codes),
        finance.STK_XR_XD.a_xr_date != None,
        finance.STK_XR_XD.report_date >= "2010-01-01")
    df = finance.run_query(q)
    df = df[pd.to_datetime(df["a_xr_date"]) <= pd.Timestamp(context.current_dt.date())].copy()
    for c in ["bonus_ratio_rmb", "dividend_ratio", "transfer_ratio"]:
        df[c] = df[c].fillna(0).astype(float)
    df["xr"] = pd.to_datetime(df["a_xr_date"])
    df["rd"] = pd.to_datetime(df["report_date"])
    df["fy"] = df["rd"].dt.year
    df["annual"] = df["rd"].dt.month == 12
    df["bonus"] = (df["dividend_ratio"] + df["transfer_ratio"]) / 10.0
    g.div = df


def ttm_dps(code, today):
    """最近 12 个月报告期的每股现金分红（折算为当前股本）。"""
    d = g.div[(g.div.code == code) & (g.div.bonus_ratio_rmb > 0) & (g.div.xr <= today)].sort_values("rd")
    if d.empty:
        return None
    allx = g.div[g.div.code == code]
    end = d.rd.max()
    start = end - pd.DateOffset(months=12)
    # 每笔分红覆盖 (上一个分红报告期, 本报告期]；同一报告期多笔分红共用同一覆盖期
    rds = sorted(pd.Timestamp(x) for x in d.rd.unique())     # 旧版 pandas 的 unique() 返回 numpy datetime64，转成 Timestamp
    prev = {r: (rds[i - 1] if i else r - pd.DateOffset(months=12)) for i, r in enumerate(rds)}
    dps = 0.0
    for _, r in d.iterrows():
        s0 = prev[pd.Timestamp(r.rd)]
        span = (r.rd - s0).days
        part = max(0, (min(r.rd, end) - max(s0, start)).days) / span if span > 0 else 0
        if part <= 0:
            continue
        # 该次及之后的送转会摊薄每股分红
        factor = 1.0
        for b in allx[(allx.xr >= r.xr) & (allx.bonus > 0)].bonus:
            factor *= 1 + b
        dps += r.bonus_ratio_rmb / 10.0 / factor * part
    return dps


def distance(y, low):
    return y - low if Cfg.RANK_MODE == "abs" else y / low - 1


# ---------------- 交易 ----------------
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    info = {}
    today = pd.Timestamp(context.current_dt.date())
    for code, (name, low, high) in Cfg.BANDS.items():
        c = cd[code]
        if c.paused or c.last_price is None or c.last_price <= 0:
            continue
        dps = ttm_dps(code, today)
        if not dps:
            continue
        y = dps / c.last_price
        info[code] = dict(name=name, y=y, low=low, high=high, dist=distance(y, low), price=c.last_price)

    held = [c for c in pos if pos[c].total_amount > 0]

    # 1) 卖出：股息率跌破卖出线
    for code in held:
        x = info.get(code)
        if x and x["y"] < x["low"]:
            _sell(context, code, f"跌破卖出线 股息率{x['y']:.2%}<{x['low']:.1%}")
    held = [c for c in held if c in pos and pos[c].total_amount > 0]

    # 2) 持仓 + 买入候选 一起按距卖出线排序，取前 MAX_HOLD
    cands = [c for c, x in info.items() if c not in held and x["y"] > x["high"] and not cd[c].high_limit == cd[c].last_price]
    pool = [c for c in held if c in info] + cands
    pool.sort(key=lambda c: info[c]["dist"], reverse=True)
    target = pool[:Cfg.MAX_HOLD]

    out = [c for c in held if c in info and c not in target]
    ins = [c for c in target if c not in held]
    # 换仓防抖：被挤出的持仓，只有最弱的新候选也比它远出 SWITCH_MARGIN 才换
    while out and ins and Cfg.SWITCH_MARGIN > 0:
        weakest_in = min(ins, key=lambda c: info[c]["dist"])
        strongest_out = max(out, key=lambda c: info[c]["dist"])
        if info[weakest_in]["dist"] - info[strongest_out]["dist"] >= Cfg.SWITCH_MARGIN:
            break
        ins.remove(weakest_in)
        out.remove(strongest_out)

    for code in out:
        if pos[code].closeable_amount > 0:
            x = info[code]
            _sell(context, code, f"换仓让位 距离{x['dist']:.2%}")
        else:
            # T+1 无法卖出，则少买一只
            ins.sort(key=lambda c: info[c]["dist"])
            if ins:
                ins.pop(0)

    # 3) 买入：每只 = 当前总资产 50%，不超过可用资金
    slots = Cfg.MAX_HOLD - len([c for c in pos if pos[c].total_amount > 0])
    for code in sorted(ins, key=lambda c: info[c]["dist"], reverse=True)[:max(slots, 0)]:
        x = info[code]
        value = min(context.portfolio.total_value * Cfg.WEIGHT, context.portfolio.available_cash)
        if value < x["price"] * 100:
            continue
        o = order_value(code, value)
        if o is not None:
            log.info(f"买入 {x['name']}({code}) 价{x['price']:.2f} 股息率{x['y']:.2%}>{x['high']:.1%} "
                     f"距卖出线{x['dist']:.2%} 金额{value:,.0f}")

    if Cfg.DEBUG:
        hs = [f"{info[c]['name']}{info[c]['y']:.2%}" for c in pos if pos[c].total_amount > 0 and c in info]
        cs = [f"{info[c]['name']}{info[c]['y']:.2%}" for c in cands]
        log.info(f"持仓:{hs or '空'} | 买入候选:{cs or '无'} | 总资产{context.portfolio.total_value:,.0f}")


def _sell(context, code, why):
    name = Cfg.BANDS[code][0]
    o = order_target(code, 0)
    if o is not None:
        log.info(f"卖出 {name}({code}) {why}")
