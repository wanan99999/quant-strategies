# -*- coding: utf-8 -*-
"""
股息率分档建仓策略 V1.0（单只股票）

以该股「最常见股息率区间」[下限 L, 上限 H] 为基准，区间宽度 W = H - L，每 0.2W 一条档位线，共 11 条。
满仓金额 = 当前总资产（每次调仓时按当时总资产计算，赚钱后仓位跟着变大）。
例：区间 2.5%~3.0%（W=0.5%，每档 0.1 个百分点）：

  档位线   2.25% 2.35% 2.45% 2.55% 2.65% 2.75% 2.85% 2.95% 3.05% 3.15% 3.25%
  仓位      0%   10%   20%   30%   40%   50%   60%   70%   80%   90%  100%

规则：股息率「到达」哪条线，就调到那条线对应的仓位（加仓、减仓都一样）。
  - 股息率上升，到达更高的线 -> 加仓到该线仓位。例：从 2.20% 升到 2.35% -> 持有 10%；升到 2.85% -> 60%
  - 股息率下降，回落到更低的线 -> 减仓到该线仓位。例：满仓后从 3.25% 回落到 3.15% -> 减到 90%
  - 在两条线之间来回波动、没碰到新的线 -> 不动。
  - 3.25% 以上一直满仓；2.25% 以下空仓。
  - 回测第一天按股息率所在区间的下沿那条线定仓位（例：2.70% -> 2.65% 线 -> 40%）。

股息率：静态股息率 = 最近一个年度分红已实施的财年每股现金分红合计 ÷ 当前价
        （数据：finance.STK_XR_XD；送转股按除权后股本折算）
成交：分钟回测，每天 14:55 用当前价近似收盘价，计算信号并下单。

【手动修改】只需改 Cfg.STOCK（股票代码）和 Cfg.BAND（最常见区间），其余可不动。
"""
import builtins
from jqdata import *
import pandas as pd
# jqdata 会覆盖内置函数，恢复
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "股息率分档建仓策略 V1.0"


class Cfg:
    # ======== 手动修改区 ========
    STOCK = "600036.XSHG"          # 股票代码：上交所 .XSHG，深交所 .XSHE
    BAND = (0.050, 0.055)          # 最常见区间 (下限, 上限)，例：2.5%~3.0% 写 (0.025, 0.030)
    QUOTA = 1.00                   # 满仓 = 当前总资产 × QUOTA
    # ===========================
    # 档位线：(位置=下限+k×宽度, 仓位)，从低到高
    GRID = [(-0.5, 0.0), (-0.3, 0.1), (-0.1, 0.2), (0.1, 0.3), (0.3, 0.4), (0.5, 0.5),
            (0.7, 0.6), (0.9, 0.7), (1.1, 0.8), (1.3, 0.9), (1.5, 1.0)]
    TRADE_TIME = "14:55"
    DEBUG = True


def initialize(context):
    set_benchmark(Cfg.STOCK)
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025,
                             close_commission=0.00025, close_today_commission=0, min_commission=5), type="stock")
    set_slippage(FixedSlippage(0))
    g.idx = None                   # 当前所在档位线序号(None=尚未初始化)
    g.div = None
    run_daily(load_dividends, time="before_open")
    run_daily(trade, time=Cfg.TRADE_TIME)
    lo, hi = Cfg.BAND
    grid = ", ".join(f"{line(k):.2%}:{f:.0%}" for k, f in Cfg.GRID)
    log.info(f"{VERSION} | {Cfg.STOCK} 区间 {lo:.2%}~{hi:.2%} | 档位 {grid}")


# ---------------- 档位 ----------------
def line(k):
    lo, hi = Cfg.BAND
    return lo + k * (hi - lo)


def floor_idx(y):
    """y 之下(含)最高的档位线序号；低于最低线返回 0（空仓线）。"""
    i = 0
    for j, (k, _) in enumerate(Cfg.GRID):
        if y >= line(k) - 1e-12:
            i = j
    return i


def ceil_idx(y):
    """y 之上(含)最低的档位线序号；高于最高线返回最高线。"""
    for j, (k, _) in enumerate(Cfg.GRID):
        if y <= line(k) + 1e-12:
            return j
    return len(Cfg.GRID) - 1


def next_idx(cur, y):
    """到达更高的线 -> 上移到该线；回落到更低的线 -> 下移到该线；否则不变。"""
    if cur is None:
        return floor_idx(y)
    f, c = floor_idx(y), ceil_idx(y)
    if f > cur:
        return f
    if c < cur:
        return c
    return cur


# ---------------- 分红数据与股息率 ----------------
def load_dividends(context):
    """每天盘前读取分红送转记录（只保留已除权除息的）。"""
    q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.report_date, finance.STK_XR_XD.a_xr_date,
              finance.STK_XR_XD.bonus_ratio_rmb, finance.STK_XR_XD.dividend_ratio,
              finance.STK_XR_XD.transfer_ratio).filter(
        finance.STK_XR_XD.code == Cfg.STOCK,
        finance.STK_XR_XD.a_xr_date != None,
        finance.STK_XR_XD.report_date >= "2010-01-01")
    df = finance.run_query(q)
    df = df[pd.to_datetime(df["a_xr_date"]) <= pd.Timestamp(context.current_dt.date())].copy()
    for c in ["bonus_ratio_rmb", "dividend_ratio", "transfer_ratio"]:
        df[c] = df[c].fillna(0).astype(float)
    df["xr"] = pd.to_datetime(df["a_xr_date"])
    df["fy"] = pd.to_datetime(df["report_date"]).dt.year
    df["annual"] = pd.to_datetime(df["report_date"]).dt.month == 12
    df["bonus"] = (df["dividend_ratio"] + df["transfer_ratio"]) / 10.0
    g.div = df


def static_dps():
    """最近一个年度分红已实施的财年每股现金分红合计（折算为当前股本）。"""
    d = g.div
    ann = d[d.annual & (d.bonus_ratio_rmb > 0)]
    if ann.empty:
        return None
    rows = d[(d.fy == ann.fy.max()) & (d.bonus_ratio_rmb > 0)]
    dps = 0.0
    for _, r in rows.iterrows():
        factor = 1.0
        for b in d[(d.xr >= r.xr) & (d.bonus > 0)].bonus:   # 该次及之后的送转摊薄每股分红
            factor *= 1 + b
        dps += r.bonus_ratio_rmb / 10.0 / factor
    return dps


# ---------------- 交易 ----------------
def trade(context):
    code = Cfg.STOCK
    c = get_current_data()[code]
    if c.paused or not c.last_price:
        return
    dps = static_dps()
    if not dps:
        return
    price = c.last_price
    y = dps / price
    pos = context.portfolio.positions
    cur = pos[code].value if code in pos else 0.0

    new = next_idx(g.idx, y)
    if new != g.idx:
        f = Cfg.GRID[new][1]
        target = context.portfolio.total_value * Cfg.QUOTA * f
        diff = target - cur
        if diff >= price * 100:                                  # 加仓
            value = min(diff, context.portfolio.available_cash)
            if value >= price * 100 and order_value(code, value) is not None:
                log.info(f"{'建仓' if cur <= 0 else '加仓'}至{f:.0%} 价{price:.2f} 股息率{y:.2%} 目标{target:,.0f} 买入{value:,.0f}")
            g.idx = new
        elif -diff >= price * 100 or (f == 0 and cur > 0):        # 减仓/清仓
            if pos[code].closeable_amount > 0:                   # T+1：当天买入的不能卖，次日再处理
                if f == 0:
                    order_target(code, 0)
                    log.info(f"清仓 价{price:.2f} 股息率{y:.2%}")
                else:
                    order_target_value(code, target)
                    log.info(f"减仓至{f:.0%} 价{price:.2f} 股息率{y:.2%} 目标{target:,.0f} 卖出{-diff:,.0f}")
                g.idx = new
        else:
            g.idx = new

    pv = pos[code].value if code in pos else 0.0
    record(yield_pct=y * 100, position_pct=pv / context.portfolio.total_value * 100)
    if Cfg.DEBUG:
        log.info(f"股息率{y:.2%} 档位{Cfg.GRID[g.idx][1]:.0%} 持仓市值{pv:,.0f} 总资产{context.portfolio.total_value:,.0f}")
