# -*- coding: utf-8 -*-
"""
股息率分档建仓策略 V1.1（5 只股票组合）

从 20 家银行股票池中选 5 只，每只独立按股息率分档。
每只满仓额度 = 当前总资产 × 20%（每次调仓时按当时总资产计算，赚钱后额度跟着变大），
区间中点仓位 = 额度的 50% = 总资产的 10%。
以该股「最常见股息率区间」[下限 L, 上限 H] 为基准，W = H - L，每 0.2W 一条档位线，共 11 条。
例：区间 2.5%~3.0%（每档 0.1 个百分点），总资产 1000 万（该股额度 200 万）：

  档位线   2.25% 2.35% 2.45% 2.55% 2.65% 2.75% 2.85% 2.95% 3.05% 3.15% 3.25%
  占额度    0%   10%   20%   30%   40%   50%   60%   70%   80%   90%  100%
  金额      0    20万  40万  60万  80万  100万 120万 140万 160万 180万 200万

规则：股息率「到达」哪条线，就调到那条线对应的仓位（加仓、减仓都一样）。
  - 股息率上升，到达更高的线 -> 加仓到该线仓位。例：从 2.20% 升到 2.35% -> 持有 10%（20 万）
  - 股息率下降，回落到更低的线 -> 减仓到该线仓位。例：满仓后从 3.25% 回落到 3.15% -> 减到 90%
  - 在两条线之间来回波动、没碰到新的线 -> 不动。
  - 3.25% 以上一直满仓；2.25% 以下空仓。
  - 回测第一天按股息率所在区间的下沿那条线定仓位（例：2.70% -> 2.65% 线 -> 40%）。

股息率：静态股息率 = 最近一个年度分红已实施的财年每股现金分红合计 ÷ 当前价
        （数据：finance.STK_XR_XD；送转股按除权后股本折算）
成交：分钟回测，每天 14:55 用当前价近似收盘价，计算信号并下单；先减仓再加仓。

【手动修改】改 Cfg.SELECTED（5 只股票代码）和 Cfg.POOL 里对应股票的区间即可。
"""
import builtins
from jqdata import *
import pandas as pd
# jqdata 会覆盖内置函数，恢复
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "股息率分档建仓策略 V1.1"


class Cfg:
    # ======== 手动修改区 ========
    # 选中的股票（最多 5 只），代码需在下面 POOL 里
    SELECTED = ["601998.XSHG", "600015.XSHG", "600016.XSHG", "000001.XSHE", "600036.XSHG"]
    # 股票池：代码: (名称, 最常见区间下限, 上限)  —— 默认值为 2023-01-01~2026-10-28 静态股息率按 0.5% 分档的最常见区间
    POOL = {
        "601398.XSHG": ("工商银行", 0.040, 0.045),
        "601288.XSHG": ("农业银行", 0.035, 0.040),
        "601988.XSHG": ("中国银行", 0.040, 0.045),
        "601939.XSHG": ("建设银行", 0.040, 0.045),
        "601658.XSHG": ("邮储银行", 0.050, 0.055),
        "601328.XSHG": ("交通银行", 0.050, 0.055),
        "601818.XSHG": ("光大银行", 0.055, 0.060),
        "601998.XSHG": ("中信银行", 0.045, 0.050),
        "600016.XSHG": ("民生银行", 0.050, 0.055),
        "600000.XSHG": ("浦发银行", 0.045, 0.050),
        "601916.XSHG": ("浙商银行", 0.050, 0.055),
        "600036.XSHG": ("招商银行", 0.050, 0.055),
        "601166.XSHG": ("兴业银行", 0.055, 0.060),
        "601169.XSHG": ("北京银行", 0.055, 0.060),
        "000001.XSHE": ("平安银行", 0.050, 0.055),
        "600919.XSHG": ("江苏银行", 0.045, 0.050),
        "600015.XSHG": ("华夏银行", 0.055, 0.060),
        "601229.XSHG": ("上海银行", 0.050, 0.055),
        "601009.XSHG": ("南京银行", 0.050, 0.055),
        "601077.XSHG": ("渝农商行", 0.045, 0.050),
    }
    QUOTA = 0.20                   # 每只满仓额度 = 当前总资产 × QUOTA
    # ===========================
    MAX_STOCKS = 5
    # 档位线：(位置=下限+k×宽度, 仓位占额度)，从低到高
    GRID = [(-0.5, 0.0), (-0.3, 0.1), (-0.1, 0.2), (0.1, 0.3), (0.3, 0.4), (0.5, 0.5),
            (0.7, 0.6), (0.9, 0.7), (1.1, 0.8), (1.3, 0.9), (1.5, 1.0)]
    TRADE_TIME = "14:55"
    DEBUG = True


def initialize(context):
    assert len(Cfg.SELECTED) <= Cfg.MAX_STOCKS, f"最多 {Cfg.MAX_STOCKS} 只股票"
    for code in Cfg.SELECTED:
        assert code in Cfg.POOL, f"{code} 不在 POOL 中，请先在 POOL 里加上它的区间"
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    log.set_level("order", "error")
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025,
                             close_commission=0.00025, close_today_commission=0, min_commission=5), type="stock")
    set_slippage(FixedSlippage(0))
    g.idx = {code: None for code in Cfg.SELECTED}    # 每只当前所在档位线序号(None=尚未初始化)
    g.div = None
    run_daily(load_dividends, time="before_open")
    run_daily(trade, time=Cfg.TRADE_TIME)
    log.info(f"{VERSION} | 每只额度 = 总资产 × {Cfg.QUOTA:.0%}")
    for code in Cfg.SELECTED:
        name, lo, hi = Cfg.POOL[code]
        grid = ", ".join(f"{line(code, k):.2%}:{f:.0%}" for k, f in Cfg.GRID)
        log.info(f"  {name}({code}) 区间 {lo:.2%}~{hi:.2%} | 档位 {grid}")


# ---------------- 分档 ----------------
def line(code, k):
    _, lo, hi = Cfg.POOL[code]
    return lo + k * (hi - lo)


def floor_idx(code, y):
    """y 之下(含)最高的档位线序号；低于最低线返回 0（空仓线）。"""
    i = 0
    for j, (k, _) in enumerate(Cfg.GRID):
        if y >= line(code, k) - 1e-12:
            i = j
    return i


def ceil_idx(code, y):
    """y 之上(含)最低的档位线序号；高于最高线返回最高线。"""
    for j, (k, _) in enumerate(Cfg.GRID):
        if y <= line(code, k) + 1e-12:
            return j
    return len(Cfg.GRID) - 1


def next_idx(code, cur, y):
    """到达更高的线 -> 上移到该线；回落到更低的线 -> 下移到该线；否则不变。"""
    if cur is None:
        return floor_idx(code, y)
    f, c = floor_idx(code, y), ceil_idx(code, y)
    if f > cur:
        return f
    if c < cur:
        return c
    return cur


# ---------------- 分红数据与股息率 ----------------
def load_dividends(context):
    """每天盘前读取选中股票的分红送转记录（只保留已除权除息的）。"""
    q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.report_date, finance.STK_XR_XD.a_xr_date,
              finance.STK_XR_XD.bonus_ratio_rmb, finance.STK_XR_XD.dividend_ratio,
              finance.STK_XR_XD.transfer_ratio).filter(
        finance.STK_XR_XD.code.in_(Cfg.SELECTED),
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


def static_dps(code):
    """最近一个年度分红已实施的财年每股现金分红合计（折算为当前股本）。"""
    d = g.div[g.div.code == code]
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


def _yield(cd, code):
    c = cd[code]
    if c.paused or not c.last_price:
        return None
    dps = static_dps(code)
    return dps / c.last_price if dps else None


# ---------------- 交易 ----------------
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    quota = context.portfolio.total_value * Cfg.QUOTA     # 本次调仓按当前总资产计算额度
    ys = {code: _yield(cd, code) for code in Cfg.SELECTED}
    plan = {}
    for code in Cfg.SELECTED:
        y = ys[code]
        if y is None:
            continue
        new = next_idx(code, g.idx[code], y)
        if new != g.idx[code]:
            plan[code] = new

    # 1) 先减仓/清仓，释放资金
    for code, new in plan.items():
        f = Cfg.GRID[new][1]
        name, price, y = Cfg.POOL[code][0], cd[code].last_price, ys[code]
        cur = pos[code].value if code in pos else 0.0
        target = quota * f
        if not (cur - target >= price * 100 or (f == 0 and cur > 0)):
            continue
        if pos[code].closeable_amount <= 0:                       # T+1：当天买入的不能卖，次日再处理
            plan[code] = None
            continue
        if f == 0:
            order_target(code, 0)
            log.info(f"清仓 {name}({code}) 价{price:.2f} 股息率{y:.2%}")
        else:
            order_target_value(code, target)
            log.info(f"减仓 {name}({code}) 至{f:.0%} 价{price:.2f} 股息率{y:.2%} 目标{target:,.0f} 卖出{cur - target:,.0f}")
        g.idx[code] = new
        plan[code] = None

    # 2) 再建仓/加仓
    for code, new in plan.items():
        if new is None:
            continue
        f = Cfg.GRID[new][1]
        name, price, y = Cfg.POOL[code][0], cd[code].last_price, ys[code]
        cur = pos[code].value if code in pos else 0.0
        target = quota * f
        value = min(target - cur, context.portfolio.available_cash)
        if value >= price * 100 and order_value(code, value) is not None:
            log.info(f"{'建仓' if cur <= 0 else '加仓'} {name}({code}) 至{f:.0%} 价{price:.2f} 股息率{y:.2%} "
                     f"目标{target:,.0f} 买入{value:,.0f}")
        g.idx[code] = new

    if Cfg.DEBUG:
        st = []
        for code in Cfg.SELECTED:
            pv = pos[code].value if code in pos else 0.0
            y, i = ys[code], g.idx[code]
            ytxt = "-" if y is None else f"{y:.2%}"
            itxt = "-" if i is None else f"{Cfg.GRID[i][1]:.0%}"
            st.append(f"{Cfg.POOL[code][0]}{ytxt}/档{itxt}/{pv / 1e4:,.0f}万")
        log.info(" | ".join(st) + f" | 现金{context.portfolio.available_cash / 1e4:,.0f}万 总资产{context.portfolio.total_value / 1e4:,.0f}万")
