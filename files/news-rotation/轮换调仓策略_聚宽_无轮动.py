# -*- coding: utf-8 -*-
"""
调仓策略 · 保留换手率 / 去掉轮动版  (聚宽 JoinQuant 平台)
========================================================
根据《短线交易规则手册 · 买卖点交易计划》实现。
【保留换手率过滤】【已去掉轮动调仓】【最多同时持有 3 只】

核心执行顺序（每日 14:50 尾盘统一评估）：
    STEP 1  换手率过滤 : 当日换手率 < 3% 的标的不参与（不买）
    STEP 2  5日均线防守: 14:50 最新价仍低于当日动态 MA5  → 卖出该持仓
    STEP 3  RSI 风控   : RSI(14) > 80 且当日涨幅 < 10% → 直接清仓
                         （涨幅 ≥ 10% 接近涨停，视为强势，保留不清）
    STEP 4  建仓/补仓  : 持仓不足 3 只时，买入符合条件的候选
                         （换手率≥3%、站上实时MA5、实时RSI<80、涨幅<10%，14:50实时数据）

说明：
  · 无轮动切换：不再因涨幅>5%或池内分化>5%在标的间倒仓。
  · 持仓上限固定为 g.max_positions = 3，绝不超过。
  · 每笔买入固定使用总资金的 33%（3 笔≈满仓）。
  · 所有价格取 14:50 实时最新价；MA5 为"动态MA5"(前4日收盘 + 当前价)/5。
"""

import numpy as np
from jqdata import *


# ============================ 初始化 ============================
def initialize(context):
    # ---- 交易环境 ----
    set_option('use_real_price', True)              # 使用真实价格
    set_option('avoid_future_data', True)           # 避免未来数据
    set_benchmark('000300.XSHG')                    # 基准：沪深300
    set_slippage(FixedSlippage(0.002))              # 滑点
    set_order_cost(OrderCost(
        open_commission=0.0003, close_commission=0.0003,
        min_commission=5, close_today_commission=0), type='stock')
    log.set_level('order', 'error')

    # ---- 股票池（请替换为你自己的池子，例如行业ETF成分股 / 主题概念股）----
    g.stock_pool = [
        '600519.XSHG',   # 贵州茅台  （示例，请替换）
        '000858.XSHE',   # 五粮液
        '600036.XSHG',   # 招商银行
        '000333.XSHE',   # 美的集团
        '600276.XSHG',   # 恒瑞医药
        '002594.XSHE',   # 比亚迪
    ]

    # ---- 策略参数 ----
    g.max_positions  = 3        # 【硬性上限】最多同时持有 3 只
    g.position_pct   = 0.33     # 每笔买入使用总资金的比例 (33%)，3笔≈满仓
    g.turnover_min   = 3.0      # 换手率门槛(%)      —— STEP1
    g.ma_period      = 5        # 均线周期           —— STEP2
    g.rsi_period     = 14       # RSI 周期           —— STEP3
    g.rsi_reduce     = 80       # RSI 超买阈值（>80 触发清仓判断）
    g.rsi_gain_cap   = 0.10     # RSI 清仓的涨幅上限：当日涨幅 < 10% 才清仓
    g.rsi_overbought = 80       # 买入 RSI 上限：实时RSI < 80 才买(14:50数据)

    # ---- 定时任务：每日 14:50 用当日实时数据评估、当日成交（不滞后）----
    #  ★ 重要：回测须设为【分钟】频率，14:50 才会准点触发；
    #     若用日线频率回测，操作会顺延到收盘统一撮合，无法精确到 14:50。
    run_daily(daily_trade, time='14:50', reference_security='000300.XSHG')


# ============================ 指标计算 ============================
def compute_metrics(context, security):
    """采集单只标的当前的量价指标；数据不足或停牌返回 None。"""
    cd = get_current_data()[security]
    if cd.paused:                       # 停牌不参与
        return None

    price = cd.last_price               # 14:50 实时最新价
    if not price or np.isnan(price):
        return None

    # 取足够长的日线收盘（含当日在成形的动态K线）用于 MA / RSI
    n = max(g.ma_period, g.rsi_period + 1) + 25
    bars = get_bars(security, count=n, unit='1d', fields=['close', 'volume'],
                    include_now=True, end_dt=context.current_dt, df=False)
    closes = np.array(bars['close'], dtype=float)
    if len(closes) < g.rsi_period + 1:
        return None
    # 用实时最新价替换当日成形K线的收盘，得到真正的"动态"数值
    closes[-1] = price

    # --- 动态 MA5 ---
    ma = closes[-g.ma_period:].mean()

    # --- RSI(14)  (Wilder 平滑) ---
    rsi = calc_rsi(closes, g.rsi_period)

    # --- 当日涨幅 ---
    prev_close = closes[-2]
    gain = (price / prev_close - 1.0) if prev_close > 0 else 0.0

    # --- 涨停判断 ---
    limit_up = cd.high_limit and price >= cd.high_limit - 0.001

    # --- 当日换手率(%) = 当日累计成交量 / 流通股本 * 100 ---
    turnover = get_turnover(context, security, bars['volume'][-1])

    return {
        'price': price, 'ma5': ma, 'rsi': rsi, 'gain': gain,
        'limit_up': bool(limit_up), 'turnover': turnover,
    }


def calc_rsi(closes, period):
    """Wilder RSI。"""
    diff = np.diff(closes)
    if len(diff) < period:
        return None
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    for i in range(period, len(diff)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def get_turnover(context, security, today_volume):
    """当日换手率(%)：当日累计成交量(股) / 流通股本(股) * 100。"""
    try:
        val = get_valuation(security, end_date=context.previous_date,
                            count=1, fields=['circulating_cap'])
        if val is None or len(val) == 0:
            return 0.0
        circ_shares = float(val['circulating_cap'].iloc[0]) * 10000.0  # 万股 -> 股
        if circ_shares <= 0:
            return 0.0
        return float(today_volume) / circ_shares * 100.0
    except Exception as e:
        log.warn('换手率计算失败 %s: %s' % (security, e))
        return 0.0


# ============================ 每日交易主流程 ============================
def daily_trade(context):
    # 一次性采集全池指标
    metrics = {}
    for s in g.stock_pool:
        m = compute_metrics(context, s)
        if m is not None:
            metrics[s] = m

    positions = context.portfolio.positions

    # -------- STEP 2：5日均线防守（对持仓）--------
    for s in list(positions.keys()):
        m = metrics.get(s)
        if m is None:
            continue
        if m['price'] < m['ma5']:
            order_target_value(s, 0)
            log.info('[STEP2 跌破MA5清仓] %s 价%.2f < MA5 %.2f' % (s, m['price'], m['ma5']))

    # -------- STEP 3：RSI 风控（对剩余持仓）--------
    # 规则：RSI(14) > 80 且当日涨幅 < 10% → 直接清仓
    #      （涨幅 ≥ 10% 接近涨停，视为强势，保留不清）
    for s in list(context.portfolio.positions.keys()):
        m = metrics.get(s)
        if m is None or m['rsi'] is None:
            continue
        if m['rsi'] > g.rsi_reduce and m['gain'] < g.rsi_gain_cap:
            order_target_value(s, 0)
            log.info('[STEP3 RSI>%d且涨幅<%.0f%%清仓] %s RSI=%.1f 涨幅%.2f%%' %
                     (g.rsi_reduce, g.rsi_gain_cap * 100, s, m['rsi'], m['gain'] * 100))

    # -------- STEP 4：建仓/补仓（不足 3 只时买入合格候选）--------
    fill_positions(context, metrics)


def is_eligible_target(m):
    """买入候选合格：换手率≥3%、站上实时MA5、实时RSI<80、涨幅<10%（全部取14:50实时数据）。"""
    if m is None or m['rsi'] is None:
        return False
    return (m['turnover'] >= g.turnover_min and          # 14:50 换手率 ≥ 3%
            m['price'] >= m['ma5'] and                    # 实时价 ≥ 实时MA5
            m['rsi'] < g.rsi_overbought and               # 实时RSI < 80
            m['gain'] < g.rsi_gain_cap)                   # 涨幅 < 10%(未涨停)


def fill_positions(context, metrics):
    """持仓不足 3 只时，买入合格候选，每笔用总资金33%（对应'再次买入条件'）。"""
    positions = context.portfolio.positions
    slots = g.max_positions - len(positions)     # 剩余可用仓位数
    if slots <= 0:                                # 已满 3 只，不再买入
        return
    candidates = [s for s in metrics.keys()
                  if s not in positions and is_eligible_target(metrics[s])]
    candidates.sort(key=lambda x: metrics[x]['gain'])   # 优先买涨幅较小的
    if not candidates:
        return
    # 每笔买入固定使用【总资金的 33%】(3 笔≈满仓，保证总数≤3)
    target_value = context.portfolio.total_value * g.position_pct
    for s in candidates[:slots]:                 # 最多只买满剩余仓位，保证总数≤3
        if context.portfolio.available_cash < 1000:
            break
        order_target_value(s, target_value)
        log.info('[买入] %s 涨%.2f%% 换手%.2f%% 目标市值%.0f(总资金%.0f%%)' %
                 (s, metrics[s]['gain'] * 100, metrics[s]['turnover'],
                  target_value, g.position_pct * 100))
