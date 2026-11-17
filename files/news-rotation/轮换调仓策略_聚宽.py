# -*- coding: utf-8 -*-
"""
轮换调仓策略  (聚宽 JoinQuant 平台)
========================================================
根据《短线交易规则手册 · 买卖点交易计划》实现。

核心执行顺序（每日 14:50 尾盘统一评估）：
    STEP 1  换手率过滤 : 当日换手率 < 3% 的标的不参与（不买、不作为轮动目标）
    STEP 2  5日均线防守: 14:50 最新价仍低于当日动态 MA5  → 卖出该持仓
    STEP 3  RSI 风控   : RSI(14) > 80 且当日涨幅 < 10% → 直接清仓
                         （涨幅 ≥ 10% 接近涨停，视为强势，保留不清）
    STEP 4  轮动调仓   : 触发(持仓涨幅>5% 或 池内分化>5%)时做高低切换：
                         → 有可用现金：不切换，直接用现金买入合格标的
                         → 已满仓无现金：强势持仓 100% 切换到"合格、当日涨幅较小"的标的
                         未触发轮换时：只要有空仓位就买入合格标的（而非保持空仓），
                                       仅当无任何标的合格时才空仓。
                            合格标准：换手率≥3%、站上实时MA5、实时RSI<80、涨幅<10%（均取14:50实时数据）

说明：
  · 本策略是"股票池内部轮动"，请在 g.stock_pool 中填入你的股票池。
  · 所有价格取 14:50 实时最新价；MA5 为"动态MA5"(前4日收盘 + 当前价)/5。
  · 每笔买入固定使用总资金的 33%，最多同时持有 3 只（≈满仓）。
  · order_target_value 精确控制每只标的的持仓市值。
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
    g.max_positions  = 3        # 最多同时持有几只
    g.position_pct   = 0.33     # 每笔买入使用总资金的比例 (33%)，3笔≈满仓
    g.turnover_min   = 3.0      # 换手率门槛(%)      —— STEP1
    g.ma_period      = 5        # 均线周期           —— STEP2
    g.rsi_period     = 14       # RSI 周期           —— STEP3
    g.rsi_reduce     = 80       # RSI 超买阈值（>80 触发清仓判断）
    g.rsi_gain_cap   = 0.10     # RSI 清仓的涨幅上限：当日涨幅 < 10% 才清仓
    g.rotate_gain    = 0.05     # 持仓涨幅触发轮动阈值 (5%)  —— STEP4
    g.divergence     = 0.05     # 池内分化触发轮动阈值 (5%)
    g.switch_ratio   = 1.00     # 每次切换转移的仓位比例 (100%)
    g.cash_threshold = 5000     # 可用现金 ≥ 此值时优先用现金买入、不做切换
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

    # -------- STEP 4：轮动调仓 --------
    rotation_adjust(context, metrics)

    # -------- 补仓：有空余仓位时买入合格候选 --------
    fill_positions(context, metrics)


def is_eligible_target(m):
    """买入/轮动候选合格：换手率≥3%、站上实时MA5、实时RSI<80、涨幅<10%（全部取14:50实时数据）。"""
    if m is None or m['rsi'] is None:
        return False
    return (m['turnover'] >= g.turnover_min and          # 14:50 换手率 ≥ 3%
            m['price'] >= m['ma5'] and                    # 实时价 ≥ 实时MA5
            m['rsi'] < g.rsi_overbought and               # 实时RSI < 80
            m['gain'] < g.rsi_gain_cap)                   # 涨幅 < 10%(未涨停)


def rotation_adjust(context, metrics):
    positions = context.portfolio.positions
    held = [s for s in positions.keys() if s in metrics]
    if not held:
        return

    # 触发条件A：任一持仓当日涨幅 > 5%
    strong = [s for s in held if metrics[s]['gain'] > g.rotate_gain]

    # 触发条件B：池内涨跌幅分化 > 5%
    pool_gains = [m['gain'] for m in metrics.values()]
    divergence = (max(pool_gains) - min(pool_gains)) if pool_gains else 0.0
    triggered = bool(strong) or divergence > g.divergence
    if not triggered:
        return
    log.info('[STEP4 触发轮动] 强势持仓=%s 池内分化=%.2f%%' %
             (strong, divergence * 100))

    # 【现金优先】账户有可用现金 → 不做切换，交由 fill_positions() 用现金买入合格标的
    if context.portfolio.available_cash >= g.cash_threshold:
        log.info('[STEP4] 账户有现金%.0f元(≥%.0f) → 不切换，改用现金买入' %
                 (context.portfolio.available_cash, g.cash_threshold))
        return

    # 找"当日涨幅较小、且符合条件"的候选目标（不在持仓中，涨幅升序）
    candidates = [s for s in metrics.keys()
                  if s not in positions and is_eligible_target(metrics[s])]
    candidates.sort(key=lambda x: metrics[x]['gain'])
    if not candidates:
        # 池内无未涨/合格标的 → 按决策树：全部上涨则不动(RSI已在STEP3处理减仓)
        log.info('[STEP4] 无符合条件的落后标的，维持现状')
        return

    # 强势持仓优先，否则用池内最强的持仓作为切出对象
    switch_from = strong if strong else [max(held, key=lambda x: metrics[x]['gain'])]

    for src in switch_from:
        if src not in context.portfolio.positions:
            continue
        if not candidates:
            break
        tgt = candidates.pop(0)                       # 涨幅最小的合格标的
        src_value = context.portfolio.positions[src].value
        move_value = src_value * g.switch_ratio       # 转移 100%

        # 先清空强势持仓（100%）
        order_target_value(src, src_value - move_value)
        # 再把释放的仓位买入目标标的
        tgt_cur = context.portfolio.positions[tgt].value if tgt in context.portfolio.positions else 0.0
        order_target_value(tgt, tgt_cur + move_value)
        log.info('[STEP4 切换] %s(涨%.2f%%) -%.0f%% → %s(涨%.2f%%) 金额%.0f' %
                 (src, metrics[src]['gain'] * 100, g.switch_ratio * 100,
                  tgt, metrics[tgt]['gain'] * 100, move_value))


def fill_positions(context, metrics):
    """空余仓位时，买入合格候选，每笔用总资金33%（对应'再次买入条件'）。"""
    positions = context.portfolio.positions
    slots = g.max_positions - len(positions)
    if slots <= 0:
        return
    candidates = [s for s in metrics.keys()
                  if s not in positions and is_eligible_target(metrics[s])]
    candidates.sort(key=lambda x: metrics[x]['gain'])   # 优先买涨幅较小的
    if not candidates:
        return
    # 每笔买入固定使用【总资金的 33%】(3 笔≈满仓，故最多同时持有 3 只)
    target_value = context.portfolio.total_value * g.position_pct
    for s in candidates[:slots]:
        if context.portfolio.available_cash < 1000:
            break
        order_target_value(s, target_value)
        log.info('[买入] %s 涨%.2f%% 换手%.2f%% 目标市值%.0f(总资金%.0f%%)' %
                 (s, metrics[s]['gain'] * 100, metrics[s]['turnover'],
                  target_value, g.position_pct * 100))
