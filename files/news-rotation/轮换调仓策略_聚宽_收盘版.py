# -*- coding: utf-8 -*-
"""
轮换调仓策略 (聚宽 JoinQuant 平台) —— 收盘数据版
根据《短线交易规则手册 · 买卖点交易计划》实现。

【本版相对"14:50 实时版"的改动】
  1. 所有原来取"14:50 实时数据"的地方，一律改用【该交易日收盘数据】：
     收盘价、真实 MA5（近5日收盘均值）、RSI(14)、当日涨幅、全日换手率。
  2. 所有盘中操作按"当日收盘评估、分两个时点成交"执行：
       · 卖出（STEP2 破MA5 / STEP3 RSI风控 / STEP4 切出强势持仓）→ 当日【收盘】成交。
       · 买入（STEP4 换入落后标的 / 补仓）→ 记入计划，于【次日开盘】成交。

核心执行顺序（每日收盘统一评估）：
  STEP 1 换手率过滤 : 当日换手率 < 3% 的标的不参与（不买、不作为轮动目标）
  STEP 2 5日均线防守: 收盘价仍低于当日 MA5 → 当日收盘卖出该持仓
  STEP 3 RSI 风控   : RSI(14) > 80 且当日涨幅 < 10% → 当日收盘清仓
                      （涨幅 ≥ 10% 接近涨停，视为强势，保留不清）
  STEP 4 轮动调仓   : 触发(持仓涨幅>5% 或 池内分化>5%)时做高低切换：
      → 有可用现金：不切换，直接次日开盘用现金买入合格标的
      → 已满仓无现金：强势持仓当日收盘卖出，其仓位次日开盘换入"合格、涨幅较小"标的
    未触发轮换时：只要有空仓位就（计划）买入合格标的，仅当无任何标的合格时才空仓。
  合格标准：换手率≥3%、站上MA5、RSI<80、涨幅<10%（均取当日收盘数据）

说明：
  · 本策略是"股票池内部轮动"，请在 g.stock_pool 中填入你的股票池。
  · 所有价格与指标取【当日收盘】数值；MA5 为近5日真实收盘均值。
  · 每笔买入固定使用总资金的 33%，最多同时持有 3 只（≈满仓）。
  · order_target_value 精确控制每只标的的持仓市值。
"""
import numpy as np
from jqdata import *


# ============================ 初始化 ============================
def initialize(context):
    # ---- 交易环境 ----
    set_option('use_real_price', True)      # 使用真实价格
    set_option('avoid_future_data', True)   # 避免未来数据
    set_benchmark('000300.XSHG')            # 基准：沪深300
    set_slippage(FixedSlippage(0.002))      # 滑点
    set_order_cost(OrderCost(
        open_commission=0.0003,
        close_commission=0.0003,
        min_commission=5,
        close_today_commission=0), type='stock')
    log.set_level('order', 'error')

    # ---- 股票池（请替换为你自己的池子，例如行业ETF成分股 / 主题概念股）----
    g.stock_pool = [
        '301387.XSHE',  # 光大同创 （示例，请替换）
        '002436.XSHE',  # 兴森科技
        '300120.XSHE',  # 经纬辉开
        '000601.XSHE',  # 韶能股份
        '003018.XSHE',  # 金富科技
        '000510.XSHE',  # 新金路
        '300128.XSHE',  # 锦富技术
        '301560.XSHE',  # 众捷汽车
        '688226.XSHG',  # 威腾电气
        '603291.XSHG',  # 联合水务
        '600418.XSHG',  # 江淮汽车
        '688011.XSHG',  # 新光光电
        '300022.XSHE',  # 吉峰科技
    ]

    # ---- 策略参数 ----
    g.max_positions = 3       # 最多同时持有几只
    g.position_pct = 0.33     # 每笔买入使用总资金的比例 (33%)，3笔≈满仓
    g.turnover_min = 3.0      # 换手率门槛(%) —— STEP1
    g.ma_period = 5           # 均线周期 —— STEP2
    g.rsi_period = 14         # RSI 周期 —— STEP3
    g.rsi_reduce = 80         # RSI 超买阈值（>80 触发清仓判断）
    g.rsi_gain_cap = 0.10     # RSI 清仓的涨幅上限：当日涨幅 < 10% 才清仓
    g.rotate_gain = 0.05      # 持仓涨幅触发轮动阈值 (5%) —— STEP4
    g.divergence = 0.05       # 池内分化触发轮动阈值 (5%)
    g.switch_ratio = 1.00     # 每次切换转移的仓位比例 (100%)
    g.cash_threshold = 5000   # 可用现金 ≥ 此值时优先用现金买入、不做切换
    g.rsi_overbought = 80     # 买入 RSI 上限：收盘RSI < 80 才买

    # ---- 次日开盘待买清单：[(security, target_value), ...] ----
    g.to_buy = []

    # ---- 定时任务 ----
    # 次日开盘：执行昨日收盘定下的买入计划（买入 → 开盘成交）
    run_daily(morning_buy, time='open', reference_security='000300.XSHG')
    # 当日收盘：用当日收盘数据评估，卖出当日收盘成交，并生成次日开盘买入计划
    # ★ 使用 'open' / 'close' 定时点，日线与分钟频率回测均可正确撮合：
    #   卖单在当日收盘价成交，买单于次日开盘价成交。
    run_daily(evening_eval, time='close', reference_security='000300.XSHG')


# ============================ 指标计算 ============================
def compute_metrics(context, security):
    """采集单只标的【当日收盘】的量价指标；数据不足或停牌返回 None。"""
    cd = get_current_data()[security]
    if cd.paused:  # 停牌不参与
        return None

    # 取足够长的日线收盘（含当日已收盘的完整K线）用于 MA / RSI
    n = max(g.ma_period, g.rsi_period + 1) + 25
    bars = get_bars(security, count=n, unit='1d', fields=['close'],
                    include_now=True, end_dt=context.current_dt, df=False)
    closes = np.array(bars['close'], dtype=float)
    if len(closes) < g.rsi_period + 1:
        return None

    price = closes[-1]  # 当日收盘价
    if not price or np.isnan(price):
        return None

    # --- 真实 MA5（近5日收盘均值）---
    ma = closes[-g.ma_period:].mean()

    # --- RSI(14) (Wilder 平滑，基于收盘) ---
    rsi = calc_rsi(closes, g.rsi_period)

    # --- 当日涨幅（以收盘价计）---
    prev_close = closes[-2]
    gain = (price / prev_close - 1.0) if prev_close > 0 else 0.0

    # --- 涨停判断 ---
    limit_up = cd.high_limit and price >= cd.high_limit - 0.001

    # --- 当日换手率(%)：直接取当日收盘的换手率字段 ---
    turnover = get_turnover(context, security)

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


def get_turnover(context, security):
    """当日换手率(%)：直接取当日收盘的 turnover_ratio 字段，不再自行计算。"""
    try:
        val = get_valuation(security, end_date=context.current_dt, count=1,
                            fields=['turnover_ratio'])
        if val is None or len(val) == 0:
            return 0.0
        tr = val['turnover_ratio'].iloc[0]
        if tr is None or np.isnan(tr):
            return 0.0
        return float(tr)  # turnover_ratio 已是百分比(%)
    except Exception as e:
        log.warn('换手率获取失败 %s: %s' % (security, e))
        return 0.0


# ==================== 每日收盘评估主流程（卖出=当日收盘） ====================
def evening_eval(context):
    # 一次性采集全池指标（当日收盘数据）
    metrics = {}
    for s in g.stock_pool:
        m = compute_metrics(context, s)
        if m is not None:
            metrics[s] = m

    # 本轮的次日开盘买入计划从空开始重建
    g.to_buy = []

    positions = context.portfolio.positions

    # -------- STEP 2：5日均线防守（对持仓）→ 当日收盘卖出 --------
    for s in list(positions.keys()):
        m = metrics.get(s)
        if m is None:
            continue
        if m['price'] < m['ma5']:
            order_target_value(s, 0)
            log.info('[STEP2 收盘破MA5清仓] %s 收%.2f < MA5 %.2f' %
                     (s, m['price'], m['ma5']))

    # -------- STEP 3：RSI 风控（对剩余持仓）→ 当日收盘卖出 --------
    # 规则：RSI(14) > 80 且当日涨幅 < 10% → 直接清仓
    # （涨幅 ≥ 10% 接近涨停，视为强势，保留不清）
    for s in list(context.portfolio.positions.keys()):
        m = metrics.get(s)
        if m is None or m['rsi'] is None:
            continue
        if m['rsi'] > g.rsi_reduce and m['gain'] < g.rsi_gain_cap:
            order_target_value(s, 0)
            log.info('[STEP3 RSI>%d且涨幅<%.0f%%收盘清仓] %s RSI=%.1f 涨幅%.2f%%' %
                     (g.rsi_reduce, g.rsi_gain_cap * 100, s, m['rsi'], m['gain'] * 100))

    # -------- STEP 4：轮动调仓（切出=当日收盘卖，换入=次日开盘买）--------
    rotation_adjust(context, metrics)

    # -------- 补仓：有空余仓位时，生成次日开盘买入计划 --------
    plan_fill_positions(context, metrics)

    if g.to_buy:
        log.info('[计划] 次日开盘拟买入 %d 只：%s' %
                 (len(g.to_buy), [s for s, _ in g.to_buy]))


def is_eligible_target(m):
    """买入/轮动候选合格：换手率≥3%、站上MA5、RSI<80、涨幅<10%（全部取当日收盘数据）。"""
    if m is None or m['rsi'] is None:
        return False
    return (m['turnover'] >= g.turnover_min and   # 当日换手率 ≥ 3%
            m['price'] >= m['ma5'] and            # 收盘价 ≥ MA5
            m['rsi'] < g.rsi_overbought and       # 收盘RSI < 80
            m['gain'] < g.rsi_gain_cap)           # 涨幅 < 10%(未涨停)


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

    # 【现金优先】账户有可用现金 → 不做切换，交由补仓在次日开盘用现金买入合格标的
    if context.portfolio.available_cash >= g.cash_threshold:
        log.info('[STEP4] 账户有现金%.0f元(≥%.0f) → 不切换，改用现金次日开盘买入' %
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
        tgt = candidates.pop(0)  # 涨幅最小的合格标的
        src_value = context.portfolio.positions[src].value
        move_value = src_value * g.switch_ratio  # 转移 100%

        # 切出强势持仓（100%）→ 当日收盘卖出
        order_target_value(src, src_value - move_value)
        # 换入目标标的 → 记入次日开盘买入计划（目标市值）
        tgt_cur = context.portfolio.positions[tgt].value if tgt in context.portfolio.positions else 0.0
        plan_buy(tgt, tgt_cur + move_value)
        log.info('[STEP4 切换] %s(涨%.2f%%) 收盘卖出 -%.0f%% → %s(涨%.2f%%) 次日开盘买入 金额%.0f' %
                 (src, metrics[src]['gain'] * 100, g.switch_ratio * 100,
                  tgt, metrics[tgt]['gain'] * 100, move_value))


def plan_fill_positions(context, metrics):
    """空余仓位时，生成次日开盘买入计划，每笔用总资金33%（对应"再次买入条件"）。"""
    positions = context.portfolio.positions
    planned = [s for s, _ in g.to_buy]
    planned_new = [s for s in planned if s not in positions]
    # 空余仓位数 = 上限 - 现有持仓 - 已计划买入的新标的
    slots = g.max_positions - len(positions) - len(planned_new)
    if slots <= 0:
        return
    candidates = [s for s in metrics.keys()
                  if s not in positions and s not in planned
                  and is_eligible_target(metrics[s])]
    candidates.sort(key=lambda x: metrics[x]['gain'])  # 优先买涨幅较小的
    if not candidates:
        return
    # 每笔买入固定使用【总资金的 33%】(3 笔≈满仓，故最多同时持有 3 只)
    target_value = context.portfolio.total_value * g.position_pct
    for s in candidates[:slots]:
        plan_buy(s, target_value)
        log.info('[补仓计划] %s 涨%.2f%% 换手%.2f%% 次日开盘目标市值%.0f(总资金%.0f%%)' %
                 (s, metrics[s]['gain'] * 100, metrics[s]['turnover'],
                  target_value, g.position_pct * 100))


def plan_buy(security, target_value):
    """把买入意图写入次日开盘计划；同一标的取较大的目标市值，避免重复下单。"""
    for i, (s, v) in enumerate(g.to_buy):
        if s == security:
            g.to_buy[i] = (s, max(v, target_value))
            return
    g.to_buy.append((security, target_value))


# ==================== 次日开盘买入（买入=开盘成交） ====================
def morning_buy(context):
    """执行昨日收盘定下的买入计划，于开盘价成交。"""
    if not g.to_buy:
        return
    plan = g.to_buy
    g.to_buy = []  # 取出后清空，避免重复执行
    for idx, (s, target_value) in enumerate(plan):
        if context.portfolio.available_cash < 1000:
            log.info('[开盘买入] 可用现金不足，剩余计划取消：%s' %
                     [x for x, _ in plan[idx:]])
            break
        cd = get_current_data()[s]
        if cd.paused:  # 开盘停牌 → 跳过
            log.info('[开盘买入] %s 停牌，跳过' % s)
            continue
        order_target_value(s, target_value)
        log.info('[开盘买入] %s 目标市值%.0f' % (s, target_value))
