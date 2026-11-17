# -*- coding: utf-8 -*-
"""
海龟交易法则策略（Turtle Trading）—— 第二部分：交易策略 (聚宽 JoinQuant) 【主策略入口】
========================================================================
职责：入场（通道突破）、金字塔加仓、出场（2N止损 / 通道退出），并串联选股与风控。
本文件含 initialize，是回测/实盘的主文件。

交易系统（对应文字版第二部分）：
  系统1(短): 突破20日高点做多，跌破10日低点退出
  系统2(长): 突破55日高点做多，跌破20日低点退出
  仓位     : 1单位=总资金×1%/N（N=ATR20），金字塔每+0.5N加1单位，单票≤4单位
  出场     : 2N硬止损；或触发对应系统的通道退出
成交时点（A股适配）：
  · 每日【收盘】用当日完整K线评估：出场→当日收盘卖出；入场/加仓→记入计划。
  · 计划买单于【次日开盘】成交（符合"次日开盘买入"）。
  · 持仓跌停无法卖出→改为次日开盘强制清仓。

★ 回测请设为【分钟】频率，'open'/'close' 定时点方能精准撮合（日线频率亦可用，
  卖单走当日收盘价、买单走次日开盘价）。
"""
import numpy as np
from jqdata import *

try:  # 研究环境模块化导入；若三文件已拼进同一策略文件则此行忽略
    from 海龟_选股策略 import *
    from 海龟_风控策略 import *
except Exception:
    pass


# ============================ 初始化 ============================
def initialize(context):
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_benchmark('000300.XSHG')
    set_slippage(FixedSlippage(0.002))
    set_order_cost(OrderCost(
        open_commission=0.0003, close_commission=0.0003,
        min_commission=5, close_today_commission=0), type='stock')
    log.set_level('order', 'error')

    # ---- 基础票池 & 基准 ----
    g.base_index = '000906.XSHG'   # 中证800为基础票池（设 None 则全市场，较慢）
    g.market_index = '000300.XSHG' # 市场环境判断用指数
    g.atr_period = 20              # N = ATR(20)

    # ---- 选股参数（第一部分）----
    g.sel = {
        'min_list_days': 250, 'min_amount': 2e8, 'min_price': 5.0,
        'atr_fast': 50, 'atr_slow': 300, 'mom_days': 20, 'mom_thresh': 0.10,
        'per_industry_max': 2, 'pool_min': 10, 'pool_max': 20,
    }

    # ---- 交易参数（第二部分）----
    g.sys1_entry, g.sys1_exit = 20, 10
    g.sys2_entry, g.sys2_exit = 55, 20
    g.add_step_atr = 0.5           # 每0.5N加一次仓

    # ---- 风控参数（第三部分）----
    g.risk = {
        'unit_risk_pct': 0.01,     # 1单位=总资金1%/N
        'stop_atr': 2.0,           # 2N硬止损
        'max_units_per_stock': 4,  # 单票≤4单位(8%)
        'max_units_per_sector': 6, # 同行业≤6单位(12%)
        'max_units_total': 10,     # 全账户≤10单位(20%)
        'dd_pause': 0.20,          # 回撤>20%暂停开仓
        'month_loss_pause': 0.10,  # 月亏>10%停开新仓
    }

    # ---- 运行时状态 ----
    g.turtle_pool = []      # 候选池
    g.pool_info = {}        # 候选池指标缓存
    g.trades = {}           # 持仓台账 {sec: {...}}
    g.failed_breakout = {}  # 系统1假突破标记 {sec: True}
    g.buy_plan = []         # 次日开盘买入计划
    g.sell_open_plan = []   # 次日开盘强制清仓（跌停无法当日卖出）

    init_risk_state(context)

    # ---- 定时任务 ----
    run_weekly(weekly_tasks, weekday=5, time='after_close',
               reference_security='000300.XSHG')          # 每周五盘后：选股+止损上移
    run_daily(turtle_open_exec, time='open',
              reference_security='000300.XSHG')            # 开盘：执行买单/强制清仓
    run_daily(turtle_close_eval, time='close',
              reference_security='000300.XSHG')            # 收盘：评估出场/入场/加仓


def weekly_tasks(context):
    select_universe(context)      # 第一部分：重建候选池
    weekly_trailing_stop(context) # 第三部分：止损仅上移


# ==================== 开盘执行：强制清仓 + 计划买入 ====================
def turtle_open_exec(context):
    # 1) 昨日标记的跌停持仓 → 开盘强制清仓
    for s in g.sell_open_plan:
        if s in context.portfolio.positions:
            order_target_value(s, 0)
            log.info('[开盘 强制清仓] %s（前日跌停无法卖出）' % s)
        g.trades.pop(s, None)
        g.failed_breakout.pop(s, None)
    g.sell_open_plan = []

    # 2) 执行买入计划（开盘价成交）
    for plan in g.buy_plan:
        s = plan['sec']
        cd = get_current_data()[s]
        if cd.paused:
            log.info('[开盘买入] %s 停牌，跳过' % s)
            continue
        price = cd.last_price
        if not price or np.isnan(price):
            continue
        shares = plan['shares']
        if shares < 100:
            continue
        if context.portfolio.available_cash < shares * price * 1.002:
            log.info('[开盘买入] %s 现金不足，跳过' % s)
            continue

        order(s, shares)
        N, sector = plan['N'], plan['sector']
        if plan['kind'] == 'open':
            g.trades[s] = {
                'system': plan['system'], 'entry_price': price, 'N': N,
                'units': plan['units'], 'shares_per_unit': plan['shares_per_unit'],
                'last_add_price': price, 'sector': sector,
                'stop_price': price - g.risk['stop_atr'] * N,
            }
            log.info('[开盘入场] 系统%d %s 价%.2f N=%.3f %.1f单位%d股 止损%.2f' %
                     (plan['system'], s, price, N, plan['units'], shares,
                      g.trades[s]['stop_price']))
        else:  # add 加仓
            t = g.trades.get(s)
            if t is None:
                continue
            t['units'] += plan['units']
            t['last_add_price'] = price
            # 加仓后止损上移到"最近加仓价 - 2N"（仅上移）
            t['stop_price'] = max(t['stop_price'], price - g.risk['stop_atr'] * t['N'])
            log.info('[开盘加仓] %s 价%.2f 累计%.1f单位 止损上移至%.2f' %
                     (s, price, t['units'], t['stop_price']))
    g.buy_plan = []


# ==================== 收盘评估：出场 / 入场 / 加仓 ====================
def turtle_close_eval(context):
    update_risk_state(context)
    g.buy_plan = []  # 本轮重建

    if not g.turtle_pool:      # 首次运行（尚未到周五）先建一次池
        select_universe(context)

    # ---------- A. 持仓出场检查（当日收盘卖出）----------
    for s in list(g.trades.keys()):
        t = g.trades[s]
        arr = get_daily_arrays(s, context.current_dt, 80,
                               fields=('high', 'low', 'close'))
        if arr is None:
            continue
        highs, lows, closes = arr['high'], arr['low'], arr['close']
        t_low, t_high, t_close = lows[-1], highs[-1], closes[-1]

        exit_reason = None
        failed = False
        # 2N 硬止损（以当日最低价是否触及止损线判定）
        if t_low <= t['stop_price']:
            exit_reason = '2N止损(止损价%.2f)' % t['stop_price']
            failed = (t['system'] == 1)   # 系统1以2N止损退出 → 记为假突破
        else:
            # 通道退出
            exit_period = g.sys1_exit if t['system'] == 1 else g.sys2_exit
            exit_level = donchian_low(lows, exit_period, exclude_today=True)
            if exit_level is not None and t_low < exit_level:
                exit_reason = '系统%d跌破%d日低点%.2f' % (t['system'], exit_period, exit_level)

        if exit_reason:
            cd = get_current_data()[s]
            limit_down = cd.low_limit and t_close <= cd.low_limit + 0.001
            if cd.paused or limit_down:      # 跌停/停牌无法卖 → 次日开盘强制清仓
                g.sell_open_plan.append(s)
                log.info('[出场挂起] %s %s（当日跌停/停牌，次日开盘清仓）' % (s, exit_reason))
            else:
                order_target_value(s, 0)
                if failed:
                    g.failed_breakout[s] = True
                g.trades.pop(s, None)
                log.info('[收盘出场] %s %s' % (s, exit_reason))

    # ---------- B. 金字塔加仓（对仍持有的仓位）----------
    perm = new_open_permission(context)
    for s in list(g.trades.keys()):
        if s in g.sell_open_plan:
            continue
        t = g.trades[s]
        if t['units'] >= g.risk['max_units_per_stock']:
            continue
        arr = get_daily_arrays(s, context.current_dt, 30, fields=('close',))
        if arr is None:
            continue
        price = arr['close'][-1]
        # 价格较上次加仓价再前进0.5N → 加1单位
        if price < t['last_add_price'] + g.add_step_atr * t['N']:
            continue
        if not perm['allow']:
            continue
        if units_available(context, s, t['sector']) <= 0:
            continue
        units = 1.0 * perm['unit_mult']
        shares = int(t['shares_per_unit'] * units // 100 * 100)
        if shares < 100:
            continue
        t['last_add_price'] = price   # 防止同日重复触发；实际成交价在开盘更新
        g.buy_plan.append({'sec': s, 'shares': shares, 'N': t['N'],
                           'system': t['system'], 'kind': 'add', 'units': units,
                           'shares_per_unit': t['shares_per_unit'], 'sector': t['sector']})
        log.info('[加仓计划] %s 现价%.2f≥阈值 拟+%.1f单位%d股(次日开盘)' % (s, price, units, shares))

    # ---------- C. 通道突破入场（对候选池中未持有标的）----------
    if not perm['allow']:
        log.info('[入场暂停] %s' % perm['reason'])
        return

    for s in g.turtle_pool:
        if s in g.trades or s in [p['sec'] for p in g.buy_plan]:
            continue
        arr = get_daily_arrays(s, context.current_dt, 80,
                               fields=('high', 'low', 'close'))
        if arr is None:
            continue
        highs, lows, closes = arr['high'], arr['low'], arr['close']
        t_high = highs[-1]
        up20 = donchian_high(highs, g.sys1_entry, exclude_today=True)
        up55 = donchian_high(highs, g.sys2_entry, exclude_today=True)
        N = calc_atr(highs, lows, closes, g.atr_period)
        if not N or N <= 0:
            continue
        sector = get_sector(s)

        system = None
        if up55 is not None and t_high > up55:
            system = 2                                   # 系统2：突破55日高点
        elif (not perm['only_system2']) and up20 is not None and t_high > up20:
            if g.failed_breakout.get(s):                 # 假突破过滤：跳过本次系统1
                g.failed_breakout[s] = False
                log.info('[入场过滤] %s 上次系统1为假突破，跳过本次系统1信号' % s)
                continue
            system = 1                                   # 系统1：突破20日高点
        if system is None:
            continue

        if units_available(context, s, sector) <= 0:
            continue
        shares_per_unit = unit_shares(context, N)
        units = 1.0 * perm['unit_mult']
        shares = int(shares_per_unit * units // 100 * 100)
        if shares < 100:
            continue
        g.buy_plan.append({'sec': s, 'shares': shares, 'N': N, 'system': system,
                           'kind': 'open', 'units': units,
                           'shares_per_unit': shares_per_unit, 'sector': sector})
        log.info('[入场计划] 系统%d %s 当日高%.2f 突破 N=%.3f 拟%.1f单位%d股(次日开盘)' %
                 (system, s, t_high, N, units, shares))

    if g.buy_plan:
        log.info('[计划汇总] 次日开盘买单%d笔：%s' %
                 (len(g.buy_plan), [(p['sec'], p['kind']) for p in g.buy_plan]))
