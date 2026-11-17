# -*- coding: utf-8 -*-
"""
双均线/三均线交叉策略（MA Crossover）—— 第二部分：交易策略 (聚宽 JoinQuant)【主策略入口】
========================================================================
职责：按每只标的的适配子策略（双均线 / 三均线）执行入场、出场、仓位调整，
      并串联选股与风控。本文件含 initialize，是回测/实盘主文件。

双均线(MA10/MA20/MA60)：金叉+MA60向上+价>MA10 入场；死叉/价<MA20/价<MA60 出场
三均线(MA5/MA10/MA20/MA60)：三线多头+新鲜金叉+MA20>MA60 入场；
      排列破坏(MA5<MA10)减半，双死叉/价<MA20/价<MA60 清仓
仓位调整：|MA5-MA20|乖离>8% 或 三均线粘合(最大差<2%) → 减仓至半仓
成交时点：每日【收盘】评估 → 出场/减仓当日收盘成交；入场→次日开盘成交。

★ 回测建议【分钟】频率，'open'/'close' 定时点精准撮合（日线频率亦可用）。
"""
import numpy as np
from jqdata import *

try:
    from 均线_选股策略 import *
    from 均线_风控策略 import *
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

    g.base_index = '000906.XSHG'    # 中证800为基础票池（None=全市场，较慢）
    g.market_index = '000001.XSHG'  # 大盘环境用上证指数

    # ---- 选股参数（第一部分）----
    g.sel = {
        'min_list_days': 120, 'min_amount': 1e8, 'min_price': 3.0,
        'per_industry_max': 2, 'pool_min': 5, 'pool_max': 10,
    }

    # ---- 交易参数（第二部分）----
    g.dual = {'fast': 10, 'slow': 20, 'filter': 60,
              'pos': 0.18, 'max_stocks': 6, 'total_cap': 0.80,
              'stop': -0.08, 'time_stop': 15}
    g.triple = {'s': 5, 'm': 10, 'l': 20, 'filter': 60,
                'pos': 0.22, 'max_stocks': 4, 'total_cap': 1.00,
                'stop': -0.06, 'time_stop': 10}
    g.bias_cap = 0.08               # MA5/MA20 乖离>8% 减半
    g.tangle_gap = 0.02             # 三均线最大差<2% 视为粘合，减半
    g.max_total_weight = 0.98       # 账户总仓位上限

    # ---- 风控参数（第三部分）----
    g.risk = {
        'adx_min': 20, 'tangle_cross': 4, 'boll_min': 0.05,
        'month_dd_pause': 0.08, 'max_dd_clear': 0.15,
        'max_per_sector_stocks': 2, 'max_per_sector_weight': 0.30,
        'max_per_type_weight': 0.50,
    }

    # ---- 运行时状态 ----
    g.pool, g.pool_info = [], {}
    g.trades = {}            # {sec: {'adapt','entry_price','entry_date','sector','sector_type','half_done'}}
    g.buy_plan = []          # 次日开盘买入
    g.sell_open_plan = []    # 次日开盘强制清仓（跌停无法当日卖出）

    init_risk_state(context)

    run_weekly(select_universe, weekday=5, time='after_close',
               reference_security='000300.XSHG')
    run_daily(ma_open_exec, time='open', reference_security='000300.XSHG')
    run_daily(ma_close_eval, time='close', reference_security='000300.XSHG')


# ==================== 开盘执行：强制清仓 + 计划买入 ====================
def ma_open_exec(context):
    for s in g.sell_open_plan:
        if s in context.portfolio.positions:
            order_target_value(s, 0)
            log.info('[开盘 强制清仓] %s（前日跌停无法卖出）' % s)
        g.trades.pop(s, None)
    g.sell_open_plan = []

    for plan in g.buy_plan:
        s = plan['sec']
        cd = get_current_data()[s]
        if cd.paused:
            log.info('[开盘买入] %s 停牌，跳过' % s); continue
        price = cd.last_price
        if not price or np.isnan(price):
            continue
        if context.portfolio.available_cash < plan['target_value'] * 0.3:
            log.info('[开盘买入] %s 现金不足，跳过' % s); continue
        order_target_value(s, plan['target_value'])
        g.trades[s] = {
            'adapt': plan['adapt'], 'entry_price': price,
            'entry_date': context.current_dt, 'sector': plan['sector'],
            'sector_type': plan['sector_type'], 'half_done': False,
        }
        log.info('[开盘入场] %s 子策略=%s 价%.2f 目标仓位%.0f元' %
                 (s, plan['adapt'], price, plan['target_value']))
    g.buy_plan = []


# ==================== 收盘评估：出场 / 减仓 / 入场 ====================
def ma_close_eval(context):
    update_risk_state(context)
    g.buy_plan = []

    if not g.pool:
        select_universe(context)

    # ---------- 大盘 / 账户级熔断 ----------
    regime = market_regime(context)
    if regime['action'] == 'clear' or account_should_clear(context):
        reason = '上证<MA20清仓' if regime['action'] == 'clear' else '最大回撤>15%清仓'
        _clear_all(context, reason)
        return
    allow_open = (regime['action'] in ('normal', 'reduce'))
    if allow_open:
        blocked, why = account_open_blocked(context)
        if blocked:
            allow_open = False
            log.info('[开仓暂停] %s' % why)
    elif regime['action'] == 'no_open':
        log.info('[开仓暂停] 上证<MA60，仅平仓')

    # ---------- A. 持仓出场 / 减仓（当日收盘成交）----------
    for s in list(g.trades.keys()):
        t = g.trades[s]
        A = get_daily_arrays(s, context.current_dt, 90, fields=('high', 'low', 'close'))
        if A is None:
            continue
        action, reason = _eval_exit(context, s, t, A)
        if action is None:
            continue
        cd = get_current_data()[s]
        limit_down = cd.low_limit and A['close'][-1] <= cd.low_limit + 0.001
        if action == 'clear':
            if cd.paused or limit_down:
                g.sell_open_plan.append(s)
                log.info('[出场挂起] %s %s（跌停/停牌，次日开盘清仓）' % (s, reason))
            else:
                ret = A['close'][-1] / t['entry_price'] - 1.0
                order_target_value(s, 0)
                g.trades.pop(s, None)
                record_trade_result(context, ret)
                log.info('[收盘清仓] %s %s 收益%.1f%%' % (s, reason, ret * 100))
        elif action == 'half' and not t['half_done']:
            pos = context.portfolio.positions.get(s)
            if pos is not None and not (cd.paused or limit_down):
                order_target_value(s, pos.value * 0.5)
                t['half_done'] = True
                log.info('[收盘减半] %s %s' % (s, reason))

    # ---------- B. 通道突破入场 ----------
    if not allow_open:
        return
    held_by_adapt = _count_by_adapt(context)
    cur_total_w = _total_weight(context)

    for s in g.pool:
        if s in g.trades or s in [p['sec'] for p in g.buy_plan]:
            continue
        info = g.pool_info.get(s, {})
        adapt = info.get('adapt')
        if adapt is None:
            continue
        cap = g.dual if adapt == 'dual' else g.triple
        if held_by_adapt.get(adapt, 0) >= cap['max_stocks']:
            continue

        A = get_daily_arrays(s, context.current_dt, 90, fields=('high', 'low', 'close'))
        if A is None:
            continue
        choppy, cw = stock_choppy_block(context, s, A)
        if choppy:
            log.info('[入场过滤] %s %s' % (s, cw)); continue

        cl = A['close']; price = cl[-1]
        m5, m10, m20, m60 = ma(cl, 5), ma(cl, 10), ma(cl, 20), ma(cl, 60)
        if None in (m5, m10, m20, m60):
            continue

        if adapt == 'dual':
            # 允许最近2日内的金叉（容忍周频池子与单日事件的错位）
            sig = (fresh_golden(cl, 10, 20, within=2) and slope_up(cl, 60, 5) and price > m10)
        else:  # triple
            sig = (m5 > m10 > m20 and fresh_golden(cl, 5, 10, within=3)
                   and m20 > m60 and slope_up(cl, 60, 5) and price > m5)
        if not sig:
            continue

        # 仓位与集中度约束
        scale = regime['scale_dual'] if adapt == 'dual' else regime['scale_triple']
        weight = cap['pos'] * scale
        if weight <= 0:
            continue
        sector, stype = info.get('sector', 'NA'), info.get('sector_type', 'cycle')
        if not concentration_ok(context, sector, stype, weight):
            log.info('[入场过滤] %s 行业/类型集中度超限' % s); continue
        if cur_total_w + weight > g.max_total_weight:
            continue

        target_value = context.portfolio.total_value * weight
        g.buy_plan.append({'sec': s, 'target_value': target_value, 'adapt': adapt,
                           'sector': sector, 'sector_type': stype})
        held_by_adapt[adapt] = held_by_adapt.get(adapt, 0) + 1
        cur_total_w += weight
        log.info('[入场计划] %s 子策略=%s 金叉确认 拟仓位%.0f%%(次日开盘)' %
                 (s, adapt, weight * 100))

    if g.buy_plan:
        log.info('[计划汇总] 次日开盘买入%d只：%s' %
                 (len(g.buy_plan), [p['sec'] for p in g.buy_plan]))


# ==================== 出场/减仓判定 ====================
def _eval_exit(context, s, t, A):
    """返回 ('clear'|'half'|None, reason)。整合风控止损 + 策略出场 + 仓位调整。"""
    # 1) 风控：硬止损 / 时间止损 / 均线止损
    action, reason = position_risk_exit(context, s, t, A)
    if action:
        return action, reason

    cl = A['close']; price = cl[-1]
    m5, m10, m20, m60 = ma(cl, 5), ma(cl, 10), ma(cl, 20), ma(cl, 60)
    if None in (m5, m10, m20, m60):
        return None, ''
    adapt = t['adapt']

    # 2) 策略清仓
    if price < m60:
        return 'clear', '收盘<MA60'
    if adapt == 'dual':
        if death_cross(cl, 10, 20):
            return 'clear', 'MA10死叉MA20'
    else:  # triple
        if price < m20:
            return 'clear', '收盘<MA20'
        if m5 < m10 < m20:
            return 'clear', '三线双死叉'

    # 3) 仓位调整（减半）：乖离过热 / 均线粘合 / 三均线排列破坏
    bias = (m5 - m20) / m20 if m20 > 0 else 0.0
    if abs(bias) > g.bias_cap:
        return 'half', 'MA5/MA20乖离%.1f%%>8%%' % (bias * 100)
    mmax, mmin = max(m5, m10, m20), min(m5, m10, m20)
    if mmin > 0 and (mmax - mmin) / mmin < g.tangle_gap:
        return 'half', '三均线粘合(<2%)方向不明'
    if adapt == 'triple' and m5 < m10:
        return 'half', 'MA5<MA10排列破坏'
    return None, ''


# ==================== 工具 ====================
def _clear_all(context, reason):
    for s in list(context.portfolio.positions.keys()):
        cd = get_current_data()[s]
        if cd.paused:
            g.sell_open_plan.append(s); continue
        t = g.trades.get(s)
        if t:
            ret = cd.last_price / t['entry_price'] - 1.0
            record_trade_result(context, ret)
        order_target_value(s, 0)
        g.trades.pop(s, None)
    log.info('[账户熔断] %s，清仓完毕' % reason)


def _count_by_adapt(context):
    res = {}
    for t in g.trades.values():
        res[t['adapt']] = res.get(t['adapt'], 0) + 1
    return res


def _total_weight(context):
    tv = context.portfolio.total_value
    if tv <= 0:
        return 0.0
    return sum(p.value for p in context.portfolio.positions.values()) / tv
