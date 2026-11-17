# -*- coding: utf-8 -*-
"""
海龟交易法则策略（Turtle Trading）—— 第三部分：风控策略 (聚宽 JoinQuant)
========================================================================
职责：提供仓位额度、账户/市场级熔断、止损追踪等风控函数，供交易模块调用。
本文件不含 initialize，导入无副作用。

风控要点（对应文字版第三部分）：
  3.1 单笔风险 : 1单位=2N止损≈总资金2% → 单票≤4单位(8%)，同行业≤6单位(12%)
  3.2 账户风险 : 总敞口≤10单位(20%)；回撤>20%暂停开仓；月亏>10%当月停开新仓
  3.3 市场环境 : 指数波动萎缩(ATR50<ATR300)→仅系统2；极端波动→新开单位数减半；
                 持仓跌停无法卖出→次日开盘清仓（在交易模块执行）
  3.4 资金曲线 : 盈利复利(1单位=最新总资金×1%/N)；每周五更新N并【上移】止损
"""
import numpy as np
from jqdata import *

try:  # 研究环境模块化导入；若三文件已拼进同一策略文件则此行忽略
    from 海龟_选股策略 import calc_atr, get_daily_arrays, get_sector
except Exception:
    pass


# ============================ 账户级状态维护 ============================
def init_risk_state(context):
    tv = context.portfolio.total_value
    g.peak_value = tv                                   # 资金曲线峰值（算回撤）
    g.month_key = (context.current_dt.year, context.current_dt.month)
    g.month_start_value = tv                            # 当月初始资金（算月亏）


def update_risk_state(context):
    """每日评估前调用：更新资金峰值与当月基准。"""
    tv = context.portfolio.total_value
    g.peak_value = max(getattr(g, 'peak_value', tv), tv)
    key = (context.current_dt.year, context.current_dt.month)
    if key != getattr(g, 'month_key', None):
        g.month_key = key
        g.month_start_value = tv


# ============================ 市场环境判断 ============================
def market_vol_shrink(context):
    """指数波动萎缩：ATR(50) < ATR(300) → 仅允许系统2。"""
    arr = get_daily_arrays(g.market_index, context.current_dt, 320,
                           fields=('high', 'low', 'close'))
    if arr is None:
        return False
    fast = calc_atr(arr['high'], arr['low'], arr['close'], g.sel['atr_fast'])
    slow = calc_atr(arr['high'], arr['low'], arr['close'], g.sel['atr_slow'])
    if not fast or not slow:
        return False
    return fast < slow


def market_extreme_vol(context):
    """极端波动：指数近20日已实现波动率 > 近1年90%分位 → 新开单位数减半。
    （A股无VIX，用指数20日收益率标准差的历史分位近似）"""
    arr = get_daily_arrays(g.market_index, context.current_dt, 300, fields=('close',))
    if arr is None or len(arr['close']) < 260:
        return False
    closes = arr['close']
    rets = np.diff(closes) / closes[:-1]
    win = 20
    roll = np.array([rets[i - win:i].std() for i in range(win, len(rets) + 1)])
    if len(roll) < 60:
        return False
    return roll[-1] > np.percentile(roll[-250:], 90)


# ============================ 开仓许可 & 单位数额度 ============================
def new_open_permission(context):
    """返回是否允许新开仓、是否仅系统2、新开单位乘数。"""
    tv = context.portfolio.total_value
    res = {'allow': True, 'only_system2': False, 'unit_mult': 1.0, 'reason': ''}

    # 3.2 账户回撤 > 20% → 暂停一切新开仓（仅允许平仓）
    peak = getattr(g, 'peak_value', tv)
    if peak > 0 and (peak - tv) / peak > g.risk['dd_pause']:
        res['allow'] = False
        res['reason'] = '账户回撤%.1f%%>20%%，暂停开仓' % ((peak - tv) / peak * 100)
        return res

    # 3.2 当月累计亏损 > 10% → 当月停开新仓
    ms = getattr(g, 'month_start_value', tv)
    if ms > 0 and (tv - ms) / ms < -g.risk['month_loss_pause']:
        res['allow'] = False
        res['reason'] = '当月亏损%.1f%%>10%%，当月停开新仓' % ((tv - ms) / ms * 100)
        return res

    # 3.3 市场环境
    if market_vol_shrink(context):
        res['only_system2'] = True
        res['reason'] = '指数波动萎缩，仅系统2'
    if market_extreme_vol(context):
        res['unit_mult'] = 0.5
        res['reason'] = (res['reason'] + '；' if res['reason'] else '') + '极端波动，新开单位减半'
    return res


def units_available(context, security, sector):
    """在【单票≤4 / 同行业≤6 / 全账户≤10】约束下，该标的还能新增的单位数。"""
    trades = getattr(g, 'trades', {})
    cur_total = sum(t['units'] for t in trades.values())
    cur_stock = trades[security]['units'] if security in trades else 0
    cur_sector = sum(t['units'] for t in trades.values() if t.get('sector') == sector)

    avail = min(g.risk['max_units_per_stock'] - cur_stock,
                g.risk['max_units_per_sector'] - cur_sector,
                g.risk['max_units_total'] - cur_total)
    return max(0, int(avail))


def unit_shares(context, N):
    """1个单位的股数 = 账户总资金×1% / N，向下取整到100股。
    （盈利复利：用最新 total_value；这里以总资金为基准，天然实现复利/缩仓）"""
    if N <= 0:
        return 0
    raw = context.portfolio.total_value * g.risk['unit_risk_pct'] / N
    return int(raw // 100 * 100)


# ============================ 止损维护（每周五盘后） ============================
def weekly_trailing_stop(context):
    """每周重算持仓N值，将止损价【仅上移】：new_stop = 最新收盘 - 2N(当前)。"""
    trades = getattr(g, 'trades', {})
    for s, t in trades.items():
        arr = get_daily_arrays(s, context.current_dt, g.atr_period + 5,
                               fields=('high', 'low', 'close'))
        if arr is None:
            continue
        N_now = calc_atr(arr['high'], arr['low'], arr['close'], g.atr_period)
        if not N_now:
            continue
        trailing = arr['close'][-1] - g.risk['stop_atr'] * N_now
        if trailing > t['stop_price']:
            log.info('[风控 止损上移] %s %.2f → %.2f' % (s, t['stop_price'], trailing))
            t['stop_price'] = trailing
