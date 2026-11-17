# -*- coding: utf-8 -*-
"""
双均线/三均线交叉策略（MA Crossover）—— 第三部分：风控策略 (聚宽 JoinQuant)
========================================================================
职责：提供单笔止损、震荡市过滤、账户熔断、大盘环境、行业集中度等风控函数，
      供交易模块调用。本文件不含 initialize，导入无副作用。

风控要点（对应文字版第三部分）：
  3.1 单笔    : 硬止损(双-8%/三-6%)、时间止损(双15日/三10日未盈利)、均线止损(双<MA20/三<MA10)
  3.2 震荡过滤: ADX(14)<20暂停开仓；近20日MA5/MA20交叉≥4次暂停该标的；布林带宽/MA20<5%暂停
  3.3 账户    : 总仓位上限(双80%/三100%)；连亏3笔暂停3日；月回撤>8%停开；最大回撤>15%清仓
  3.4 大盘    : 上证>MA60且向上=正常；>MA60走平=降仓；<MA60=只平仓；<MA20=清仓
  3.5 行业    : 同申万一级≤2只且≤30%；同类型(成长/周期/主题)合计≤50%
"""
import numpy as np
from jqdata import *

try:
    from 均线_选股策略 import ma, slope_up, calc_adx, cross_count, boll_bandwidth, get_daily_arrays
except Exception:
    pass


# ============================ 账户级状态 ============================
def init_risk_state(context):
    tv = context.portfolio.total_value
    g.peak_value = tv
    g.month_key = (context.current_dt.year, context.current_dt.month)
    g.month_start_value = tv
    g.loss_streak = 0            # 连续亏损笔数
    g.pause_open_until = None    # 暂停开仓截止日（含）


def update_risk_state(context):
    tv = context.portfolio.total_value
    g.peak_value = max(getattr(g, 'peak_value', tv), tv)
    key = (context.current_dt.year, context.current_dt.month)
    if key != getattr(g, 'month_key', None):
        g.month_key = key
        g.month_start_value = tv


def record_trade_result(context, ret):
    """一笔平仓后调用：更新连亏计数；连亏≥3 → 暂停开仓3个交易日。"""
    if ret < 0:
        g.loss_streak = getattr(g, 'loss_streak', 0) + 1
    else:
        g.loss_streak = 0
    if g.loss_streak >= 3:
        fdays = get_trade_days(start_date=context.current_dt, count=4)
        g.pause_open_until = fdays[-1] if len(fdays) >= 4 else None
        g.loss_streak = 0
        log.info('[风控] 连亏3笔 → 暂停开仓至 %s' % g.pause_open_until)


# ============================ 账户熔断查询 ============================
def account_should_clear(context):
    """最大回撤 > 15% → 清仓所有持仓。"""
    tv = context.portfolio.total_value
    peak = getattr(g, 'peak_value', tv)
    return peak > 0 and (peak - tv) / peak > g.risk['max_dd_clear']


def account_open_blocked(context):
    """账户级是否禁止新开仓（连亏暂停期 / 当月回撤>8%）。返回(bool, reason)。"""
    d = context.current_dt.date()
    pu = getattr(g, 'pause_open_until', None)
    if pu is not None and d <= (pu.date() if hasattr(pu, 'date') else pu):
        return True, '连亏暂停期内'
    tv = context.portfolio.total_value
    ms = getattr(g, 'month_start_value', tv)
    if ms > 0 and (tv - ms) / ms < -g.risk['month_dd_pause']:
        return True, '当月回撤>%.0f%%' % (g.risk['month_dd_pause'] * 100)
    return False, ''


# ============================ 大盘环境 ============================
def market_regime(context):
    """返回 {'action': 'normal'/'reduce'/'no_open'/'clear', 'scale_dual','scale_triple'}。"""
    arr = get_daily_arrays(g.market_index, context.current_dt, 90, fields=('close',))
    if arr is None:
        return {'action': 'normal', 'scale_dual': 1.0, 'scale_triple': 1.0}
    cl = arr['close']
    price = cl[-1]
    m20, m60 = ma(cl, 20), ma(cl, 60)
    if m20 is None or m60 is None:
        return {'action': 'normal', 'scale_dual': 1.0, 'scale_triple': 1.0}

    if price < m20:
        return {'action': 'clear', 'scale_dual': 0.0, 'scale_triple': 0.0}
    if price < m60:
        return {'action': 'no_open', 'scale_dual': 0.0, 'scale_triple': 0.0}
    # price > m60
    if slope_up(cl, 60, 5):     # MA60 向上 → 正常
        return {'action': 'normal', 'scale_dual': 1.0, 'scale_triple': 1.0}
    # MA60 走平 → 降仓
    return {'action': 'reduce', 'scale_dual': 0.5, 'scale_triple': 0.6}


# ============================ 震荡市过滤（个股级） ============================
def stock_choppy_block(context, security, A):
    """A 为含 high/low/close 的数组字典。返回(bool, reason)。"""
    highs, lows, closes = A['high'], A['low'], A['close']
    adx = calc_adx(highs, lows, closes, 14)
    if adx is not None and adx < g.risk['adx_min']:
        return True, 'ADX%.1f<20无趋势' % adx
    if cross_count(closes, 5, 20, window=20) >= g.risk['tangle_cross']:
        return True, '近20日MA5/MA20缠绕≥4次'
    bw = boll_bandwidth(closes, 20, 2)
    if bw is not None and bw < g.risk['boll_min']:
        return True, '布林带宽%.1f%%<5%%' % (bw * 100)
    return False, ''


# ============================ 行业集中度 ============================
def concentration_ok(context, sector, sector_type, add_weight):
    """加入 add_weight 后是否仍满足：同行业≤2只且≤30%、同类型≤50%。"""
    tv = context.portfolio.total_value
    trades = getattr(g, 'trades', {})
    same_sec = [t for t in trades.values() if t.get('sector') == sector]
    if len(same_sec) >= g.risk['max_per_sector_stocks']:
        return False
    sec_w = sum(t_weight(context, s) for s, t in trades.items() if t.get('sector') == sector)
    if sec_w + add_weight > g.risk['max_per_sector_weight']:
        return False
    type_w = sum(t_weight(context, s) for s, t in trades.items()
                 if t.get('sector_type') == sector_type)
    if type_w + add_weight > g.risk['max_per_type_weight']:
        return False
    return True


def t_weight(context, security):
    pos = context.portfolio.positions.get(security)
    tv = context.portfolio.total_value
    if pos is None or tv <= 0:
        return 0.0
    return pos.value / tv


# ============================ 单笔风险出场（硬/时间/均线止损） ============================
def position_risk_exit(context, security, t, A):
    """返回('clear', reason) 或 (None, '')。A 含 close 序列与均线。"""
    closes = A['close']
    price = closes[-1]
    adapt = t['adapt']
    stop = g.dual['stop'] if adapt == 'dual' else g.triple['stop']
    time_stop = g.dual['time_stop'] if adapt == 'dual' else g.triple['time_stop']

    # 硬止损
    if t['entry_price'] > 0 and (price / t['entry_price'] - 1.0) <= stop:
        return 'clear', '硬止损%.0f%%' % (stop * 100)
    # 时间止损：持仓达 N 日仍未盈利
    held_days = len(get_trade_days(start_date=t['entry_date'], end_date=context.previous_date))
    if held_days >= time_stop and price <= t['entry_price']:
        return 'clear', '时间止损(持%d日未盈利)' % held_days
    # 均线止损：双<MA20 / 三<MA10
    if adapt == 'dual':
        m20 = ma(closes, 20)
        if m20 is not None and price < m20:
            return 'clear', '收盘<MA20'
    else:
        m10 = ma(closes, 10)
        if m10 is not None and price < m10:
            return 'clear', '收盘<MA10'
    return None, ''
