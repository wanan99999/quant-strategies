# -*- coding: utf-8 -*-
"""
海龟交易法则策略（Turtle Trading）—— 第一部分：选股策略 (聚宽 JoinQuant)
========================================================================
职责：构建/更新候选股票池 g.turtle_pool（每周五盘后运行一次）。
本文件同时提供 ATR / 唐奇安通道 等【公共指标函数】，供交易、风控模块调用。

选股三层过滤（对应文字版第一部分）：
  第一层 基础过滤 : 非ST、上市≥250交易日、近20日日均成交额≥2亿、收盘价≥5元
  第二层 波动率筛选: ATR(50) > ATR(300)（波动趋势）、|近20日涨跌幅| > 10%（方向确认）
  第三层 多品种分散: 同一申万一级行业≤2只，最终池 10~20 只，按波动趋势比排序

组合方式（三选一）：
  A. 聚宽【研究环境】上传本文件与"风控"文件，交易文件顶部
     `from 海龟_选股策略 import *` / `from 海龟_风控策略 import *`。
  B. 直接把三个文件内容按【选股→风控→交易】顺序拼进同一个策略文件（函数名无冲突）。
  说明：本文件不含 initialize，导入无副作用。
"""
import numpy as np
from jqdata import *


# ============================ 公共指标函数 ============================
def true_range(high, low, prev_close):
    """真实波幅 TR = Max(高-低, |高-昨收|, |低-昨收|)。"""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def calc_atr(highs, lows, closes, period):
    """Wilder ATR(period)。highs/lows/closes 为等长序列（含足够历史）。"""
    n = len(closes)
    if n < period + 1:
        return None
    trs = [true_range(highs[i], lows[i], closes[i - 1]) for i in range(1, n)]
    if len(trs) < period:
        return None
    atr = float(np.mean(trs[:period]))
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def donchian_high(highs, period, exclude_today=True):
    """唐奇安上轨：过去 period 日最高价的最大值。exclude_today=True 表示不含当日。"""
    seq = highs[:-1] if exclude_today else highs
    if len(seq) < period:
        return None
    return float(np.max(seq[-period:]))


def donchian_low(lows, period, exclude_today=True):
    """唐奇安下轨：过去 period 日最低价的最小值。exclude_today=True 表示不含当日。"""
    seq = lows[:-1] if exclude_today else lows
    if len(seq) < period:
        return None
    return float(np.min(seq[-period:]))


def get_daily_arrays(security, end_dt, count, fields=('high', 'low', 'close')):
    """取日线数组，返回 dict{field: np.array}；数据不足返回 None。"""
    bars = get_bars(security, count=count, unit='1d', fields=list(fields),
                    include_now=True, end_dt=end_dt, df=False)
    if bars is None or len(bars['close']) < 2:
        return None
    return {f: np.array(bars[f], dtype=float) for f in fields}


# ============================ 选股主流程 ============================
def select_universe(context):
    """每周五盘后：重建候选池 g.turtle_pool，并缓存每只标的的行业与指标信息。"""
    p = g.sel
    prev = context.previous_date

    # ---- 基础票池：默认取中证800成分（可在 initialize 里改 g.base_index=None 用全市场）----
    if g.base_index:
        base = get_index_stocks(g.base_index, date=prev)
    else:
        base = list(get_all_securities(types=['stock'], date=prev).index)

    # ---- 一次性算好"上市≥250交易日"的门槛日：第251个交易日之前上市即合格 ----
    td = get_trade_days(end_date=prev, count=p['min_list_days'] + 1)
    list_cutoff = td[0] if len(td) > 0 else prev

    cd = get_current_data()

    # 第一层：基础过滤（ST / 上市时长 / 停牌）
    stage1 = []
    for s in base:
        info = cd[s]
        if info.paused or info.is_st or ('ST' in info.name) or ('退' in info.name):
            continue
        sec_info = get_security_info(s)
        if sec_info is None or sec_info.start_date > list_cutoff:
            continue
        stage1.append(s)

    # 行业映射（申万一级），供分散约束 & 风控行业敞口使用
    ind = get_industry(stage1, date=prev)

    # 第一层(续) + 第二层：流动性 / 价格 / 波动率 / 方向
    candidates = []  # (security, vol_trend, info_dict)
    need = max(p['atr_slow'] + 5, 320)
    for s in stage1:
        arr = get_daily_arrays(s, context.current_dt, need,
                               fields=('high', 'low', 'close', 'money'))
        if arr is None or len(arr['close']) < p['atr_slow'] + 1:
            continue
        highs, lows, closes, money = arr['high'], arr['low'], arr['close'], arr['money']

        # S4 价格 ≥ 5元
        price = closes[-1]
        if price < p['min_price']:
            continue
        # S3 流动性：近20日日均成交额 ≥ 2亿
        if float(np.mean(money[-20:])) < p['min_amount']:
            continue

        # 第二层：波动趋势 ATR(50) > ATR(300)
        atr_fast = calc_atr(highs, lows, closes, p['atr_fast'])
        atr_slow = calc_atr(highs, lows, closes, p['atr_slow'])
        if not atr_fast or not atr_slow or atr_slow <= 0:
            continue
        if atr_fast <= atr_slow:
            continue
        # 第二层：方向确认 |近20日涨跌幅| > 10%
        base_close = closes[-(p['mom_days'] + 1)]
        if base_close <= 0:
            continue
        momentum = abs(price / base_close - 1.0)
        if momentum < p['mom_thresh']:
            continue

        atr20 = calc_atr(highs, lows, closes, g.atr_period)
        if not atr20 or atr20 <= 0:
            continue

        info = {
            'atr20': atr20,                                   # 即 N 值
            'vol_trend': atr_fast / atr_slow,                 # 波动趋势比
            'don_up_20': donchian_high(highs, 20, True),      # 系统1 上轨
            'don_up_55': donchian_high(highs, 55, True),      # 系统2 上轨
            'don_dn_10': donchian_low(lows, 10, True),        # 系统1 退出下轨
            'don_dn_20': donchian_low(lows, 20, True),        # 系统2 退出下轨
            'sector': (ind.get(s, {}) or {}).get('sw_l1', {}).get('industry_code', 'NA'),
            'price': price,
        }
        candidates.append((s, info['vol_trend'], info))

    # 第三层：多品种分散（同行业≤2）+ 池容量 10~20，按波动趋势比降序优选
    candidates.sort(key=lambda x: x[1], reverse=True)
    pool, pool_info, sector_count = [], {}, {}
    for s, _, info in candidates:
        sec = info['sector']
        if sector_count.get(sec, 0) >= p['per_industry_max']:
            continue
        pool.append(s)
        pool_info[s] = info
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(pool) >= p['pool_max']:
            break

    g.turtle_pool = pool
    g.pool_info = pool_info
    log.info('[选股] 候选池更新：%d只（合格候选%d，行业分布%s）' %
             (len(pool), len(candidates), sector_count))
    if len(pool) < p['pool_min']:
        log.warn('[选股] 候选池不足%d只，当前仅%d只（可放宽过滤或扩大基础票池）' %
                 (p['pool_min'], len(pool)))


def get_sector(security):
    """取某标的申万一级行业代码（优先用选股缓存，缺失时实时查询）。"""
    info = getattr(g, 'pool_info', {}).get(security)
    if info and info.get('sector'):
        return info['sector']
    ind = get_industry(security)
    return (ind.get(security, {}) or {}).get('sw_l1', {}).get('industry_code', 'NA')
