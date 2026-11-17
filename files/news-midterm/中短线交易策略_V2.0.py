# -*- coding: utf-8 -*-
"""
中短线交易策略 V2.0
========================================================
策略类型：中短线趋势跟踪 + 大盘顶背离风控 + 评分排序 + 调仓换仓
平台：聚宽 JoinQuant

对应《中短线交易策略说明书_V2.0》。相对 V1 的新增：
  1) 选股排序：四层漏斗（硬过滤 → A/B/C 信号分档 → 档内百分制加权评分 → 同分决胜）
  2) 调仓决策：按持仓数×持仓状态×候选档位的决策矩阵（建仓/升级/换仓）
  3) 换仓规则：升级优先；先卖半仓、再卖低分全仓(差≥15)；换仓冷却5日；单日≤2次；半仓保护3日
卖出/止损/顶背离/仓位硬上限沿用 V1 已验证逻辑（含裸生成器 bug 修复）。

执行口径：14:50 盘后决策。买入/加仓/换仓在当日收盘价成交（换仓卖出也在当日收盘，
保证资金当日到位）；顶背离减半/继续跌10%/止损 登记后次日开盘成交。
量能类指标（量比/换手率分位）盘中当日数据不完整，用最近完整交易日近似。
"""

import numpy as np
import pandas as pd
from jqdata import *


# ==================================================================
# 一、参数配置
# ==================================================================
class Cfg:
    BENCHMARK = '000001.XSHG'

    # ---- 仓位（硬上限）----
    MAX_HOLDINGS = 2              # 持仓硬上限（是否买入只看持仓数，不看可用资金比例）
    HALF_RATIO   = 0.25
    FULL_RATIO   = 0.50

    # ---- 股票池 ----
    POOL_MODE   = 'custom'         # 'custom' 自定义消息股池 / 'market' 全市场
    CUSTOM_POOL = [
        # '301387.XSHE',
    ]
    CUSTOM_POOL_APPLY_FILTERS = True

    # ---- 选股过滤（V2: 仅 非ST + 上市满60日；换手率不再作硬过滤）----
    MIN_LIST_DAYS = 60

    # ---- 均线 ----
    MA_SHORT = 5
    MA_LONG  = 10
    MA_MID   = 20                  # 均线发散判定
    MA_POS   = 60                  # 位置安全（距60日线）
    GOLDEN_CONFIRM = 2

    # ---- 调仓/换仓 ----
    SWAP_MARGIN_1HALF = 10         # 1只半仓时：候选A档评分 > 持仓评分 + 10 才换仓
    SWAP_MARGIN_2     = 15         # 2只时：A档评分 - 低分持仓 ≥ 15 才换仓
    MAX_SWAPS_PER_DAY = 2          # 单日换仓上限（对数）
    HALF_PROTECT_BARS = 3          # 半仓保护期（交易日），期内不被换仓（止损照常）

    # ---- 顶背离四条件 ----
    NEW_HIGH_DAYS = 10
    BREADTH_MA    = 5
    WEAK_PCT      = 0.50
    CUM_DOWN_DAYS = 3
    CUM_DOWN_PCT  = 0.50

    # ---- 二级卖出 / 止损 / 冷却 ----
    SELL2_DROP    = 0.10
    STOP_LOSS     = 0.15        # 从 max(买入均价,持仓最高价) 回撤达此比例清仓（即涨到最高点后回撤15%）
    COOLDOWN_DAYS = 5

    # ---- 大盘预警（顶背离触发后禁买窗口）----
    # 顶背离=大盘预警：触发后含当日共 N 个交易日禁止一切买入/加仓/升级/换仓，只允许卖出。
    WARN_BARS     = 10

    # ---- 买入后动能确认（每日判断：未创近N日新高则清仓）----
    BREAKOUT_DAYS     = 3      # 买入满 N 个交易日后开始【每日】检查（设小=更早开始）
    BREAKOUT_LOOKBACK = 3      # 每日对比"前 N 个交易日最高价"

    # ---- 附加 ----
    PAUSE_BUY_ON_DIVERGENCE = False

    # ---- 调试开关（生产默认关闭）----
    DEBUG_HOLDINGS   = False
    DEBUG_DIVERGENCE = False
    DEBUG_SCORE      = False        # 打印候选评分卡
    DEBUG_WATCH      = []


# ==================================================================
# 二、初始化
# ==================================================================
def initialize(context):
    set_benchmark(Cfg.BENCHMARK)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(PriceRelatedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5), type='stock')

    # 持仓：code -> {stage:'full'/'half', half_source:'probe'/'divergence'/None,
    #               half_price, full_price, buy_date, buy_bar, half_bar, highest, reduce_basis}
    g.holdings = {}
    g.cooldown = {}                 # code -> 剩余交易日
    g.pending_sells = []            # 次日开盘执行的卖出
    g.divergence_today = False
    g.warn_left = 0                 # 大盘预警剩余禁买交易日（>0 时禁止一切买入）
    g.custom_pool = list(Cfg.CUSTOM_POOL)
    g.bar = 0                       # 交易日计数（用于半仓保护期）
    g.swaps_today = 0               # 当日已换仓对数

    run_daily(before_market_open, time='before_open')
    run_daily(market_open,        time='09:31')
    run_daily(before_close,       time='14:50')

    log.info('中短线策略 V2.0 初始化完成 [BUILD=v2-release+warn] 股票池模式=%s' % Cfg.POOL_MODE)
    if Cfg.POOL_MODE == 'custom' and not g.custom_pool:
        log.warn('自定义股票池为空：请填入 Cfg.CUSTOM_POOL 或在 update_custom_pool() 中赋值 g.custom_pool。')


# ==================================================================
# 三、盘前
# ==================================================================
def before_market_open(context):
    g.bar += 1
    g.swaps_today = 0
    for code in list(g.cooldown.keys()):
        g.cooldown[code] -= 1
        if g.cooldown[code] <= 0:
            del g.cooldown[code]
    if g.warn_left > 0:                # 大盘预警禁买窗口递减
        g.warn_left -= 1
    g.divergence_today = False
    update_custom_pool(context)


# ==================================================================
# 四、开盘：执行昨日登记的卖出（顶背离减半 / 继续跌10% / 止损）
# ==================================================================
def market_open(context):
    if not g.pending_sells:
        return
    cd = get_current_data()
    retry = []
    for od in g.pending_sells:
        code, action = od['code'], od['action']
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            if action == 'clear':
                g.holdings.pop(code, None)
                g.cooldown[code] = Cfg.COOLDOWN_DAYS
            else:
                g.holdings.pop(code, None)
            continue
        before = pos.total_amount
        order_target(code, od['target_amount'])
        pos2 = context.portfolio.positions.get(code)
        remaining = pos2.total_amount if pos2 is not None else 0
        if action == 'reduce':
            if remaining < before:
                if code in g.holdings:
                    g.holdings[code]['stage'] = 'half'
                    g.holdings[code]['half_source'] = 'divergence'
                    g.holdings[code]['reduce_basis'] = cd[code].day_open
                log.info('[减半->半仓] %s %d->%d 股，减仓价 %.3f' %
                         (code, before, remaining, cd[code].day_open))
            else:
                retry.append(od)
                log.warn('[减半未成交] %s（疑跌停/停牌），下一交易日重试' % code)
        else:  # clear
            if remaining <= 0:
                g.holdings.pop(code, None)
                g.cooldown[code] = Cfg.COOLDOWN_DAYS
                log.info('[清仓] %s（%s）' % (code, od.get('reason', '')))
            else:
                retry.append(od)
                log.warn('[清仓未成交] %s 仍持 %d 股，下一交易日重试' % (code, remaining))
    g.pending_sells = retry


# ==================================================================
# 五、收盘前主流程
# ==================================================================
def before_close(context):
    cd = get_current_data()
    prev = context.previous_date

    _update_holdings(context, cd)
    _reconcile(context, cd)
    _debug_watch(context, cd)

    idx_hist = _hist_close(Cfg.BENCHMARK, prev, 12)
    idx_today = _cur_index(Cfg.BENCHMARK, context.current_dt)
    idx_ok = idx_hist is not None and len(idx_hist) >= 11 and idx_today is not None

    # 1) 大盘顶背离（=大盘预警）——空仓也计算，以便随时检测预警。
    #    c1/c2（上涨+创10日新高）是廉价预过滤，多数日子短路，不增负担。
    divergence = False
    if idx_ok:
        try:
            divergence = _check_divergence(context, cd, idx_hist, idx_today)
        except Exception as e:
            divergence = False
            log.warn('顶背离计算异常，本日按无顶背离处理：%s' % e)
    g.divergence_today = divergence
    if divergence:
        g.warn_left = Cfg.WARN_BARS
        log.info('[大盘预警] 顶背离触发：含当日共 %d 个交易日禁止买入/加仓/换仓，仅允许卖出' % Cfg.WARN_BARS)

    # 2) 卖出评估（顶背离/继续跌10%/止损；登记，次日开盘执行）
    _evaluate_sells(context, cd, divergence)

    if not idx_ok:
        log.warn('上证数据不足：本日跳过买入调仓，仅执行止损/二级卖出')
        return

    # 3) 大盘预警窗口内：禁止一切买入/加仓/升级/换仓，只允许卖出。
    if g.warn_left > 0:
        return

    # 4) 调仓/买入（当日收盘价执行）
    idx_ma5 = (idx_hist[-4:].sum() + idx_today) / 5.0
    try:
        _rebalance(context, cd, idx_hist, idx_today, idx_ma5)
    except Exception as e:
        import traceback
        log.error('[调仓异常] %s\n%s' % (e, traceback.format_exc()))


# ==================================================================
# 六、卖出评估（沿用 V1，含裸生成器修复）
# ==================================================================
def _evaluate_sells(context, cd, divergence):
    # divergence 由 before_close 统一计算并传入（空仓也算，支撑大盘预警检测）。
    for code in list(g.holdings.keys()):
        try:
            _eval_one_sell(context, cd, code, divergence)
        except Exception as e:
            log.warn('持仓 %s 卖出评估异常：%s' % (code, e))


def _eval_one_sell(context, cd, code, divergence):
    pos = context.portfolio.positions.get(code)
    if pos is None or pos.total_amount <= 0:
        g.holdings.pop(code, None)
        return
    if _pending_has(code):
        return
    h = g.holdings[code]
    cur = cd[code].last_price
    avg = _avg_cost(h)
    stop_base = max(avg, h['highest'])
    stop_price = stop_base * (1 - Cfg.STOP_LOSS)

    if Cfg.DEBUG_HOLDINGS:
        dd = (cur / stop_base - 1) * 100 if stop_base else 0.0
        log.info('[持仓诊断] %s 状态=%s/%s 现价=%.2f 均价=%.2f 最高=%.2f 止损价=%.2f 距止损=%.1f%%'
                 % (code, h['stage'], h.get('half_source'), cur, avg, h['highest'], stop_price, dd))

    # 新增卖出条款（最高优先级）：买入后满3个交易日未创近3日新高 -> 清仓
    if _breakout_exit(context, cd, code, h):
        _queue_clear(code, pos, '3日未创近3日新高清仓')
        return

    # 优先级：卖出策略 > 止损策略
    if h['stage'] == 'full':
        if divergence:
            _queue_reduce(code, pos)
        elif cur <= stop_price:
            _queue_clear(code, pos, '止损')
    else:  # half
        if h.get('half_source') == 'divergence':
            if h.get('reduce_basis') and cur <= h['reduce_basis'] * (1 - Cfg.SELL2_DROP):
                _queue_clear(code, pos, '继续下跌10%清仓')
            elif cur <= stop_price:
                _queue_clear(code, pos, '止损')
        else:
            if divergence:
                _queue_clear(code, pos, '顶背离清仓(试探半仓)')
            elif cur <= stop_price:
                _queue_clear(code, pos, '止损')


def _queue_reduce(code, pos):
    target = _round100(pos.total_amount * 0.5)
    g.pending_sells.append({'code': code, 'action': 'reduce', 'target_amount': target})
    log.info('[登记减半] %s 目标 %d 股' % (code, target))


def _queue_clear(code, pos, reason):
    g.pending_sells.append({'code': code, 'action': 'clear', 'target_amount': 0, 'reason': reason})
    log.info('[登记清仓] %s（%s）' % (code, reason))


# ==================================================================
# 七、调仓 / 买入引擎（V2 核心）
# ==================================================================
def _rebalance(context, cd, idx_hist, idx_today, idx_ma5):
    total = context.portfolio.total_value

    # ---- Step 0：半仓大前提（上证站上5日线）；否则不新开/换仓，仅允许持仓自身升级 ----
    idx_above = idx_today > idx_ma5

    # ---- Step 1：持仓信号升级（试探半仓 -> 全仓），优先于任何换仓 ----
    for code in list(g.holdings.keys()):
        h = g.holdings[code]
        if h['stage'] == 'half' and h.get('half_source') == 'probe' and not _pending_has(code):
            ok, _ = _golden_confirm(code, context.previous_date, cd)
            if ok:
                order_target_value(code, Cfg.FULL_RATIO * total)
                h['stage'] = 'full'
                h['full_price'] = cd[code].last_price
                log.info('[全仓确认] %s 升级至满仓 @%.3f' % (code, cd[code].last_price))

    if not idx_above:
        return

    # ---- Step 2：候选评分排序 ----
    cands = _rank_candidates(context, cd, idx_hist, idx_today, idx_ma5)
    if not cands:
        return
    if Cfg.DEBUG_SCORE:
        for c in cands[:5]:
            log.info('[评分] %s 档=%s 总分=%.1f 趋势=%.1f 量能=%.1f 位置=%.1f 共振=%.1f 换手=%.2f'
                     % (c['code'], c['tier'], c['score'], c['trend'], c['vol'],
                        c['pos'], c['reso'], c['turnover']))

    # ---- Step 3：当前持仓评分（用于换仓比较）----
    held = list(g.holdings.keys())
    hinfo = {}
    for code in held:
        s = _evaluate_stock(context, code, cd, context.previous_date, idx_hist, idx_today)
        hinfo[code] = {
            'stage': g.holdings[code]['stage'],
            'score': s['score'] if s else 0.0,
            'protected': _half_protected(code),
            'pending': _pending_has(code),
        }

    # ---- Step 4：决策矩阵 ----
    n = len(held)
    if n == 0:
        _rb_empty(context, cd, total, cands)
    elif n == 1:
        _rb_one(context, cd, total, cands, held[0], hinfo)
    else:  # n >= 2（硬上限2）
        _rb_two(context, cd, total, cands, held, hinfo)


# ---------- 决策：空仓 ----------
def _rb_empty(context, cd, total, cands):
    slots = Cfg.MAX_HOLDINGS
    for c in cands:
        if slots <= 0:
            break
        if c['tier'] == 'A':
            _buy_full(context, cd, c['code'], total)
        else:
            _buy_half(context, cd, c['code'], total)
        slots -= 1


# ---------- 决策：持仓1只 ----------
def _rb_one(context, cd, total, cands, hcode, hinfo):
    h = hinfo[hcode]
    if h['pending']:
        return
    top = cands[0]
    if h['stage'] == 'half':
        # 该半仓若可升级，Step1 已处理；此处为不可升级的半仓
        if top['tier'] == 'A':
            if (top['score'] > h['score'] + Cfg.SWAP_MARGIN_1HALF
                    and not h['protected'] and _can_swap()):
                # 换仓：卖出持仓半仓 -> 全仓买入A档最优
                _swap_out(context, cd, hcode)
                _buy_full(context, cd, top['code'], total)
                # 剩余仓位若还有A档第二则全仓
                if len(cands) > 1 and cands[1]['tier'] == 'A':
                    _buy_full(context, cd, cands[1]['code'], total)
            else:
                _buy_half(context, cd, top['code'], total)  # 保留持仓，买第2只半仓
        else:  # B/C
            _buy_half(context, cd, top['code'], total)
    else:  # full
        if top['tier'] == 'A':
            _buy_full(context, cd, top['code'], total)      # 第2只全仓 -> 100%
        else:
            _buy_half(context, cd, top['code'], total)      # 第2只半仓 -> 75%


# ---------- 决策：持仓2只 ----------
def _rb_two(context, cd, total, cands, held, hinfo):
    top = cands[0]
    if top['tier'] != 'A':
        return                      # 候选仅B/C -> 两只已满，不操作
    stages = [hinfo[c]['stage'] for c in held]
    n_half = stages.count('half')

    # 可换出的持仓（排除 pending / 半仓保护期内）
    def swappable(c):
        return (not hinfo[c]['pending']) and (not (hinfo[c]['stage'] == 'half' and hinfo[c]['protected']))

    if n_half == 0:
        # C1：两只全仓。A档评分 - 最低持仓评分 ≥ 15 才换
        low = min(held, key=lambda c: hinfo[c]['score'])
        if swappable(low) and (top['score'] - hinfo[low]['score'] >= Cfg.SWAP_MARGIN_2) and _can_swap():
            _swap_out(context, cd, low)
            _buy_full(context, cd, top['code'], total)
    elif n_half == 2:
        # C3：两只半仓。有A档 -> 卖低分半仓换A档全仓
        cand_halves = [c for c in held if swappable(c)]
        if cand_halves and _can_swap():
            low = min(cand_halves, key=lambda c: hinfo[c]['score'])
            _swap_out(context, cd, low)
            _buy_full(context, cd, top['code'], total)
    else:
        # C2：一全一半。优先卖半仓换A档全仓；否则若全仓评分最低且差≥15则卖全仓
        half_c = [c for c in held if hinfo[c]['stage'] == 'half'][0]
        full_c = [c for c in held if hinfo[c]['stage'] == 'full'][0]
        if swappable(half_c) and _can_swap():
            _swap_out(context, cd, half_c)
            _buy_full(context, cd, top['code'], total)
        elif (swappable(full_c) and hinfo[full_c]['score'] <= hinfo[half_c]['score']
              and (top['score'] - hinfo[full_c]['score'] >= Cfg.SWAP_MARGIN_2) and _can_swap()):
            _swap_out(context, cd, full_c)
            _buy_full(context, cd, top['code'], total)


# ---------- 买入/换仓 动作 ----------
def _buy_full(context, cd, code, total):
    if code in g.holdings or _pending_has(code):
        return
    # 是否买入只看持仓数是否已达上限（由调仓矩阵控制），不看可用资金比例。
    # 资金不足时 order_target_value 自动按可用资金买入（可能不足名义满仓）。
    order_target_value(code, Cfg.FULL_RATIO * total)
    pos = context.portfolio.positions.get(code)
    if pos is None or pos.total_amount <= 0:      # 可用资金≈0 或涨停未成交，不记录空持仓
        log.warn('[全仓买入未成交] %s（可用资金≈0 或涨停）' % code)
        return
    p = cd[code].last_price
    g.holdings[code] = {
        'stage': 'full', 'half_source': 'probe',
        'half_price': p, 'full_price': p,
        'buy_date': context.current_dt.date(), 'buy_bar': g.bar, 'half_bar': g.bar,
        'highest': p, 'reduce_basis': None,
    }
    log.info('[全仓买入] %s @%.3f（直接建满仓）' % (code, p))


def _buy_half(context, cd, code, total):
    if code in g.holdings or _pending_has(code):
        return
    # 只看持仓数是否已达上限；资金不足时按可用资金买入。
    order_target_value(code, Cfg.HALF_RATIO * total)
    pos = context.portfolio.positions.get(code)
    if pos is None or pos.total_amount <= 0:
        log.warn('[半仓买入未成交] %s（可用资金≈0 或涨停）' % code)
        return
    p = cd[code].last_price
    g.holdings[code] = {
        'stage': 'half', 'half_source': 'probe',
        'half_price': p, 'full_price': None,
        'buy_date': context.current_dt.date(), 'buy_bar': g.bar, 'half_bar': g.bar,
        'highest': p, 'reduce_basis': None,
    }
    log.info('[半仓试探] %s @%.3f（25%%）' % (code, p))


def _swap_out(context, cd, code):
    """换仓卖出：当日收盘价直接卖出，进入冷却，计一次换仓。"""
    order_target(code, 0)
    g.holdings.pop(code, None)
    g.cooldown[code] = Cfg.COOLDOWN_DAYS
    g.swaps_today += 1
    log.info('[换仓卖出] %s（冷却%d日）' % (code, Cfg.COOLDOWN_DAYS))


def _can_swap():
    return g.swaps_today < Cfg.MAX_SWAPS_PER_DAY


def _half_protected(code):
    h = g.holdings.get(code)
    if not h or h['stage'] != 'half':
        return False
    return (g.bar - h.get('half_bar', g.bar)) < Cfg.HALF_PROTECT_BARS


# ==================================================================
# 八、候选评分排序（四层漏斗）
# ==================================================================
def _rank_candidates(context, cd, idx_hist, idx_today, idx_ma5):
    prev = context.previous_date
    base = _tradable_universe(context, cd)          # 第一层：硬过滤（非ST/上市≥60日）
    base = [c for c in base if c not in g.holdings
            and c not in g.cooldown and not _pending_has(c)]
    out = []
    for code in base:
        try:
            s = _evaluate_stock(context, code, cd, prev, idx_hist, idx_today)
        except Exception as e:
            log.warn('[评分异常] %s: %s' % (code, e))
            s = None
        if s is None or s['tier'] is None:            # tier=None 表示未触发半仓（未站上5MA）
            continue
        out.append(s)
    if not out:
        return []
    tier_rank = {'A': 0, 'B': 1, 'C': 2}
    # 档间优先 -> 总分 -> 趋势强度 -> 换手率 -> 流通市值小
    out.sort(key=lambda s: (tier_rank[s['tier']], -s['score'], -s['trend'],
                            -s['turnover'], s['circ_mv']))
    return out


def _evaluate_stock(context, code, cd, prev, idx_hist, idx_today):
    """返回评分与信号档位；tier=None 表示当日未站上5MA（不入候选）。"""
    closes = _hist_close(code, prev, Cfg.MA_POS)     # 至昨日的收盘（最多60）
    if closes is None or len(closes) < Cfg.MA_LONG:
        return None
    today = cd[code].last_price
    ser = np.append(closes, today)                    # 含今日
    ma5 = ser[-5:].mean()
    ma10 = ser[-10:].mean()
    ma20 = ser[-20:].mean() if len(ser) >= 20 else ma10
    ma60 = ser[-60:].mean() if len(ser) >= 60 else ma20
    ma5_y = closes[-5:].mean() if len(closes) >= 5 else ma5
    ma10_y = closes[-10:].mean() if len(closes) >= 10 else ma10

    above_ma5 = today > ma5
    gx = (ma5 - ma10) / ma10 if ma10 else 0.0
    golden_t = ma5 > ma10
    golden_y = ma5_y > ma10_y

    # ---- 信号分档（A/B/C）----
    tier = None
    if above_ma5:
        if golden_t and golden_y:
            tier = 'A'                                # 金叉确认2日
        elif golden_t and not golden_y:
            tier = 'B'                                # 金叉刚形成，未满2日
        else:
            tier = 'C'                                # 仅站上5MA
    # 未站上5MA -> tier=None（对持仓评分仍继续算分，仅候选据 tier 过滤）

    # ---- 维度一：趋势强度（35）----
    if gx >= 0.03:      s_angle = 20
    elif gx >= 0.02:    s_angle = 15
    elif gx >= 0.01:    s_angle = 10
    else:               s_angle = 5
    if ma5 > ma10 > ma20:   s_spread = 15
    elif ma5 > ma10:        s_spread = 8
    else:                   s_spread = 3
    trend = s_angle + s_spread

    # ---- 维度二：量能配合（25）----
    turnover, tpct, circ_mv = _turnover_stats(code, prev)
    if tpct >= 0.80:    s_to = 15
    elif tpct >= 0.50:  s_to = 10
    elif tpct >= 0.30:  s_to = 5
    else:               s_to = 2
    vr = _volume_ratio(code, prev)
    if vr >= 1.5:       s_vr = 10
    elif vr >= 1.2:     s_vr = 7
    elif vr >= 1.0:     s_vr = 4
    else:               s_vr = 2
    vol = s_to + s_vr

    # ---- 维度三：位置安全（25）----
    r60 = today / ma60 if ma60 else 1.0
    if 0.95 <= r60 <= 1.10:     s_dev = 15
    elif 1.10 < r60 <= 1.25:    s_dev = 10
    elif 1.25 < r60 <= 1.40:    s_dev = 5
    else:                       s_dev = 2
    ret5 = (today / ser[-6] - 1) if len(ser) >= 6 else 0.0
    if 0 <= ret5 <= 0.08:       s_ret = 10
    elif 0.08 < ret5 <= 0.15:   s_ret = 6
    elif 0.15 < ret5 <= 0.25:   s_ret = 3
    else:                       s_ret = 1
    pos = s_dev + s_ret

    # ---- 维度四：大盘共振（15）----
    same_days, rel = _resonance(ser, idx_hist, idx_today)
    if same_days >= 5:   s_sync = 10
    elif same_days == 4: s_sync = 7
    elif same_days == 3: s_sync = 4
    else:                s_sync = 1
    if rel > 0.05:       s_rel = 5
    elif rel > 0.02:     s_rel = 3
    elif rel >= 0:       s_rel = 2
    else:                s_rel = 1
    reso = s_sync + s_rel

    score = trend + vol + pos + reso
    return {'code': code, 'tier': tier, 'score': float(score),
            'trend': float(trend), 'vol': float(vol), 'pos': float(pos), 'reso': float(reso),
            'turnover': float(turnover), 'circ_mv': float(circ_mv), 'gx': gx}


def _resonance(ser, idx_hist, idx_today):
    """近5日 个股与上证 同向天数 与 相对强度。"""
    idx_ser = np.append(np.array(idx_hist, dtype=float), float(idx_today))
    if len(ser) < 6 or len(idx_ser) < 6:
        return 0, 0.0
    ind_r = np.sign(np.diff(ser[-6:]))
    idx_r = np.sign(np.diff(idx_ser[-6:]))
    same = int(np.sum(ind_r == idx_r))
    ind_ret5 = ser[-1] / ser[-6] - 1
    idx_ret5 = idx_ser[-1] / idx_ser[-6] - 1
    return same, float(ind_ret5 - idx_ret5)


def _turnover_stats(code, prev):
    """返回 (最近完整日换手率, 20日百分位, 流通市值)。盘中用最近完整日近似。"""
    try:
        df = get_valuation(code, end_date=prev, count=20,
                           fields=['turnover_ratio', 'circulating_market_cap'])
        if df is None or len(df) == 0:
            return 0.0, 0.0, 1e12
        tos = df['turnover_ratio'].values.astype(float)
        latest = tos[-1]
        pct = float(np.mean(tos <= latest))          # 百分位
        cmv = float(df['circulating_market_cap'].values[-1])
        return float(latest), pct, cmv
    except Exception as e:
        log.warn('[换手统计异常] %s: %s' % (code, e))
        return 0.0, 0.0, 1e12


def _volume_ratio(code, prev):
    """量比≈最近完整日成交量/前5日均量（盘中当日量不完整，用近似）。"""
    vols = _hist_field(code, None, prev, 'volume', count=6)
    if vols is None or len(vols) < 6:
        return 1.0
    base = np.mean(vols[-6:-1])
    return float(vols[-1] / base) if base > 0 else 1.0


# ==================================================================
# 九、顶背离四条件（沿用 V1）
# ==================================================================
def _check_divergence(context, cd, idx_hist, idx_today):
    c1 = idx_today > idx_hist[-1]
    c2 = idx_today > idx_hist[-(Cfg.NEW_HIGH_DAYS - 1):].max()
    if not (c1 and c2):
        if Cfg.DEBUG_DIVERGENCE:
            log.info('[顶背离] 前置未过 c1=%s c2=%s (今=%.2f 昨=%.2f 前9高=%.2f)'
                     % (c1, c2, idx_today, idx_hist[-1], idx_hist[-(Cfg.NEW_HIGH_DAYS - 1):].max()))
        return False

    uni = _breadth_universe(context, cd)
    if not uni:
        return False
    piv = _hist_close_panel(uni, context.previous_date, Cfg.BREADTH_MA)
    if piv is None or piv.empty or len(piv) < Cfg.CUM_DOWN_DAYS:
        return False
    cols = list(piv.columns)
    today = pd.Series({c: cd[c].last_price for c in cols})

    recent = piv.tail(Cfg.BREADTH_MA - 1)
    ma5 = (recent.sum(min_count=Cfg.BREADTH_MA - 1) + today) / float(Cfg.BREADTH_MA)
    valid5 = ma5.notna() & today.notna()
    if valid5.sum() == 0:
        return False
    weak = ((today < ma5) & valid5).sum() / float(valid5.sum())
    c3 = weak > Cfg.WEAK_PCT

    close_3ago = piv.iloc[-Cfg.CUM_DOWN_DAYS]
    validc = close_3ago.notna() & today.notna() & (close_3ago != 0)
    if validc.sum() == 0:
        return False
    cum = (today - close_3ago) / close_3ago
    down = ((cum < 0) & validc).sum() / float(validc.sum())
    c4 = down > Cfg.CUM_DOWN_PCT

    if Cfg.DEBUG_DIVERGENCE:
        log.info('[顶背离] c1=%s c2=%s c3=%.1f%%>50=%s c4=%.1f%%>50=%s 样本=%d'
                 % (c1, c2, weak * 100, c3, down * 100, c4, int(valid5.sum())))
    result = c1 and c2 and c3 and c4
    if result:
        log.info('★ 顶背离触发：弱势广度=%.1f%% 累计下跌广度=%.1f%%' % (weak * 100, down * 100))
    return result


# ==================================================================
# 十、辅助函数
# ==================================================================
def _golden_confirm(code, prev, cd):
    """全仓/A档条件：MA5>MA10 且 今日与昨日均金叉（连续2日）。"""
    closes = _hist_close(code, prev, 11)
    if closes is None or len(closes) < 10:
        return False, 0.0
    today = cd[code].last_price
    ser = np.append(closes, today)
    ma5_t = ser[-5:].mean()
    ma10_t = ser[-10:].mean()
    ma5_y = closes[-5:].mean()
    ma10_y = closes[-10:].mean()
    ok = (ma5_t > ma10_t) and (ma5_y > ma10_y)
    return ok, ((ma5_t - ma10_t) / ma10_t if ma10_t else 0.0)


def _reconcile(context, cd):
    """未跟踪的实际持仓纳入跟踪，确保止损必覆盖。"""
    for code, pos in context.portfolio.positions.items():
        if pos.total_amount <= 0 or code in g.holdings or _pending_has(code):
            continue
        ac = pos.avg_cost if getattr(pos, 'avg_cost', 0) else cd[code].last_price
        g.holdings[code] = {
            'stage': 'full', 'half_source': None,
            'half_price': ac, 'full_price': ac,
            'buy_date': context.previous_date, 'buy_bar': g.bar, 'half_bar': g.bar,
            'highest': max(ac, cd[code].last_price), 'reduce_basis': None,
        }
        log.warn('[持仓校准] 未跟踪持仓 %s 已纳入(成本价 %.3f)' % (code, ac))


def _update_holdings(context, cd):
    for code in list(g.holdings.keys()):
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            g.holdings.pop(code, None)
            continue
        h = g.holdings[code]
        highs = _hist_field(code, h['buy_date'], context.previous_date, 'high')
        hi = h['highest']
        if highs is not None and len(highs) > 0:
            hi = max(hi, float(np.nanmax(highs)))
        hi = max(hi, cd[code].last_price)
        h['highest'] = hi


def _avg_cost(h):
    if h.get('full_price') is not None:
        return (h['half_price'] + h['full_price']) / 2.0
    return h['half_price']


def _debug_watch(context, cd):
    if not Cfg.DEBUG_WATCH:
        return
    for code in Cfg.DEBUG_WATCH:
        h = g.holdings.get(code)
        stage = ('%s/%s' % (h['stage'], h.get('half_source'))) if h else '未跟踪'
        pos = context.portfolio.positions.get(code)
        amt = pos.total_amount if pos is not None else 0
        s = None
        try:
            idx_hist = _hist_close(Cfg.BENCHMARK, context.previous_date, 12)
            idx_today = _cur_index(Cfg.BENCHMARK, context.current_dt)
            s = _evaluate_stock(context, code, cd, context.previous_date, idx_hist, idx_today)
        except Exception:
            pass
        log.info('[WATCH] %s 跟踪=%s 持股=%s 冷却=%s 档=%s 评分=%.1f'
                 % (code, stage, amt, g.cooldown.get(code, 0),
                    (s['tier'] if s else '-'), (s['score'] if s else 0.0)))


def update_custom_pool(context):
    """每日刷新自定义股票池的钩子。如消息股清单每日变化，在此重写 g.custom_pool。"""
    # g.custom_pool = get_concept_stocks('...', date=context.previous_date)
    pass


def load_stock_pool(context):
    if Cfg.POOL_MODE == 'custom':
        return list(g.custom_pool)
    stocks = get_all_securities(['stock'], date=context.previous_date)
    stocks = stocks[(stocks.index.str.endswith('.XSHG') | stocks.index.str.endswith('.XSHE'))]
    return list(stocks.index)


def _tradable_universe(context, cd):
    """第一层硬过滤：非ST + 上市满60交易日（换手率不再作硬过滤）。"""
    base = load_stock_pool(context)
    if not base:
        return []
    info = get_all_securities(['stock'], date=context.previous_date)
    cutoff = _list_cutoff(context)
    apply_f = (Cfg.POOL_MODE == 'market') or Cfg.CUSTOM_POOL_APPLY_FILTERS
    out = []
    for code in base:
        if code not in info.index:
            continue
        if not (code.endswith('.XSHG') or code.endswith('.XSHE')):
            continue
        if apply_f:
            if info.loc[code, 'start_date'] > cutoff:
                continue
            c = cd[code]
            if c.is_st or c.paused:
                continue
            name = c.name if hasattr(c, 'name') else ''
            if 'ST' in name or '退' in name:
                continue
        out.append(code)
    return out


def _breadth_universe(context, cd):
    stocks = get_all_securities(['stock'], date=context.previous_date)
    stocks = stocks[(stocks.index.str.endswith('.XSHG') | stocks.index.str.endswith('.XSHE'))]
    cutoff = _list_cutoff(context)
    stocks = stocks[stocks['start_date'] <= cutoff]
    return list(stocks.index)


def _list_cutoff(context):
    return get_trade_days(end_date=context.previous_date, count=Cfg.MIN_LIST_DAYS)[0]


def _hist_close(code, end_date, count):
    try:
        df = get_price(code, end_date=end_date, count=count, frequency='daily',
                       fields=['close'], skip_paused=False, fq='pre', panel=False)
        return df['close'].values
    except Exception as e:
        log.warn('取收盘失败 %s: %s' % (code, e))
        return None


def _hist_field(code, start_date, end_date, field, count=None):
    try:
        if count is not None:
            df = get_price(code, end_date=end_date, count=count, frequency='daily',
                           fields=[field], skip_paused=False, fq='pre', panel=False)
        else:
            df = get_price(code, start_date=start_date, end_date=end_date, frequency='daily',
                           fields=[field], skip_paused=False, fq='pre', panel=False)
        return df[field].values
    except Exception:
        return None


def _trading_days_held(buy_date, cur_date):
    """从买入日到当前日历经的交易日数（买入当日为0）。"""
    if not buy_date:
        return 0
    try:
        return len(get_trade_days(start_date=buy_date, end_date=cur_date)) - 1
    except Exception:
        return 0


def _today_high(context, cd, code):
    """今日盘中最高价（14:50 决策时点用分钟线求最高；异常回退到当前价）。"""
    try:
        df = get_price(code, end_date=context.current_dt, count=240,
                       frequency='1m', fields=['high'], panel=False)
        if df is not None and len(df) > 0:
            d = context.current_dt.date()
            todays = [float(hv) for tv, hv in zip(df.index, df['high'].values)
                      if hasattr(tv, 'date') and tv.date() == d]
            if todays:
                return max(todays)
    except Exception:
        pass
    return cd[code].last_price


def _breakout_exit(context, cd, code, h):
    """买入满 BREAKOUT_DAYS 个交易日后，【每个交易日】判断：若当日最高价 未高于 前
    BREAKOUT_LOOKBACK 个交易日最高价（未创近N日新高）-> 清仓。"""
    if _trading_days_held(h.get('buy_date'), context.current_dt.date()) < Cfg.BREAKOUT_DAYS:
        return False
    highs = _hist_field(code, None, context.previous_date, 'high', count=Cfg.BREAKOUT_LOOKBACK)
    if highs is None or len(highs) < Cfg.BREAKOUT_LOOKBACK:
        return False                      # 数据不足，本日不判
    prev_high = float(np.nanmax(highs))
    today_high = _today_high(context, cd, code)
    return today_high is not None and today_high <= prev_high


def _hist_close_panel(codes, end_date, count, batch=800):
    codes = list(codes)
    if not codes:
        return None
    frames = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        try:
            df = get_price(chunk, end_date=end_date, count=count, frequency='daily',
                           fields=['close'], skip_paused=False, fq='pre', panel=False)
            if df is not None and len(df) > 0:
                frames.append(df)
        except Exception as e:
            log.warn('取收盘面板分批失败(%d): %s' % (i, e))
    if not frames:
        return None
    try:
        df = pd.concat(frames, ignore_index=True)
        return df.pivot(index='time', columns='code', values='close').sort_index()
    except Exception as e:
        log.warn('收盘面板拼接失败: %s' % e)
        return None


def _cur_index(code, dt):
    try:
        df = get_price(code, end_date=dt, count=1, frequency='1m', fields=['close'], panel=False)
        return float(df['close'].iloc[-1])
    except Exception:
        h = _hist_close(code, dt, 1)
        return float(h[-1]) if h is not None and len(h) else None


def _round100(x):
    return int(x // 100) * 100


def _pending_has(code):
    for od in g.pending_sells:
        if od['code'] == code:
            return True
    return False
