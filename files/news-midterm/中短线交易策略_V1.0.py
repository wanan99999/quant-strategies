# -*- coding: utf-8 -*-
"""
中短线交易策略 V1.0
========================================================
策略类型：中短线趋势跟踪 + 大盘顶背离风控
平台：聚宽 JoinQuant

对应说明书《中短线交易策略说明书_V1.0》，实现要点：
  1. 买入：分步建仓（半仓试探 -> 全仓确认）
  2. 卖出：分层退出（大盘顶背离减半 -> 继续下跌10%清仓）
  3. 止损：距买入均价 / 持仓最高点回撤 >=20% 清仓（优先级低于卖出）
  4. 仓位：最多2只、单票上限50%、半仓25%、全仓50%、总仓位上限100%
  5. 执行：盘后决策；买入按确认日收盘价，卖出按次日开盘价；清仓/止损后5日冷却

执行时点近似说明（回测口径）：
  - 14:50（收盘前）用当前价近似“当日收盘价”，做出全部信号判断；
    买入在此刻按“收盘价”成交；卖出信号则登记为待执行。
  - 次日 09:31（开盘后）按开盘价执行前一日登记的卖出。
  这样即实现说明书“买入以确认日收盘价、卖出以次日开盘价”的口径。
"""

import numpy as np
import pandas as pd
from jqdata import *


# ==================================================================
# 一、参数配置（集中管理，便于调优）
# ==================================================================
class Cfg:
    BENCHMARK = '000001.XSHG'   # 上证指数

    # ---- 仓位 ----
    MAX_HOLDINGS   = 2          # 最大持仓只数
    HALF_RATIO     = 0.25       # 半仓（占总资金）
    FULL_RATIO     = 0.50       # 全仓（占总资金）

    # ---- 股票池 ----
    # 模式：'market' 全市场沪深A股扫描 ；'custom' 使用自定义股票池（消息股/自选股）
    POOL_MODE      = 'custom'
    # 自定义股票池代码（POOL_MODE='custom' 时生效）。可在此静态维护，
    # 也可在 update_custom_pool() 中动态刷新（如每日读取消息股清单）。
    CUSTOM_POOL    = [
        # '600519.XSHG',
        # '000858.XSHE',
    ]
    # 自定义池是否仍套用 ST / 上市满60日 / 换手率 过滤（关掉则完全信任传入清单）
    CUSTOM_POOL_APPLY_FILTERS = True

    # ---- 选股过滤 ----
    # 换手率过滤：按用户要求默认关闭。开启后若首次入场日换手率不足，会错过“站上5日线”
    # 的半仓建仓点，导致后续无法通过金叉加满仓。说明书原为“换手率>=3%”，如需恢复置 True。
    USE_TURNOVER_FILTER = False
    MIN_TURNOVER   = 3.0        # 换手率下限（%），仅 USE_TURNOVER_FILTER=True 时生效
    MIN_LIST_DAYS  = 60         # 上市满 N 个交易日

    # ---- 均线 ----
    MA_SHORT       = 5
    MA_LONG        = 10
    GOLDEN_CONFIRM = 2          # 金叉连续确认交易日数

    # ---- 顶背离（卖出一级）四条件 ----
    NEW_HIGH_DAYS  = 10         # 创 N 日收盘新高
    BREADTH_MA     = 5          # 弱势广度用的均线
    WEAK_PCT       = 0.50       # 收盘价<5日线 占比阈值
    CUM_DOWN_DAYS  = 3          # 累计下跌统计天数
    CUM_DOWN_PCT   = 0.50       # 累计下跌 占比阈值

    # ---- 卖出二级 / 止损 ----
    SELL2_DROP     = 0.10       # 减半后继续下跌阈值（清仓）
    STOP_LOSS      = 0.15       # 回撤止损阈值：从 max(买入均价,持仓最高价) 回撤达此比例清仓
                                # （最高价≥均价，实际即"涨到最高点后回撤15%"）

    # ---- 冷却 ----
    COOLDOWN_DAYS  = 5          # 清仓/止损后 N 个交易日不买回

    # ---- 大盘预警（顶背离触发后禁买窗口）----
    # 顶背离=大盘预警：触发后含当日共 N 个交易日禁止一切买入（新开/加仓），只允许卖出。
    WARN_BARS      = 10

    # ---- 买入后动能确认（每日判断：未创近N日新高则清仓）----
    BREAKOUT_DAYS     = 3       # 买入满 N 个交易日后开始【每日】检查（设小=更早开始）
    BREAKOUT_LOOKBACK = 3       # 每日对比"前 N 个交易日最高价"

    # ---- 附加安全项（说明书未强制，可关）----
    # 顶背离触发当日是否暂停新开/加仓（默认关闭以严格贴合说明书）
    PAUSE_BUY_ON_DIVERGENCE = False
    # ---- 调试开关（生产默认关闭，排查时按需打开）----
    # 每日打印持仓诊断（现价/均价/最高价/止损价/距止损）
    DEBUG_HOLDINGS = False
    # 打印顶背离四条件明细，便于核对某日是否/为何触发
    DEBUG_DIVERGENCE = False
    # 观察名单：逐日打印这些票的跟踪状态与买入链路判定（填代码即开启）
    DEBUG_WATCH = []


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

    # ---- 持仓状态：code -> dict ----
    # 单只股票只有三态：'full'(满仓) / 'half'(半仓) / 清仓(从字典移除)
    # half_source: 'probe'(试探半仓，可加仓至满) / 'divergence'(满仓顶背离减仓而来)
    # {stage:'half'/'full', half_source, half_price, full_price,
    #  buy_date, highest, reduce_basis}
    g.holdings = {}
    # 冷却：code -> 剩余交易日
    g.cooldown = {}
    # 待执行卖出（次日开盘）：[{code, action:'reduce'/'clear', target_amount, basis}]
    g.pending_sells = []
    # 当日顶背离结果缓存
    g.divergence_today = False
    # 大盘预警剩余禁买交易日（>0 时禁止一切买入，只允许卖出）
    g.warn_left = 0
    # 自定义股票池（运行期可动态刷新）
    g.custom_pool = list(Cfg.CUSTOM_POOL)

    run_daily(before_market_open, time='before_open')
    run_daily(market_open,        time='09:31')
    run_daily(before_close,       time='14:50')

    log.info('中短线策略 V1.0 初始化完成 [BUILD=release+warn] 股票池模式=%s' % Cfg.POOL_MODE)
    if Cfg.POOL_MODE == 'custom' and not g.custom_pool:
        log.warn('自定义股票池为空：请在 Cfg.CUSTOM_POOL 或 update_custom_pool() 中填入代码，'
                 '否则不会产生任何买入。')


# ==================================================================
# 三、盘前：冷却递减、清理
# ==================================================================
def before_market_open(context):
    # 冷却期递减
    for code in list(g.cooldown.keys()):
        g.cooldown[code] -= 1
        if g.cooldown[code] <= 0:
            del g.cooldown[code]
    # 大盘预警禁买窗口递减
    if g.warn_left > 0:
        g.warn_left -= 1
    g.divergence_today = False
    # 每日刷新自定义股票池（消息股清单可能每日变化）
    update_custom_pool(context)


# ==================================================================
# 四、开盘：执行昨日登记的卖出（按开盘价）
# ==================================================================
def market_open(context):
    if not g.pending_sells:
        return
    cd = get_current_data()
    retry = []   # 未成交（跌停/停牌）的委托，保留到下一交易日重试
    for od in g.pending_sells:
        code, action = od['code'], od['action']
        pos = context.portfolio.positions.get(code)
        # 持仓已不存在：视为已了结，清理跟踪
        if pos is None or pos.total_amount <= 0:
            if action == 'clear':
                g.holdings.pop(code, None)
                g.cooldown[code] = Cfg.COOLDOWN_DAYS
            else:
                g.holdings.pop(code, None)
            continue

        before = pos.total_amount
        target = od['target_amount']
        order_target(code, target)

        # 依据实际持仓变化判断成交结果（跌停/停牌会成交 0 股）
        pos2 = context.portfolio.positions.get(code)
        remaining = pos2.total_amount if pos2 is not None else 0

        if action == 'reduce':
            if remaining < before:          # 有减仓成交（部分或全部）
                if code in g.holdings:
                    g.holdings[code]['stage'] = 'half'
                    g.holdings[code]['half_source'] = 'divergence'
                    g.holdings[code]['reduce_basis'] = cd[code].day_open
                log.info('[减半->半仓] %s %d->%d 股，减仓价 %.3f' %
                         (code, before, remaining, cd[code].day_open))
            else:                           # 未成交，下一交易日重挂
                retry.append(od)
                log.warn('[减半未成交] %s（疑跌停/停牌），下一交易日重试' % code)
        else:  # clear
            if remaining <= 0:              # 已清仓
                g.holdings.pop(code, None)
                g.cooldown[code] = Cfg.COOLDOWN_DAYS
                log.info('[清仓] %s（%s）' % (code, od.get('reason', '')))
            else:                           # 未卖光，保留跟踪 + 下一交易日重挂
                retry.append(od)
                log.warn('[清仓未成交] %s 仍持 %d 股（疑跌停/停牌），下一交易日重试'
                         % (code, remaining))
    g.pending_sells = retry


# ==================================================================
# 五、收盘前：盘后决策主流程
# ==================================================================
def before_close(context):
    cd = get_current_data()
    prev = context.previous_date
    total_value = context.portfolio.total_value

    # 1) 更新持仓最高价 / 均价
    _update_holdings(context, cd)
    # 2) 持仓校准：任何未被跟踪的实际持仓都纳入，确保止损必覆盖
    _reconcile(context, cd)
    # 观察名单买入链路诊断
    _debug_watch(context, cd)

    # 3) 上证指数序列（含“今日”当前价）
    idx_close_hist = _hist_close(Cfg.BENCHMARK, prev, 12)      # 至昨日的收盘序列
    idx_today = _cur_index(Cfg.BENCHMARK, context.current_dt)  # 今日当前价≈收盘
    idx_ok = (idx_close_hist is not None and len(idx_close_hist) >= 11
              and idx_today is not None)

    # 4) 大盘顶背离（=大盘预警）——空仓也计算，以便随时检测预警。
    #    c1/c2（上涨+创10日新高）是廉价预过滤，多数日子短路，不增负担。
    divergence = False
    if idx_ok:
        try:
            divergence = _check_divergence(context, cd, idx_close_hist, idx_today)
        except Exception as e:
            divergence = False
            log.warn('顶背离计算异常，本日按无顶背离处理（不影响止损）：%s' % e)
    g.divergence_today = divergence
    if divergence:
        g.warn_left = Cfg.WARN_BARS           # 大盘预警：刷新禁买窗口
        log.info('[大盘预警] 顶背离触发：含当日共 %d 个交易日禁止买入，仅允许卖出' % Cfg.WARN_BARS)

    # 5) 卖出判断（登记，次日开盘执行）；止损/继续跌10%不依赖大盘数据，恒执行。
    _evaluate_sells(context, cd, divergence)

    if not idx_ok:
        log.warn('上证数据不足：本日跳过买入，仅执行止损/二级卖出')
        return

    # 6) 大盘预警窗口内：禁止一切买入（新开/加仓），只允许卖出。
    if g.warn_left > 0:
        return

    # 7) 买入判断（当日收盘价执行）
    idx_ma5 = (idx_close_hist[-4:].sum() + idx_today) / 5.0
    try:
        _evaluate_buys(context, cd, idx_today, idx_ma5, total_value)
    except Exception as e:
        import traceback
        log.error('[BUY异常] _evaluate_buys 抛错: %s\n%s' % (e, traceback.format_exc()))


# ==================================================================
# 六、卖出评估
# ==================================================================
def _evaluate_sells(context, cd, divergence):
    # divergence 由 before_close 统一计算并传入（空仓也算，支撑大盘预警检测）。
    # 每只持仓单独用 try 包裹：单只异常不得连累其他持仓的止损/卖出判断。
    for code in list(g.holdings.keys()):
        try:
            _eval_one_sell(context, cd, code, divergence)
        except Exception as e:
            log.warn('持仓 %s 卖出评估异常（跳过本只，不影响其他）：%s' % (code, e))


def _eval_one_sell(context, cd, code, divergence):
    pos = context.portfolio.positions.get(code)
    if pos is None or pos.total_amount <= 0:
        g.holdings.pop(code, None)
        return
    # 避免同日重复登记
    if _pending_has(code):
        return

    h = g.holdings[code]
    cur = cd[code].last_price
    avg = _avg_cost(h)
    stop_base = max(avg, h['highest'])       # 止损基准取较高者
    stop_price = stop_base * (1 - Cfg.STOP_LOSS)

    if Cfg.DEBUG_HOLDINGS:
        dd = (cur / stop_base - 1) * 100 if stop_base else 0.0
        log.info('[持仓诊断] %s 状态=%s/%s 现价=%.2f 均价=%.2f 最高=%.2f '
                 '止损价=%.2f 距止损=%.1f%%'
                 % (code, h['stage'], h.get('half_source'), cur, avg,
                    h['highest'], stop_price, dd))

    # —— 新增卖出条款（最高优先级）：买入后满3个交易日未创近3日新高 -> 清仓 ——
    if _breakout_exit(context, cd, code, h):
        _queue_clear(code, pos, '3日未创近3日新高清仓')
        return

    # —— 优先级：卖出策略 > 止损策略 ——
    if h['stage'] == 'full':
        # 卖出一级：满仓遇顶背离 -> 减半（次日开盘价成交）
        if divergence:
            _queue_reduce(code, pos)
        elif cur <= stop_price:
            _queue_clear(code, pos, '止损')
    else:  # 'half'
        if h.get('half_source') == 'divergence':
            # 卖出二级：减仓半仓后继续下跌10%（基准=减仓卖出价）-> 清仓
            if h.get('reduce_basis') and cur <= h['reduce_basis'] * (1 - Cfg.SELL2_DROP):
                _queue_clear(code, pos, '继续下跌10%清仓')
            elif cur <= stop_price:
                _queue_clear(code, pos, '止损')
        else:
            # 试探半仓遇顶背离：直接清仓（不减半，三态约束）；否则止损兜底
            if divergence:
                _queue_clear(code, pos, '顶背离清仓(试探半仓)')
            elif cur <= stop_price:
                _queue_clear(code, pos, '止损')


def _queue_reduce(code, pos):
    target = _round100(pos.total_amount * 0.5)
    g.pending_sells.append({'code': code, 'action': 'reduce',
                            'target_amount': target})
    log.info('[登记减半] %s 目标 %d 股' % (code, target))


def _queue_clear(code, pos, reason):
    g.pending_sells.append({'code': code, 'action': 'clear',
                            'target_amount': 0, 'reason': reason})
    log.info('[登记清仓] %s（%s）' % (code, reason))


# ==================================================================
# 七、买入评估
# ==================================================================
def _evaluate_buys(context, cd, idx_today, idx_ma5, total_value):
    positions = context.portfolio.positions

    # (a) 已持半仓 -> 全仓确认加仓（不含已登记卖出/顶背离减半的）
    for code in list(g.holdings.keys()):
        h = g.holdings[code]
        # 仅“试探半仓”可加仓至满；减仓而来的半仓(divergence)不再加仓
        if h['stage'] != 'half' or h.get('half_source') != 'probe':
            continue
        if _pending_has(code):
            continue
        ok, _ = _golden_confirm(code, context.previous_date, cd)
        if ok:
            order_target_value(code, Cfg.FULL_RATIO * total_value)
            h['stage'] = 'full'
            h['full_price'] = cd[code].last_price
            log.info('[全仓确认] %s 加至50%%，加仓价 %.3f' % (code, cd[code].last_price))

    # (b) 剩余空位 -> 半仓试探新开
    slots = Cfg.MAX_HOLDINGS - len(g.holdings)
    if slots <= 0:
        return
    # 半仓大前提：上证站上5日线
    if not (idx_today > idx_ma5):
        return

    candidates = _scan_half_candidates(context, cd)
    for code in candidates[:slots]:
        order_target_value(code, Cfg.HALF_RATIO * total_value)
        p = cd[code].last_price
        g.holdings[code] = {
            'stage': 'half', 'half_source': 'probe',
            'half_price': p, 'full_price': None,
            'buy_date': context.current_dt.date(), 'highest': p,
            'reduce_basis': None,
        }
        log.info('[半仓试探] %s 买入25%%，价 %.3f' % (code, p))


def _scan_half_candidates(context, cd):
    """扫描全市场，返回满足半仓条件的候选，按 金叉强度降序 排序。"""
    prev = context.previous_date
    base = _tradable_universe(context, cd)                 # 过滤 ST/上市/换手率
    # 排除已持仓、冷却、已登记
    pend = {od['code'] for od in g.pending_sells}
    base = [c for c in base if c not in g.holdings
            and c not in g.cooldown and c not in pend]
    if not base:
        return []

    piv = _hist_close_panel(base, prev, 10)                # 至昨日收盘
    if piv is None or piv.empty:
        return []
    cols = list(piv.columns)
    today = pd.Series({c: cd[c].last_price for c in cols})

    ma5 = (piv.tail(4).sum() + today) / 5.0
    ma10 = (piv.tail(9).sum() + today) / 10.0

    # 半仓条件：个股当日收盘价 > 5日均线
    mask = today > ma5
    picks = [c for c in cols if bool(mask.get(c, False))
             and not np.isnan(ma10.get(c, np.nan))]
    if not picks:
        return []

    # 金叉强度（幅度）：(MA5-MA10)/MA10，越大越强；仅按此降序排序
    strength = ((ma5 - ma10) / ma10)
    picks.sort(key=lambda c: float(strength.get(c, 0.0)), reverse=True)
    return picks


# ==================================================================
# 八、顶背离四条件
# ==================================================================
def _check_divergence(context, cd, idx_hist, idx_today):
    """四条件全部满足返回 True。"""
    # 条件一：上证当日上涨
    c1 = idx_today > idx_hist[-1]
    # 条件二：创10日收盘新高（严格高于前9日最高收盘）
    c2 = idx_today > idx_hist[-(Cfg.NEW_HIGH_DAYS - 1):].max()
    if not (c1 and c2):
        if Cfg.DEBUG_DIVERGENCE:
            log.info('[顶背离] 前置未过 c1(上涨)=%s c2(创10日新高)=%s '
                     '(今=%.2f 昨=%.2f 前9高=%.2f)'
                     % (c1, c2, idx_today, idx_hist[-1],
                        idx_hist[-(Cfg.NEW_HIGH_DAYS - 1):].max()))
        return False

    # 条件三、四需全市场广度，较重，放在前两条通过后再算
    uni = _breadth_universe(context, cd)
    if not uni:
        log.warn('[顶背离] 广度universe为空，无法判定条件三/四')
        return False
    piv = _hist_close_panel(uni, context.previous_date, Cfg.BREADTH_MA)  # 至昨日收盘
    if piv is None or piv.empty or len(piv) < Cfg.CUM_DOWN_DAYS:
        log.warn('[顶背离] 广度面板数据不足，无法判定条件三/四')
        return False
    cols = list(piv.columns)
    today = pd.Series({c: cd[c].last_price for c in cols})

    # 条件三：收盘价 < 5日均线 的占比 > 50%
    # 用 min_count 严格口径：完整数据不足 BREADTH_MA-1 天的股票记为 NaN 并剔除，避免误算。
    recent = piv.tail(Cfg.BREADTH_MA - 1)
    ma5 = (recent.sum(min_count=Cfg.BREADTH_MA - 1) + today) / float(Cfg.BREADTH_MA)
    valid5 = ma5.notna() & today.notna()
    if valid5.sum() == 0:
        return False
    weak = ((today < ma5) & valid5).sum() / float(valid5.sum())
    c3 = weak > Cfg.WEAK_PCT

    # 条件四：近3日累计下跌 占比 > 50%
    # (今日收盘 - 3日前收盘) / 3日前收盘 < 0
    close_3ago = piv.iloc[-Cfg.CUM_DOWN_DAYS]
    validc = close_3ago.notna() & today.notna() & (close_3ago != 0)
    if validc.sum() == 0:
        return False
    cum = (today - close_3ago) / close_3ago
    down = ((cum < 0) & validc).sum() / float(validc.sum())
    c4 = down > Cfg.CUM_DOWN_PCT

    if Cfg.DEBUG_DIVERGENCE:
        log.info('[顶背离] c1=%s c2=%s c3(弱势%%)=%.1f%%>50=%s '
                 'c4(下跌%%)=%.1f%%>50=%s 样本=%d'
                 % (c1, c2, weak * 100, c3, down * 100, c4, int(valid5.sum())))

    result = c1 and c2 and c3 and c4
    if result:
        log.info('★ 顶背离触发：涨=%s 新高=%s 弱势广度=%.1f%% 累计下跌广度=%.1f%%'
                 % (c1, c2, weak * 100, down * 100))
    return result


# ==================================================================
# 九、辅助函数
# ==================================================================
def _golden_confirm(code, prev, cd):
    """全仓条件：MA5>MA10 且 连续2个交易日维持（今日与昨日均金叉）。"""
    piv = _hist_close_panel([code], prev, 11)
    if piv is None or code not in piv.columns:
        return False, 0.0
    s = piv[code].dropna()
    if len(s) < 10:
        return False, 0.0
    today = cd[code].last_price
    ma5_t = (s.tail(4).sum() + today) / 5.0
    ma10_t = (s.tail(9).sum() + today) / 10.0
    ma5_y = s.tail(5).mean()
    ma10_y = s.tail(10).mean()
    ok = (ma5_t > ma10_t) and (ma5_y > ma10_y)
    return ok, (ma5_t - ma10_t) / ma10_t if ma10_t else 0.0


def _reconcile(context, cd):
    """持仓校准：把账户里存在、但 g.holdings 未跟踪的持仓重新纳入跟踪，
    避免任何原因（委托未成交被误删、异常等）产生“孤儿持仓”而逃过止损。
    保守地按满仓 + 成本价纳入，后续止损/顶背离/二级卖出即可正常作用。"""
    for code, pos in context.portfolio.positions.items():
        if pos.total_amount <= 0:
            continue
        if code in g.holdings:
            continue
        if _pending_has(code):
            continue
        ac = pos.avg_cost if getattr(pos, 'avg_cost', 0) else cd[code].last_price
        g.holdings[code] = {
            'stage': 'full', 'half_source': None,
            'half_price': ac, 'full_price': ac,
            'buy_date': context.previous_date,
            'highest': max(ac, cd[code].last_price),
            'reduce_basis': None,
        }
        log.warn('[持仓校准] 未跟踪持仓 %s 已纳入(按满仓/成本价 %.3f)' % (code, ac))


def _debug_watch(context, cd):
    """逐日打印观察名单每只票的买入全链路 + 跟踪状态，定位为何未买入/未加仓/未减仓。"""
    if not Cfg.DEBUG_WATCH:
        return
    prev = context.previous_date
    for code in Cfg.DEBUG_WATCH:
        # 跟踪/持仓状态（最关键：g.holdings 里是什么、实际持仓多少、是否在待卖队列）
        h = g.holdings.get(code)
        stage = ('%s/%s' % (h['stage'], h.get('half_source'))) if h else '未跟踪'
        pos = context.portfolio.positions.get(code)
        amt = pos.total_amount if pos is not None else 0
        pend = [od['action'] for od in g.pending_sells if od['code'] == code]
        # 金叉 / 站上MA5
        today = ma5 = ma10 = ma5_y = ma10_y = float('nan')
        try:
            piv = _hist_close_panel([code], prev, 11)
            today = cd[code].last_price
            if piv is not None and code in piv.columns:
                s = piv[code].dropna()
                if len(s) >= 4:
                    ma5 = (s.tail(4).sum() + today) / 5.0
                if len(s) >= 9:
                    ma10 = (s.tail(9).sum() + today) / 10.0
                if len(s) >= 5:
                    ma5_y = s.tail(5).mean()
                if len(s) >= 10:
                    ma10_y = s.tail(10).mean()
        except Exception as e:
            log.warn('[WATCH] %s MA计算异常: %s' % (code, e))
        golden = (ma5 > ma10) and (ma5_y > ma10_y)
        log.info('[WATCH] %s 跟踪=%s 实际持股=%s 待卖=%s 冷却=%s 现价=%.2f '
                 'MA5=%.2f 站上MA5=%s MA10=%.2f 金叉2日=%s'
                 % (code, stage, amt, pend or '无', g.cooldown.get(code, 0),
                    today, ma5, today > ma5, ma10, golden))


def _update_holdings(context, cd):
    """更新持仓最高价（含日内高点）与状态同步。"""
    for code in list(g.holdings.keys()):
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            # 已被外部清空
            g.holdings.pop(code, None)
            continue
        h = g.holdings[code]
        # 用买入以来的日内最高价 + 当前价
        highs = _hist_field(code, h['buy_date'], context.previous_date, 'high')
        hi = h['highest']
        if highs is not None and len(highs) > 0:
            hi = max(hi, float(np.nanmax(highs)))
        hi = max(hi, cd[code].last_price)
        h['highest'] = hi


def _avg_cost(h):
    """买入均价：曾满仓(有加仓价)=(半仓价+加仓价)/2；纯试探半仓=半仓价。
    注意：减仓而来的 divergence 半仓虽为 half，但已建过满仓，仍按满仓均价。"""
    if h.get('full_price') is not None:
        return (h['half_price'] + h['full_price']) / 2.0
    return h['half_price']


def update_custom_pool(context):
    """每日刷新自定义股票池的钩子（默认沿用静态 CUSTOM_POOL）。

    若“消息股”清单每日变化，可在此重写 g.custom_pool，例如：
      - 从聚宽研究环境的数据文件读取当日清单；
      - 按板块/概念动态生成 get_concept_stocks / get_industry_stocks；
      - 依据自有事件信号筛选。
    只需保证赋值 g.custom_pool = [代码列表] 即可，其余流程自动套用过滤与排序。
    """
    # 示例（默认不动态刷新）：
    # g.custom_pool = ['600519.XSHG', '000858.XSHE']
    pass


def load_stock_pool(context):
    """返回候选股票池的“基础代码列表”（过滤前）。
    - POOL_MODE='custom'：使用 g.custom_pool（消息股/自选股）
    - POOL_MODE='market'：全市场沪深A股
    """
    if Cfg.POOL_MODE == 'custom':
        return list(g.custom_pool)
    stocks = get_all_securities(['stock'], date=context.previous_date)
    stocks = stocks[(stocks.index.str.endswith('.XSHG') |
                     stocks.index.str.endswith('.XSHE'))]
    return list(stocks.index)


def _tradable_universe(context, cd):
    """可交易股票池：在“股票池基础清单”上套用 非ST / 上市满60日 / 换手率>=3%。"""
    base = load_stock_pool(context)
    if not base:
        return []
    info = get_all_securities(['stock'], date=context.previous_date)
    cutoff = _list_cutoff(context)
    # 自定义池可选择是否套用过滤；全市场模式恒套用
    apply_filters = (Cfg.POOL_MODE == 'market') or Cfg.CUSTOM_POOL_APPLY_FILTERS

    codes = []
    for code in base:
        if code not in info.index:                       # 非A股/已退市
            continue
        if not (code.endswith('.XSHG') or code.endswith('.XSHE')):
            continue
        if apply_filters:
            if info.loc[code, 'start_date'] > cutoff:    # 上市未满60日
                continue
            c = cd[code]
            if c.is_st or c.paused:
                continue
            name = c.name if hasattr(c, 'name') else ''
            if 'ST' in name or '退' in name:
                continue
        codes.append(code)

    if apply_filters and Cfg.USE_TURNOVER_FILTER:
        tmap = _turnover_map(codes, context.previous_date)   # 前一交易日换手率，仅筛选
        codes = [c for c in codes if tmap.get(c, 0.0) >= Cfg.MIN_TURNOVER]
    return codes


def _breadth_universe(context, cd):
    """顶背离广度用全市场：沪深A股，上市满60日（不含北交所、不含新股）。"""
    stocks = get_all_securities(['stock'], date=context.previous_date)
    stocks = stocks[(stocks.index.str.endswith('.XSHG') |
                     stocks.index.str.endswith('.XSHE'))]
    cutoff = _list_cutoff(context)
    stocks = stocks[stocks['start_date'] <= cutoff]
    # 停牌股当前价取昨收，广度计算保留（不影响“收盘价<5日线”口径）
    return list(stocks.index)


def _list_cutoff(context):
    days = get_trade_days(end_date=context.previous_date, count=Cfg.MIN_LIST_DAYS)
    return days[0]


def _turnover_map(codes, date):
    """换手率(%) 映射，取最近一个交易日估值。"""
    if not codes:
        return {}
    out = {}
    try:
        df = get_valuation(codes, end_date=date, count=1, fields=['turnover_ratio'])
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                out[row['code']] = float(row['turnover_ratio'])
    except Exception as e:
        log.warn('换手率获取失败: %s' % e)
    return out


# ---------- 价格序列工具 ----------
def _hist_close(code, end_date, count):
    """单标的：至 end_date 的收盘序列（numpy）。"""
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
            df = get_price(code, start_date=start_date, end_date=end_date,
                           frequency='daily', fields=[field],
                           skip_paused=False, fq='pre', panel=False)
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
    """多标的收盘面板：index=日期, columns=code。分批请求，避免全市场一次性
    get_price 过大导致异常/超时；单批失败只跳过该批，不影响整体。"""
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
            log.warn('取收盘面板分批失败(%d-%d): %s' % (i, i + len(chunk), e))
    if not frames:
        return None
    try:
        df = pd.concat(frames, ignore_index=True)
        piv = df.pivot(index='time', columns='code', values='close')
        return piv.sort_index()
    except Exception as e:
        log.warn('收盘面板拼接失败: %s' % e)
        return None


def _cur_index(code, dt):
    """指数当前价（用1分钟线近似当前/收盘）。"""
    try:
        df = get_price(code, end_date=dt, count=1, frequency='1m',
                       fields=['close'], panel=False)
        return float(df['close'].iloc[-1])
    except Exception:
        h = _hist_close(code, dt, 1)
        return float(h[-1]) if h is not None and len(h) else None


def _round100(x):
    return int(x // 100) * 100


def _pending_has(code):
    """该标的是否已在待卖队列中。显式循环，杜绝裸生成器恒真的坑。"""
    for od in g.pending_sells:
        if od['code'] == code:
            return True
    return False
