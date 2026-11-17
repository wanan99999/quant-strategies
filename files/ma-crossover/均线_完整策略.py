# -*- coding: utf-8 -*-
"""
双均线/三均线交叉策略（MA Crossover）—— 选股+交易+风控【完整单文件版】(聚宽 JoinQuant)
========================================================================
本文件把三部分合并为一个可直接回测/实盘的策略文件（主入口 initialize 在文末）。
内容与分模块版一致，函数名无冲突，直接整份粘贴到聚宽策略即可运行。

结构：
  【第一部分 选股】行业类型映射 + 公共指标(MA/斜率/ADX/布林/缠绕) + select_universe
  【第三部分 风控】账户状态/大盘环境/震荡过滤/集中度/单笔止损
  【第二部分 交易】initialize + 开盘执行 + 收盘评估(出场/减仓/入场)

规则要点：
  选股: 非ST/上市≥120日/成交额≥1亿/价≥3元 → 均线多头排列(双/三) →
        行业适配(成长→三均线、周期/主题→双均线、防御→排除) → 四维评分取前10
  交易: 双均线金叉入场、三均线三线多头+新鲜金叉入场；死叉/破位清仓、
        排列破坏减半、乖离>8%或均线粘合减半；单票(双18%/三22%) 只数(双6/三4)
  风控: 硬止损(-8%/-6%)、时间止损(15/10日)、均线止损；ADX<20/缠绕≥4/布林收缩过滤；
        连亏3笔停3日、月回撤>8%停开、最大回撤>15%清仓；大盘(上证vs MA60/MA20)分档；
        同行业≤2只且≤30%、同类型≤50%
  成交: 每日收盘评估 → 出场/减仓当日收盘成交；入场次日开盘成交；跌停改次日开盘清仓
★ 回测建议【分钟】频率，'open'/'close' 定时点精准撮合（日线频率亦可用）。
"""
import numpy as np
from jqdata import *


# ==========================================================================
# 第一部分：选股策略
# ==========================================================================

# ---- 申万一级行业 → 类型映射 ----
GROWTH = {'电力设备', '电子', '计算机', '医药生物'}
CYCLE  = {'机械设备', '汽车', '基础化工', '有色金属', '房地产',
          '钢铁', '建筑材料', '家用电器', '轻工制造'}
THEME  = {'通信', '传媒', '国防军工', '社会服务'}
DEFENSE = {'银行', '公用事业', '交通运输', '煤炭'}   # 防御型 → 直接排除


def sector_type_of(name):
    """返回 'growth'/'cycle'/'theme'/'defense'；未知行业默认按 'cycle'(双均线)。"""
    if name in GROWTH:
        return 'growth'
    if name in DEFENSE:
        return 'defense'
    if name in THEME:
        return 'theme'
    return 'cycle'


def adapt_of(sector_type):
    """行业类型 → 适配子策略。'defense' 返回 None 表示排除。"""
    if sector_type == 'growth':
        return 'triple'
    if sector_type == 'defense':
        return None
    return 'dual'   # cycle / theme


# ---- 公共指标函数 ----
def ma(cl, n, shift=0):
    """简单移动平均：shift=0 为最新，shift=k 为 k 日前的 MA。"""
    end = len(cl) - shift
    if end < n:
        return None
    return float(np.mean(cl[end - n:end]))


def slope_up(cl, n, k):
    """MA(n) 近 k 日斜率 > 0。"""
    a, b = ma(cl, n, 0), ma(cl, n, k)
    return a is not None and b is not None and a > b


def golden_cross(cl, nf, ns, shift=0):
    """shift 日处是否发生金叉：快线由下向上穿慢线。"""
    f1, s1 = ma(cl, nf, shift + 1), ma(cl, ns, shift + 1)
    f0, s0 = ma(cl, nf, shift), ma(cl, ns, shift)
    if None in (f1, s1, f0, s0):
        return False
    return f1 <= s1 and f0 > s0


def death_cross(cl, nf, ns, shift=0):
    """shift 日处是否发生死叉。"""
    f1, s1 = ma(cl, nf, shift + 1), ma(cl, ns, shift + 1)
    f0, s0 = ma(cl, nf, shift), ma(cl, ns, shift)
    if None in (f1, s1, f0, s0):
        return False
    return f1 >= s1 and f0 < s0


def fresh_golden(cl, nf, ns, within=3):
    """最近 within 个交易日内是否出现过金叉（新鲜金叉）。"""
    return any(golden_cross(cl, nf, ns, shift=i) for i in range(within))


def consecutive_above(cl, nf, ns, cap=20):
    """MA(nf) > MA(ns) 已连续多少个交易日（上限 cap）。"""
    cnt = 0
    for i in range(cap):
        a, b = ma(cl, nf, i), ma(cl, ns, i)
        if a is None or b is None or a <= b:
            break
        cnt += 1
    return cnt


def cross_count(cl, nf, ns, window=20):
    """近 window 日 MA(nf) 与 MA(ns) 的交叉次数（衡量均线缠绕）。"""
    signs = []
    for i in range(window + 1):
        a, b = ma(cl, nf, i), ma(cl, ns, i)
        if a is None or b is None:
            break
        signs.append(1 if a > b else -1)
    return sum(1 for i in range(len(signs) - 1) if signs[i] != signs[i + 1])


def true_range(high, low, prev_close):
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def calc_adx(highs, lows, closes, period=14):
    """Wilder ADX(14)。数据不足返回 None。"""
    n = len(closes)
    if n < 2 * period + 2:
        return None
    pdm, mdm, tr = [], [], []
    for i in range(1, n):
        up, dn = highs[i] - highs[i - 1], lows[i - 1] - lows[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(true_range(highs[i], lows[i], closes[i - 1]))

    def wilder(x):
        s = sum(x[:period]); out = [s]
        for i in range(period, len(x)):
            s = s - s / period + x[i]; out.append(s)
        return out

    atr, sp, sm = wilder(tr), wilder(pdm), wilder(mdm)
    dx = []
    for i in range(len(atr)):
        if atr[i] <= 0:
            dx.append(0.0); continue
        dip, dim = 100 * sp[i] / atr[i], 100 * sm[i] / atr[i]
        denom = (dip + dim) or 1.0
        dx.append(100 * abs(dip - dim) / denom)
    if len(dx) < period:
        return None
    adx = float(np.mean(dx[:period]))
    for i in range(period, len(dx)):
        adx = (adx * (period - 1) + dx[i]) / period
    return adx


def boll_bandwidth(cl, n=20, k=2):
    """布林带宽 / MA20 =(上轨-下轨)/中轨 = 2k*std/MA。"""
    if len(cl) < n:
        return None
    mid = float(np.mean(cl[-n:]))
    std = float(np.std(cl[-n:]))
    if mid <= 0:
        return None
    return (2 * k * std) / mid


def get_daily_arrays(security, end_dt, count, fields=('high', 'low', 'close')):
    bars = get_bars(security, count=count, unit='1d', fields=list(fields),
                    include_now=True, end_dt=end_dt, df=False)
    if bars is None or len(bars['close']) < 2:
        return None
    return {f: np.array(bars[f], dtype=float) for f in fields}


# ---- 选股主流程（每周五盘后）----
def select_universe(context):
    """重建候选池 g.pool，并缓存每只标的的均线/行业/适配信息。"""
    p = g.sel
    prev = context.previous_date

    if g.base_index:
        base = get_index_stocks(g.base_index, date=prev)
    else:
        base = list(get_all_securities(types=['stock'], date=prev).index)

    cd = get_current_data()
    from datetime import timedelta
    list_cutoff = prev - timedelta(days=p['min_list_days'])

    # 第一层：基础过滤
    stage1 = []
    for s in base:
        info = cd[s]
        if info.paused or info.is_st or ('ST' in info.name) or ('退' in info.name):
            continue
        sinfo = get_security_info(s)
        if sinfo is None or sinfo.start_date > list_cutoff:
            continue
        stage1.append(s)

    ind = get_industry(stage1, date=prev)

    raw = []
    for s in stage1:
        arr = get_daily_arrays(s, context.current_dt, 90,
                               fields=('high', 'low', 'close', 'volume', 'money'))
        if arr is None or len(arr['close']) < 65:
            continue
        highs, lows, closes = arr['high'], arr['low'], arr['close']
        vol, money = arr['volume'], arr['money']
        price = closes[-1]

        if price < p['min_price']:
            continue
        if float(np.mean(money[-20:])) < p['min_amount']:
            continue

        # 第三层：行业类型 → 适配子策略（防御型排除）
        name = (ind.get(s, {}) or {}).get('sw_l1', {}).get('industry_name', '')
        stype = sector_type_of(name)
        adapt = adapt_of(stype)
        if adapt is None:
            continue

        m5, m10, m20, m60 = ma(closes, 5), ma(closes, 10), ma(closes, 20), ma(closes, 60)
        if None in (m5, m10, m20, m60):
            continue

        # 第二层（修正）：选股只圈定"中期趋势向上"的慢速universe。
        # ★ 不再在选股里要求"已金叉/完全多头排列"——金叉是单日事件，而选股每周才刷新，
        #   若选股已要求金叉状态，则入场端"当日金叉"永远与之错开(4年零买入的根因)。
        #   快速金叉统一交给每日入场逻辑在该universe上捕捉。
        if not (m20 > m60 and slope_up(closes, 60, 5)):
            continue

        # 第四层原始指标
        div = (m5 - m60) / m60 if m60 > 0 else 0.0
        persist = consecutive_above(closes, 5, 10, cap=20)
        v5, v20 = float(np.mean(vol[-5:])), float(np.mean(vol[-20:]))
        volratio = v5 / v20 if v20 > 0 else 1.0
        hi5 = float(np.max(highs[-5:]))
        pullback = (hi5 - price) / hi5 if hi5 > 0 else 0.0

        info = {
            'adapt': adapt, 'sector': name or 'NA', 'sector_type': stype,
            'ma5': m5, 'ma10': m10, 'ma20': m20, 'ma60': m60,
            'div': div, 'price': price,
        }
        raw.append((s, info, {'div': div, 'persist': persist,
                              'volratio': volratio, 'pullback': pullback}))

    if not raw:
        g.pool, g.pool_info = [], {}
        log.warn('[选股] 无标的通过筛选')
        return

    # 第四层：四维加权评分（组内 min-max 归一化）
    def norm(vals):
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return [(v - lo) / rng for v in vals]

    divs = norm([r[2]['div'] for r in raw])
    pers = norm([r[2]['persist'] for r in raw])
    vols = norm([r[2]['volratio'] for r in raw])
    pbs  = norm([r[2]['pullback'] for r in raw])
    scored = []
    for i, (s, info, _) in enumerate(raw):
        info['score'] = (0.30 * divs[i] + 0.25 * pers[i] +
                         0.25 * vols[i] + 0.20 * (1 - pbs[i]))
        scored.append((s, info))
    scored.sort(key=lambda x: x[1]['score'], reverse=True)

    pool, pool_info, sector_cnt = [], {}, {}
    for s, info in scored:
        sec = info['sector']
        if sector_cnt.get(sec, 0) >= p['per_industry_max']:
            continue
        pool.append(s)
        pool_info[s] = info
        sector_cnt[sec] = sector_cnt.get(sec, 0) + 1
        if len(pool) >= p['pool_max']:
            break

    g.pool, g.pool_info = pool, pool_info
    n_dual = sum(1 for s in pool if pool_info[s]['adapt'] == 'dual')
    log.info('[选股] 候选池%d只（双均线%d / 三均线%d）行业分布%s' %
             (len(pool), n_dual, len(pool) - n_dual, sector_cnt))


# ==========================================================================
# 第三部分：风控策略
# ==========================================================================
def init_risk_state(context):
    tv = context.portfolio.total_value
    g.peak_value = tv
    g.month_key = (context.current_dt.year, context.current_dt.month)
    g.month_start_value = tv
    g.loss_streak = 0
    g.pause_open_until = None


def update_risk_state(context):
    tv = context.portfolio.total_value
    g.peak_value = max(getattr(g, 'peak_value', tv), tv)
    key = (context.current_dt.year, context.current_dt.month)
    if key != getattr(g, 'month_key', None):
        g.month_key = key
        g.month_start_value = tv


def record_trade_result(context, ret):
    """一笔平仓后：更新连亏计数；连亏≥3 → 暂停开仓3个交易日。"""
    if ret < 0:
        g.loss_streak = getattr(g, 'loss_streak', 0) + 1
    else:
        g.loss_streak = 0
    if g.loss_streak >= 3:
        fdays = get_trade_days(start_date=context.current_dt, count=4)
        g.pause_open_until = fdays[-1] if len(fdays) >= 4 else None
        g.loss_streak = 0
        log.info('[风控] 连亏3笔 → 暂停开仓至 %s' % g.pause_open_until)


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
    if slope_up(cl, 60, 5):
        return {'action': 'normal', 'scale_dual': 1.0, 'scale_triple': 1.0}
    return {'action': 'reduce', 'scale_dual': 0.5, 'scale_triple': 0.6}


def stock_choppy_block(context, security, A):
    """A 含 high/low/close。ADX<20 / 缠绕≥4 / 布林收缩 任一命中即过滤。返回(bool, reason)。"""
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


def t_weight(context, security):
    pos = context.portfolio.positions.get(security)
    tv = context.portfolio.total_value
    if pos is None or tv <= 0:
        return 0.0
    return pos.value / tv


def concentration_ok(context, sector, sector_type, add_weight):
    """加入 add_weight 后是否仍满足：同行业≤2只且≤30%、同类型≤50%。"""
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


def position_risk_exit(context, security, t, A):
    """单笔风险出场：硬止损 / 时间止损 / 均线止损。返回('clear', reason) 或 (None, '')。"""
    closes = A['close']
    price = closes[-1]
    adapt = t['adapt']
    stop = g.dual['stop'] if adapt == 'dual' else g.triple['stop']
    time_stop = g.dual['time_stop'] if adapt == 'dual' else g.triple['time_stop']

    if t['entry_price'] > 0 and (price / t['entry_price'] - 1.0) <= stop:
        return 'clear', '硬止损%.0f%%' % (stop * 100)
    held_days = len(get_trade_days(start_date=t['entry_date'], end_date=context.previous_date))
    if held_days >= time_stop and price <= t['entry_price']:
        return 'clear', '时间止损(持%d日未盈利)' % held_days
    if adapt == 'dual':
        m20 = ma(closes, 20)
        if m20 is not None and price < m20:
            return 'clear', '收盘<MA20'
    else:
        m10 = ma(closes, 10)
        if m10 is not None and price < m10:
            return 'clear', '收盘<MA10'
    return None, ''


# ==========================================================================
# 第二部分：交易策略（主入口）
# ==========================================================================
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

    g.sel = {
        'min_list_days': 120, 'min_amount': 1e8, 'min_price': 3.0,
        'per_industry_max': 2, 'pool_min': 5, 'pool_max': 10,
    }
    g.dual = {'fast': 10, 'slow': 20, 'filter': 60,
              'pos': 0.18, 'max_stocks': 6, 'total_cap': 0.80,
              'stop': -0.08, 'time_stop': 15}
    g.triple = {'s': 5, 'm': 10, 'l': 20, 'filter': 60,
                'pos': 0.22, 'max_stocks': 4, 'total_cap': 1.00,
                'stop': -0.06, 'time_stop': 10}
    g.bias_cap = 0.08
    g.tangle_gap = 0.02
    g.max_total_weight = 0.98
    g.risk = {
        'adx_min': 20, 'tangle_cross': 4, 'boll_min': 0.05,
        'month_dd_pause': 0.08, 'max_dd_clear': 0.15,
        'max_per_sector_stocks': 2, 'max_per_sector_weight': 0.30,
        'max_per_type_weight': 0.50,
    }

    g.pool, g.pool_info = [], {}
    g.trades = {}
    g.buy_plan = []
    g.sell_open_plan = []

    init_risk_state(context)

    run_weekly(select_universe, weekday=5, time='after_close',
               reference_security='000300.XSHG')
    run_daily(ma_open_exec, time='open', reference_security='000300.XSHG')
    run_daily(ma_close_eval, time='close', reference_security='000300.XSHG')


# ---- 开盘执行：强制清仓 + 计划买入 ----
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


# ---- 收盘评估：出场 / 减仓 / 入场 ----
def ma_close_eval(context):
    update_risk_state(context)
    g.buy_plan = []

    if not g.pool:
        select_universe(context)

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

    # A. 持仓出场 / 减仓（当日收盘成交）
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

    # B. 入场
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


# ---- 出场/减仓判定：风控止损 + 策略出场 + 仓位调整 ----
def _eval_exit(context, s, t, A):
    action, reason = position_risk_exit(context, s, t, A)
    if action:
        return action, reason

    cl = A['close']; price = cl[-1]
    m5, m10, m20, m60 = ma(cl, 5), ma(cl, 10), ma(cl, 20), ma(cl, 60)
    if None in (m5, m10, m20, m60):
        return None, ''
    adapt = t['adapt']

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

    bias = (m5 - m20) / m20 if m20 > 0 else 0.0
    if abs(bias) > g.bias_cap:
        return 'half', 'MA5/MA20乖离%.1f%%>8%%' % (bias * 100)
    mmax, mmin = max(m5, m10, m20), min(m5, m10, m20)
    if mmin > 0 and (mmax - mmin) / mmin < g.tangle_gap:
        return 'half', '三均线粘合(<2%)方向不明'
    if adapt == 'triple' and m5 < m10:
        return 'half', 'MA5<MA10排列破坏'
    return None, ''


# ---- 工具 ----
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
