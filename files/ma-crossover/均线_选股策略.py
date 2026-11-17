# -*- coding: utf-8 -*-
"""
双均线/三均线交叉策略（MA Crossover）—— 第一部分：选股策略 (聚宽 JoinQuant)
========================================================================
职责：构建候选池 g.pool（每周五盘后运行），并按行业类型给每只标的分配
      适配子策略：成长型→三均线；周期/主题型→双均线；防御型→排除。
本文件同时提供 MA / 斜率 / ADX / 布林带宽 / 均线缠绕 等【公共指标函数】，
供交易、风控模块调用。本文件不含 initialize，导入无副作用。

选股四层（对应文字版第一部分）：
  第一层 基础过滤 : 非ST、上市≥120自然日、近20日日均成交额≥1亿、收盘价≥3元
  第二层 均线多头排列: 双均线(MA10>MA20+向上+价>MA10) / 三均线(MA5>MA10>MA20>MA60...)
  第三层 行业适配   : 成长→三均线；周期/主题→双均线；防御→排除
  第四层 终筛排序   : 发散度30% + 持续性25% + 量能25% + 回调20%，取前5~10只

组合方式：可在研究环境按模块 import，或按【选股→风控→交易】拼进同一文件。
"""
import numpy as np
from jqdata import *


# ============================ 申万一级行业 → 类型映射 ============================
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


# ============================ 公共指标函数 ============================
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


# ============================ 选股主流程 ============================
def select_universe(context):
    """每周五盘后：重建候选池 g.pool，并缓存每只标的的均线/行业/适配信息。"""
    p = g.sel
    prev = context.previous_date

    if g.base_index:
        base = get_index_stocks(g.base_index, date=prev)
    else:
        base = list(get_all_securities(types=['stock'], date=prev).index)

    cd = get_current_data()
    from datetime import timedelta
    list_cutoff = prev - timedelta(days=p['min_list_days'])

    # 第一层：基础过滤（ST / 上市时长 / 停牌）
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

    raw = []  # 通过前三层的候选：(sec, info, 四维原始指标)
    for s in stage1:
        arr = get_daily_arrays(s, context.current_dt, 90,
                               fields=('high', 'low', 'close', 'volume', 'money'))
        if arr is None or len(arr['close']) < 65:
            continue
        highs, lows, closes = arr['high'], arr['low'], arr['close']
        vol, money = arr['volume'], arr['money']
        price = closes[-1]

        # S3/S4：流动性 & 价格
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
        #   快速金叉统一交给每日入场逻辑(均线_交易策略.py)在该universe上捕捉。
        if not (m20 > m60 and slope_up(closes, 60, 5)):
            continue

        # 第四层原始指标
        div = (m5 - m60) / m60 if m60 > 0 else 0.0                  # 发散度
        persist = consecutive_above(closes, 5, 10, cap=20)         # 持续性
        v5, v20 = float(np.mean(vol[-5:])), float(np.mean(vol[-20:]))
        volratio = v5 / v20 if v20 > 0 else 1.0                    # 量能配合
        hi5 = float(np.max(highs[-5:]))
        pullback = (hi5 - price) / hi5 if hi5 > 0 else 0.0         # 回调幅度

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
        score = (0.30 * divs[i] + 0.25 * pers[i] +
                 0.25 * vols[i] + 0.20 * (1 - pbs[i]))  # 回调越小得分越高
        info['score'] = score
        scored.append((s, info))
    scored.sort(key=lambda x: x[1]['score'], reverse=True)

    # 终选：同一申万一级行业≤2只，池容量取前 pool_max
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


def get_sector(security):
    info = getattr(g, 'pool_info', {}).get(security)
    if info:
        return info.get('sector', 'NA'), info.get('sector_type', 'cycle')
    ind = get_industry(security)
    name = (ind.get(security, {}) or {}).get('sw_l1', {}).get('industry_name', 'NA')
    return name, sector_type_of(name)
