# -*- coding: utf-8 -*-
"""
均线多头 + 六维评分 选股交易策略 V2.3（聚宽 JoinQuant）

V1.6 = V1.0 + 评分第 3 项「日均影线率」拆成两项（各占半项权重，合计仍按 6 项平分）：
       3a 振幅水平：t-5..t-1 日均单日振幅，<=4% 满分，>=8% 0 分
       3b 最大振幅倍数：最大单日振幅 / max(其余4日均值, 1%)，<=1.5 满分，>=3.0 0 分
       单日振幅 = (当日最高 - 当日最低) / 前一交易日收盘。其余与 V1.0 完全一致。
V1.7 = V1.6 + 评分第 6 项「爆发后滞涨」替换为「上行后平台期」：
       t-30..t-1 最高收盘 Hc 在 D 日；上行段 = Hc / (D 及之前 20 日最低收盘) - 1；
       上行段 < 20% 满分（无一波上行，不受影响）；否则 停滞天数 k = (t-1) - D，
       k<=3 满分，k>=10 0 分，中间线性。其余与 V1.6 完全一致。
V1.8 = V1.7 + 评分第 1 项改为「路径效率2.0」：窗口 10 日并包含当日，分子去掉绝对值：
       ER = (C(t) - C(t-10)) / Σ|C(k) - C(k-1)|，k = t-9..t，C(t) 用 14:55 实时价；
       ER 取值 -1..1（下跌为负），ER <= 0.2 得 0 分，>= 0.7 满分，中间线性。其余与 V1.7 完全一致。
V1.9 = V1.8 + 一票否决「距启动低点涨幅过大」按市值分档：
       现价 / (t-20..t-1 最低价) - 1，总市值 > 500 亿的大盘股超 30% 直接 0 分，其余股票超 50% 直接 0 分。
       总市值取前一交易日（聚宽 valuation.market_cap，单位亿元）。其余与 V1.8 完全一致。
V2.0 = V1.9 + 所有评分项和否决项的窗口都改为以当日 t 结尾（当日 K 线用 14:55 实时数据），窗口长度不变：
       启动低点 t-19..t 最低价；前5日涨幅 C(t)/C(t-5)-1；振幅 t-4..t；距5日最高价 t-4..t；
       趋势确认 t-4..t；平台期最高收盘在 t-29..t 中找、停滞天数 k = t - D。路径效率2.0 本已含当日。
V2.1 = V2.0 + 买入否决「近 5 日涨停过」：t-4..t 任一日最高价触及涨停价（含炸板、一字板）即不买入。
       只作用于买入，不影响评分；已持有的照常持有，按原卖出条件处理。其余与 V2.0 完全一致。
V2.2 = V2.1 + 回撤卖出的「最高点」改为买入后各日盘中最高价（原为各日 14:55 现价），
       开盘冲高、盘中冲高后回落也计入最高点。其余与 V2.1 完全一致。
V2.3 = V2.2 + 卖出评分阈值 80 -> 60；涨停买入否决 5 日 -> 10 日（t-9..t）；
       新增买入否决「近 3 日长上影线」：t-2..t 任一日 (最高价 - max(开盘价, 收盘价)) / 收盘价 > 3% 不买入
       （当日收盘用 14:55 实时价）。与涨停否决一样只拦截买入，已持有不受影响。

回测设置：频率必须选「分钟」。每日 14:55 用实时价近似收盘价，完成 选股 -> 评分 -> 卖出 -> 买入。
  * 选股/评分里的「当日」= 截至 14:55 的当日 K 线（get_bars include_now=True），
    当日成交量为 14:55 时的累计量（约为全天 95%）。
  * 换手率 = 当日累计成交量 / 前一交易日流通股本（聚宽当日换手率需收盘后才有）。

流程：
  1. 选股：全 A（剔除 ST/停牌/北交所/上市不足）跑 select()，选中即加入选股池并跨日保留，
     池内评分 < 50 才移出
  2. 评分：选股池 + 持仓 逐只 score()，三条一票否决 + 六项子分加权
  3. 卖出：持仓满足任一卖出条件即清仓
  4. 买入：选股池中评分 >= 90 的，按评分从高到低填补空位，每只 20% 总资产，最多 5 只
"""
from jqdata import *
import numpy as np
import builtins

# jqdata 的 import * 会覆盖同名内置函数（sum 已确认被覆盖），这里恢复为 Python 内置版本
# any/all 也会被覆盖成 numpy 版：np.any(生成器) 恒为 True，会导致长上影线否决对所有股票都成立
sum, min, max, abs = builtins.sum, builtins.min, builtins.max, builtins.abs
any, all, round = builtins.any, builtins.all, builtins.round


# ==================================================================
# 一、参数
# ==================================================================
class Cfg:
    BENCHMARK = '000300.XSHG'
    RUN_TIME = '14:55'
    EXCLUDE_BJ = True           # 剔除北交所
    EXCLUDE_KCB = False         # 剔除科创板（688）
    EXCLUDE_CYB = False         # 剔除创业板（300/301）
    MIN_LIST_DAYS = 90          # 上市不足 N 自然日的不参与（保证 MA60 有效）
    BARS = 70                   # 拉取的日 K 根数（含当日）
    BATCH = 500                 # get_bars 分批

    # ---- 交易 ----
    MAX_HOLD = 5
    POS_PCT = 0.20              # 每只占总资产
    BUY_SCORE = 90              # 买入阈值（>=）
    SELL_SCORE = 60             # 卖出阈值（<）
    CHECK_DAY = 3               # 买入后第 N 个交易日检查涨幅
    CHECK_MIN_GAIN = 0.0        # 第 N 日涨幅 <= 此值卖出
    STOP_LOSS = 0.05            # 距买入价下跌超 5% 卖出
    TRAIL_DD = 0.08             # 距买入后最高点（含盘中最高价）回撤超 8% 卖出
    SKIP_LIMIT_UP = True        # 涨停价买不进，跳过
    LIMIT_UP_DAYS = 10          # 买入否决：t-9..t 内最高价触及过涨停价的不买入（已持有不受影响）
    SHADOW_DAYS = 3             # 买入否决：t-2..t 内出现长上影线的不买入（已持有不受影响），0 = 关闭
    SHADOW_MAX = 0.03           # 长上影线：(当日最高价 - 实体上沿) / 当日收盘价 > 3%，实体上沿 = max(开盘, 收盘)
    POOL_EXIT_SCORE = 50        # 选股池退出：评分 < 50 移出（选中后跨日保留）
    KCB_PROTECT = 0.02          # 科创板市价单保护限价：买入 现价*(1+2%)，卖出 现价*(1-2%)

    # ---- 评分 ----
    # 六项各 100 分，按权重加权成总分。原文写"每项占 20%"但列了 6 项（合计 120%），
    # 这里默认六项等权（各 1/6）；如需改权重直接改这里，权重和不必为 1，会自动归一。
    # V1.6：第 3 项拆成 振幅水平(amp) + 最大振幅倍数(amp_x)，各 0.5，合计仍是 6 份
    WEIGHTS = {'er': 1, 'gain': 1, 'amp': 0.5, 'amp_x': 0.5, 'drop': 1, 'trend': 1, 'plat': 1}
    ER_N = 10                   # 路径效率2.0窗口：t-9..t 共 10 个变动（含当日实时价）
    ER_ZERO, ER_FULL = 0.2, 0.7 # ER <=0.2 得 0 分，>=0.7 满分
    AMP_FULL, AMP_ZERO = 0.04, 0.08         # 日均振幅 <=4% 满分，>=8% 0 分
    AMPX_FULL, AMPX_ZERO = 1.5, 3.0         # 振幅倍数 <=1.5 满分，>=3.0 0 分
    AMPX_FLOOR = 0.01                       # 倍数分母下限 1%（防一字板等极小振幅放大倍数）
    LAUNCH_LOOKBACK = 20        # 启动前低点：t-19..t 的最低价（含当日）
    LAUNCH_MAX_GAIN = 0.50      # 现价距启动低点涨幅超 50% 直接 0 分
    BIG_CAP = 500               # 大盘股：前一交易日总市值 > 500 亿元
    LAUNCH_MAX_GAIN_BIG = 0.30  # 大盘股现价距启动低点涨幅超 30% 直接 0 分
    PLAT_LOOK = 30              # 平台期：在 t-29..t 找最高收盘（含当日）
    PLAT_BASE_LOOK = 20         # 上行段起点：最高收盘日及之前 20 个交易日的最低收盘
    PLAT_RISE_MIN = 0.20        # 上行段 < 20% 视为没有一波上行，直接满分
    PLAT_K_FULL = 3             # 距最高收盘 <= 3 个交易日满分
    PLAT_K_ZERO = 10            # 距最高收盘 >= 10 个交易日 0 分

    DEBUG = False               # True 时打印每只候选的子分明细


# ==================================================================
# 二、初始化
# ==================================================================
def initialize(context):
    set_benchmark(Cfg.BENCHMARK)
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    set_slippage(PriceRelatedSlippage(0.002))
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0.0005,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5), type='stock')
    log.set_level('order', 'error')

    # 持仓跟踪：code -> {buy_price, buy_date, days, peak, force_sell}
    g.pos = {}
    # 选股池：code -> 入池日期。select() 选中即入池，跨日保留，评分 < POOL_EXIT_SCORE 移出
    g.pool = {}
    run_daily(before_market_open, time='before_open')
    run_daily(trade, time=Cfg.RUN_TIME)
    log.info('均线多头评分策略 V2.3 初始化完成（请确认回测频率为"分钟"）')


def before_market_open(context):
    # 持仓交易日计数 +1（买入当日为第 0 日）
    for code, p in g.pos.items():
        p['days'] += 1


# ==================================================================
# 三、主流程
# ==================================================================
def trade(context):
    cd = get_current_data()
    _reconcile(context)

    universe = _universe(context, cd)
    uset = set(universe)
    held = list(context.portfolio.positions.keys())
    # 池内股票若变 ST/退市/上市状态异常（不在 universe 且非停牌）直接移出
    for c in list(g.pool):
        if c not in uset and not cd[c].paused:
            g.pool.pop(c)
    bars = _get_bars(context, list(uset | set(held) | set(g.pool)))

    # ---- 选股：当日选中的加入选股池 ----
    pre = [c for c in universe if c in bars and _select_price(bars[c])]
    turnover = _turnover(context, pre, bars)
    picked = [c for c in pre if 3 < turnover.get(c, 0) < 10]
    new_in = [c for c in picked if c not in g.pool]
    for c in new_in:
        g.pool[c] = context.current_dt.date()

    # ---- 评分（选股池 + 持仓），池内评分 < 50 移出 ----
    scores = {}
    to_score = [c for c in set(g.pool) | set(held) if c in bars]
    caps = _market_cap(context, to_score)
    for c in to_score:
        scores[c], detail = score(bars[c], caps.get(c))
        if Cfg.DEBUG and c in g.pool:
            log.info('[评分] %s %.1f %s' % (c, scores[c], detail))
    out = [c for c in g.pool if c in scores and scores[c] < Cfg.POOL_EXIT_SCORE]
    for c in out:
        g.pool.pop(c)

    # ---- 卖出 ----
    sold = set()
    for code in held:
        pos = context.portfolio.positions[code]
        if pos.closeable_amount <= 0:
            continue                      # 当日买入 T+1 不可卖
        reason = _sell_reason(code, bars.get(code), scores.get(code), cd)
        if reason:
            _sell(context, code, reason, scores.get(code), cd)
            sold.add(code)

    # ---- 买入 ----
    positions = context.portfolio.positions
    holding = [c for c in list(positions.keys()) if c not in sold and positions[c].total_amount > 0]
    slots = Cfg.MAX_HOLD - len(holding)
    log.info('今日选中 %d 只(新入池 %d)，移出 %d，选股池 %d 只，持仓 %d'
             % (len(picked), len(new_in), len(out), len(g.pool), len(holding)))
    if slots <= 0:
        return
    cands = [c for c in g.pool if c not in holding and c not in sold and not cd[c].paused
             and scores.get(c, 0) >= Cfg.BUY_SCORE]
    if Cfg.SKIP_LIMIT_UP:
        cands = [c for c in cands if cd[c].last_price < cd[c].high_limit]
    if Cfg.LIMIT_UP_DAYS > 0:
        touched = _touched_limit_up(context, cands, bars, cd)
        for c in cands:
            if c in touched:
                log.info('[买入否决] %s 近%d日涨停过' % (c, Cfg.LIMIT_UP_DAYS))
        cands = [c for c in cands if c not in touched]
    if Cfg.SHADOW_DAYS > 0:
        shadow = [c for c in cands if _long_upper_shadow(bars[c])]
        for c in shadow:
            log.info('[买入否决] %s 近%d日有长上影线' % (c, Cfg.SHADOW_DAYS))
        cands = [c for c in cands if c not in shadow]
    cands.sort(key=lambda c: -scores[c])
    log.info('池内评分>=%d 共 %d 只，空位 %d' % (Cfg.BUY_SCORE, len(cands), slots))

    for code in cands[:slots]:
        _buy(context, code, scores[code], cd)


# ==================================================================
# 四、选股（与原 select() 口径一致；ST 已在股票池剔除，换手率单独批量算）
# ==================================================================
def _select_price(b):
    close, volume = b['close'], b['volume']
    if len(close) < 60:
        return False
    if close[-1] <= 0 or volume[-1] <= 0 or close[-6] <= 0:
        return False
    ma5, ma10, ma20, ma60 = (close[-n:].mean() for n in (5, 10, 20, 60))
    if not (ma5 > ma10 > ma20 > ma60):           # 均线多头
        return False
    if close[-1] <= ma5:                          # 股价 > MA5
        return False
    if (close[-1] - close[-6]) / close[-6] <= 0.05:   # 近 5 日涨幅 > 5%
        return False
    vol_ma5_prev = volume[-6:-1].mean()           # = MA(volume,5)[-2]
    if vol_ma5_prev <= 0 or volume[-1] / vol_ma5_prev <= 1.3:   # 放量 1.3 倍
        return False
    return True


def _turnover(context, codes, bars):
    """换手率(%) = 当日累计成交量 / 前一交易日流通股本"""
    out = {}
    if not codes:
        return out
    df = get_fundamentals(query(valuation.code, valuation.circulating_cap)
                          .filter(valuation.code.in_(codes)), date=context.previous_date)
    for code, cap in zip(df['code'], df['circulating_cap']):
        if cap and cap > 0:                       # circulating_cap 单位：万股
            out[code] = bars[code]['volume'][-1] / (cap * 1e4) * 100
    return out


def _touched_limit_up(context, codes, bars, cd):
    """t-4..t 内最高价触及过涨停价的股票（不复权价比较；当日用 14:55 的最高价）"""
    out = set()
    if not codes:
        return out
    # 当日：bars 前复权到当日，当日价格即真实价
    for c in codes:
        if bars[c]['high'][-1] >= cd[c].high_limit - 0.005:
            out.add(c)
    # t-(N-1)..t-1：不复权日 K 与当日涨停价
    if Cfg.LIMIT_UP_DAYS <= 1:
        return out
    df = get_price(codes, end_date=context.previous_date, count=Cfg.LIMIT_UP_DAYS - 1, frequency='daily',
                   fields=['high', 'high_limit', 'paused'], fq=None, panel=False)
    hit = df[(df['paused'] == 0) & (df['high'] >= df['high_limit'] - 0.005)]
    out.update(hit['code'].tolist())
    return out


def _long_upper_shadow(b):
    """t-(N-1)..t 内是否有一日 上影线 / 收盘 > SHADOW_MAX（当日收盘为 14:55 实时价）
    上影线 = 最高 - max(开盘, 收盘)；阴线实体不算上影线"""
    o, h, c = b['open'], b['high'], b['close']
    i = len(c) - 1
    return any((h[k] - max(o[k], c[k])) / c[k] > Cfg.SHADOW_MAX
               for k in range(i - Cfg.SHADOW_DAYS + 1, i + 1))


def _market_cap(context, codes):
    """前一交易日总市值（亿元）"""
    if not codes:
        return {}
    df = get_fundamentals(query(valuation.code, valuation.market_cap)
                          .filter(valuation.code.in_(codes)), date=context.previous_date)
    return dict(zip(df['code'], df['market_cap']))


# ==================================================================
# 五、评分
# ==================================================================
def _ma(c, n, k):
    """以索引 k 结尾的 n 日均线"""
    return c[k - n + 1:k + 1].mean()


def _lin(x, zero, full):
    """线性插值：x=zero 时 0 分，x=full 时 100 分，两端截断"""
    return float(min(max((x - zero) / (full - zero), 0.0), 1.0) * 100)


def score(b, cap=None):
    """cap: 前一交易日总市值（亿元），取不到时按非大盘股处理"""
    o, h, l, c = b['open'], b['high'], b['low'], b['close']
    i = len(c) - 1                                # t（当日，收盘用实时价）
    if i < 30:
        return 0.0, 'bars<31'

    # ---- 一票否决 ----
    if _ma(c, 5, i) < _ma(c, 10, i):
        return 0.0, 'veto:MA5<MA10'
    if l[i] < _ma(c, 20, i):
        return 0.0, 'veto:low<MA20'
    launch_low = l[i - Cfg.LAUNCH_LOOKBACK + 1:i + 1].min()      # t-19..t
    max_gain = Cfg.LAUNCH_MAX_GAIN_BIG if cap and cap > Cfg.BIG_CAP else Cfg.LAUNCH_MAX_GAIN
    if launch_low > 0 and c[i] / launch_low - 1 > max_gain:
        return 0.0, 'veto:launch+%.0f%%' % ((c[i] / launch_low - 1) * 100)

    s = {}
    # 1. 路径效率2.0 ER = (C(t)-C(t-N)) / Σ|C(k)-C(k-1)|, k=t-N+1..t（含当日实时价，下跌为负）
    seg = c[i - Cfg.ER_N:i + 1]
    path = np.abs(np.diff(seg)).sum()
    er = (seg[-1] - seg[0]) / path if path > 0 else 0.0
    s['er'] = _lin(er, Cfg.ER_ZERO, Cfg.ER_FULL)

    # 2. 前 5 日涨幅（t vs t-5，C(t) 为实时价）
    s['gain'] = _gain_score(c[i] / c[i - 5] - 1)

    # 3. 单日振幅 = (最高 - 最低) / 前收，取 t-4..t（当日用 14:55 的最高/最低）
    amps = np.array([(h[k] - l[k]) / c[k - 1] for k in range(i - 4, i + 1)])
    # 3a. 振幅水平：日均振幅 <=4% 满分，>=8% 0 分
    s['amp'] = _lin(amps.mean(), Cfg.AMP_ZERO, Cfg.AMP_FULL)
    # 3b. 最大振幅倍数：最大单日振幅 / max(其余 4 日均值, 1%)，<=1.5 满分，>=3.0 0 分
    amp_max = amps.max()
    rest = (amps.sum() - amp_max) / 4
    s['amp_x'] = _lin(amp_max / max(rest, Cfg.AMPX_FLOOR), Cfg.AMPX_ZERO, Cfg.AMPX_FULL)

    # 4. 现价距 5 日（t-4..t）最高价，>-3% 满分，<-10% 0 分
    s['drop'] = _lin(c[i] / h[i - 4:i + 1].max() - 1, -0.10, -0.03)

    # 5. 趋势确认：t-4..t 每日 收盘>MA5>MA10>MA20，少一日 -25（当日收盘用实时价）
    miss = 0
    for k in range(i - 4, i + 1):
        m5, m10, m20 = _ma(c, 5, k), _ma(c, 10, k), _ma(c, 20, k)
        if not (c[k] > m5 and m5 > m10 and m10 > m20):
            miss += 1
    s['trend'] = float(max(0, 100 - 25 * miss))

    # 6. 上行后平台期
    s['plat'] = score_platform(c, i)

    w = Cfg.WEIGHTS
    total = sum(s[k] * w[k] for k in s) / sum(w.values())
    detail = ' '.join('%s=%.0f' % (k, s[k]) for k in ('er', 'gain', 'amp', 'amp_x', 'drop', 'trend', 'plat'))
    return float(total), detail


def _gain_score(g):
    g *= 100
    if 5 <= g <= 20:
        return 100.0
    if 4 <= g < 5 or 20 < g <= 23:
        return 75.0
    if 3 <= g < 4 or 23 < g <= 26:
        return 50.0
    if 2 <= g < 3 or 26 < g <= 30:
        return 25.0
    return 0.0


def score_platform(close, i):
    """上行后平台期。close: 收盘价序列; i: 当前交易日 t 的索引（当日收盘为实时价）"""
    lo = max(0, i - Cfg.PLAT_LOOK + 1)
    D = lo + int(np.argmax(close[lo:i + 1]))      # t-29..t 最高收盘日（并列取最早）
    base = close[max(0, D - Cfg.PLAT_BASE_LOOK):D + 1].min()
    if base <= 0 or close[D] / base - 1 < Cfg.PLAT_RISE_MIN:
        return 100.0                               # 没有一波上行, 不影响
    k = i - D                                      # 停滞天数
    return _lin(k, Cfg.PLAT_K_ZERO, Cfg.PLAT_K_FULL)


# ==================================================================
# 六、卖出 / 买入
# ==================================================================
def _sell_reason(code, b, sc, cd):
    p = g.pos.get(code)
    price = cd[code].last_price
    if p is None or b is None or price <= 0 or np.isnan(price):
        return None
    if p.get('force_sell'):
        return '前日卖出未成交，继续卖出'

    # 最高点含当日盘中最高价（bars 前复权到当日，当日价格即真实价；买入当日 T+1 不会走到这里）
    p['peak'] = max(p['peak'], price, b['high'][-1])
    c = b['close']
    i = len(c) - 1
    if i >= 10 and _ma(c, 5, i) < _ma(c, 10, i) and _ma(c, 5, i - 1) >= _ma(c, 10, i - 1):
        return 'MA5下穿MA10死叉'
    if p['days'] == Cfg.CHECK_DAY and price / p['buy_price'] - 1 <= Cfg.CHECK_MIN_GAIN:
        return '买入后第%d日涨幅%.2f%%<=0' % (Cfg.CHECK_DAY, (price / p['buy_price'] - 1) * 100)
    if sc is not None and sc < Cfg.SELL_SCORE:
        return '评分%.1f<%d' % (sc, Cfg.SELL_SCORE)
    if price / p['buy_price'] - 1 < -Cfg.STOP_LOSS:
        return '止损：距买入价%.2f%%' % ((price / p['buy_price'] - 1) * 100)
    if price / p['peak'] - 1 < -Cfg.TRAIL_DD:
        return '回撤：距最高点%.2f%%' % ((price / p['peak'] - 1) * 100)
    return None


def _sell(context, code, reason, sc, cd):
    o = order_target(code, 0, _style(code, cd, side='sell'))
    if o is not None and o.filled > 0 and code not in context.portfolio.positions:
        p = g.pos.pop(code, {})
        pnl = cd[code].last_price / p.get('buy_price', cd[code].last_price) - 1
        log.info('[卖出] %s %s | 评分=%s 持有%s日 收益%.2f%%'
                 % (code, reason, '%.1f' % sc if sc is not None else '-', p.get('days', '-'), pnl * 100))
    else:
        # 跌停等原因未成交：次日继续卖
        if code in g.pos:
            g.pos[code]['force_sell'] = True
        log.warn('[卖出未成交] %s %s' % (code, reason))


def _buy(context, code, sc, cd):
    value = min(context.portfolio.total_value * Cfg.POS_PCT, context.portfolio.available_cash)
    if value < cd[code].last_price * 100:
        return
    o = order_value(code, value, _style(code, cd, side='buy'))
    if o is not None and o.filled > 0:
        pos = context.portfolio.positions[code]
        g.pos[code] = {'buy_price': pos.avg_cost, 'buy_date': context.current_dt.date(),
                       'days': 0, 'peak': pos.avg_cost, 'force_sell': False}
        log.info('[买入] %s 评分=%.1f 价=%.2f 数量=%d' % (code, sc, pos.avg_cost, o.filled))


def _style(code, cd, side):
    """科创板市价单必须带保护限价；其他板块用普通市价单"""
    if not code.startswith('688'):
        return None
    d = cd[code]
    if side == 'buy':
        px = min(d.last_price * (1 + Cfg.KCB_PROTECT), d.high_limit)
    else:
        px = max(d.last_price * (1 - Cfg.KCB_PROTECT), d.low_limit)
    return MarketOrderStyle(round(px, 2))


def _reconcile(context):
    """g.pos 与实际持仓对齐（防重启/部分成交造成脱钩）"""
    held = context.portfolio.positions
    for code in list(g.pos):
        if code not in held or held[code].total_amount == 0:
            g.pos.pop(code)
    for code, pos in held.items():
        if code not in g.pos and pos.total_amount > 0:
            g.pos[code] = {'buy_price': pos.avg_cost, 'buy_date': context.current_dt.date(),
                           'days': 0, 'peak': pos.avg_cost, 'force_sell': False}


# ==================================================================
# 七、数据
# ==================================================================
def _universe(context, cd):
    today = context.current_dt.date()
    secs = get_all_securities(['stock'], date=today)
    out = []
    for code, row in secs.iterrows():
        if (today - row['start_date']).days < Cfg.MIN_LIST_DAYS:
            continue
        if Cfg.EXCLUDE_BJ and code.endswith('.BJ'):
            continue
        if Cfg.EXCLUDE_KCB and code.startswith('688'):
            continue
        if Cfg.EXCLUDE_CYB and code.startswith(('300', '301')):
            continue
        d = cd[code]
        if d.paused or d.is_st or 'ST' in d.name or '退' in d.name:
            continue
        out.append(code)
    return out


def _get_bars(context, codes):
    """日 K（含当日 14:55 的未完成 K 线），前复权到当日"""
    out = {}
    fields = ['open', 'high', 'low', 'close', 'volume']
    for k in range(0, len(codes), Cfg.BATCH):
        chunk = codes[k:k + Cfg.BATCH]
        try:
            data = get_bars(chunk, count=Cfg.BARS, unit='1d', fields=fields, include_now=True,
                            end_dt=context.current_dt, fq_ref_date=context.current_dt.date(), df=False)
        except Exception as e:
            log.warn('get_bars 失败(%d-%d): %s' % (k, k + len(chunk), e))
            continue
        for code, arr in data.items():
            if arr is None or len(arr) == 0:
                continue
            b = {f: np.asarray(arr[f], dtype=float) for f in fields}
            if np.isnan(b['close']).any():
                continue
            out[code] = b
    return out
