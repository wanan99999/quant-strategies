# -*- coding: utf-8 -*-
"""
热点板块交易策略 V2.0
========================================================
对应《热点板块交易策略框架_V2.0》。平台：聚宽 JoinQuant。
策略类型：中短线热点交易（均线金叉 + 大盘顶背离 双层架构 + 热点生命周期/退潮撤退）。

继承基础策略（中短线交易策略_V2.0）的：
  · 2 只持仓上限、单票 50%、25%/50% 两阶段仓位
  · 大盘顶背离四条件、换手率/ST 过滤、买入后动能(可选)
  · 14:50 盘后决策口径：买入/加仓当日收盘成交；顶背离减半/退潮/破位/止损
    次日开盘成交；隔夜止损开盘检查。

新增（热点适配）：
  1) 板块环境：板块成分股等权归一化合成"板块指数"，据此算 板块MA5 / 近5日涨幅 /
     单日涨幅 / 成交额萎缩 / MA20，用于生命周期与退潮识别。
  2) 五维评分（趋势25 量能20 位置20 热点强度20 大盘共振15），科技/小市值股
     取消大盘共振、并入热点强度(→35)；蓝筹正常考核；中间类型宽松考核。
  3) 热点生命周期：启动/扩散/加速/高潮/退潮/冰点。高潮/退潮/冰点禁止新开仓。
  4) 四重退潮信号 D1-D4（龙头破位/扩散失败/量能萎缩/板块破位），任一触发分批撤退，
     ≥2 个同时触发直接清仓。
  5) 三阶段撤退：D 触发→卖50%(止损容忍收紧至10%)；次日板块再跌>1%→再卖50%(总25%)；
     板块/个股破MA20 或 顶背离→清仓。
  6) 四类特殊止损：时间止损(5日未盈利)、加速止损(退潮期回撤10%)、
     隔夜止损(涨停次日低开>3%卖50%)、板块崩盘止损(板块单日跌>5%清仓)。
  7) 大盘顶背离冷静期：触发后 10 个交易日禁止一切买入/加仓。

数据源口径：板块 = 聚宽概念/行业代码的成分股（可每日切换）。板块指数为成分股
等权归一化均值（非真实指数），故"板块涨幅/MA"为相对量纲，阈值判断仍成立。
"""

import numpy as np
import pandas as pd
from jqdata import *


# ==================================================================
# 一、参数配置（对应文档第七部分：策略参数汇总）
# ==================================================================
class Cfg:
    BENCHMARK = '000001.XSHG'

    # ---- 热点板块定义（成分股来源）----
    # SECTOR_TYPE: 'concept' 概念板块 / 'industry' 行业 / 'custom' 自定义清单
    SECTOR_TYPE = 'concept'
    # 【推荐】按名称填写，程序自动查代码：
    #   concept  → get_concepts() 里的 name（如 '新能源汽车' '存储器' 'CPO'）
    #   industry → 自动扫 sw_l1/sw_l2/sw_l3/jq_l1/jq_l2/zjw 的 name（如 '半导体'）
    SECTOR_NAME = '新能源汽车'
    # 若已知代码可直接填（SECTOR_NAME 留空时生效）：
    #   concept  → get_concepts() 的 index（如 'SC0001'）
    #   industry → get_industries() 的 index（如 'HY007' / 申万 '801080'）
    SECTOR_CODE = ''
    SECTOR_CUSTOM = [                # SECTOR_TYPE='custom' 时使用
        # '300308.XSHE', '002049.XSHE', ...
    ]

    # ---- 仓位（硬上限）----
    MAX_HOLDINGS = 2
    PROBE_NORMAL = 0.25              # 试探（正常期）
    PROBE_ACCEL  = 0.15             # 试探（加速期）
    FULL_RATIO   = 0.50            # 确认加仓上限（单票 50%）

    # ---- 选股硬过滤 ----
    MIN_LIST_DAYS   = 60
    TURNOVER_MIN    = 3.0          # 近5日日均换手率 <3% 剔除（get_valuation 单位=%）
    ONE_WORD_LIMIT_SKIP = True     # 一字涨停(开=收=最高且涨停)剔除，确保可买

    # ---- 均线 ----
    MA_SHORT = 5
    MA_LONG  = 10
    MA_MID   = 20

    # ---- 选股 / 持仓 ----
    TOP_N     = 2                   # 取评分前 N 作候选
    WARN_RANK = 5                   # 持仓跌出前 5 名 → 预警(仅记录)

    # ---- 板块生命周期阈值（板块近5日涨幅）----
    SECTOR_ACCEL_LOW  = 0.08        # ≥8% 进入加速期
    SECTOR_HIGH       = 0.15        # >15% 进入高潮期（禁开仓）
    ACCEL_NO_SURGE    = 0.07        # 加速期确认加仓额外条件：近3日无单日涨幅>7%

    # ---- 大盘顶背离四条件 ----
    NEW_HIGH_DAYS = 10
    BREADTH_MA    = 5
    WEAK_PCT      = 0.50
    CUM_DOWN_DAYS = 3
    CUM_DOWN_PCT  = 0.50
    WARN_BARS     = 10             # 顶背离冷静期（交易日，禁一切买入/加仓）

    # ---- 股票类型分类（大盘共振差异化）----
    TECH_INDUSTRIES = {'电子', '计算机', '通信', '传媒'}
    TECH_MKTCAP_MAX = 200.0         # 总市值 <200 亿 视为小市值/成长（get_valuation 单位=亿元）
    BLUE_INDUSTRIES = {'银行', '食品饮料', '家用电器', '非银金融'}
    BLUE_MKTCAP_MIN = 1000.0        # 总市值 >1000 亿 视为蓝筹
    CORR_HIGH = 0.6                 # 蓝筹大盘共振：相关系数 >0.6 满分
    CORR_LOW  = 0.3                 # 0.3-0.6 半分，<0.3 不得分
    IDX_WEAK_DAYS = 3               # 上证连续3日收于MA5下方 → 大盘单边下跌(中间类型扣分)

    # ---- 退潮信号 D1-D4 ----
    D1_LEADER_DROP = 0.05           # 龙头单日跌幅 >5% 且破 MA10
    D2_DROP        = 0.03           # 前5中，单日跌 >3%
    D2_COUNT       = 3              # ≥3 只 → 触发
    D3_SHRINK      = 0.30           # 近5日均额较前5日萎缩 >30%
    D4_BELOW_DAYS  = 2              # 板块指数连续2日收于MA5下方
    RETREAT_STAGE2_DROP = 0.01     # 撤退阶段二：次日板块再跌 >1%

    # ---- 止损 ----
    BASE_STOP      = 0.20           # 基础回撤止损(从买入价/最高价回撤)
    ACCEL_STOP     = 0.10           # 加速止损(退潮期回撤)
    TIME_STOP_BARS = 5              # 时间止损：买入满5个交易日未盈利
    OVERNIGHT_GAP  = 0.03           # 隔夜止损：涨停次日低开 >3% → 卖50%
    SECTOR_CRASH   = 0.05           # 板块崩盘：板块单日跌 >5% → 清仓
    G3_DROP        = 0.07           # G3：单日跌 >7%
    G3_VOL_MULT    = 2.0            # G3：且成交量放大 >2 倍

    COOLDOWN_DAYS  = 5              # 清仓后冷却(交易日)不再买回

    # ---- 调试 ----
    DEBUG_SCORE   = False
    DEBUG_SECTOR  = False
    DEBUG_HOLDINGS = False


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

    # 持仓：code -> {stage:'full'/'half', half_source:'probe'/'divergence',
    #   half_price, full_price, buy_date, buy_bar, highest, retreat_stage(0/1/2)}
    g.holdings = {}
    g.cooldown = {}                 # code -> 剩余交易日
    g.pending_sells = []            # 次日开盘执行的卖出
    g.warn_left = 0                 # 顶背离冷静期剩余交易日（>0 禁买）
    g.divergence_today = False
    g.bar = 0                       # 交易日计数
    g.sector_members = []           # 当前热点板块成分股
    g.sector_code_cache = {}        # (type,name)->code 解析缓存
    g.sector_meta = {}              # code -> {'ind':申万一级, 'mktcap':亿元, 'type':'tech'/'blue'/'mid'}
    g.sector_retreat_stage = 0      # 板块撤退阶段 0/1/2/3
    g.retreat_ref = None            # 撤退阶段一时的板块指数基准（用于阶段二比较）

    run_daily(before_market_open, time='before_open')
    run_daily(market_open,        time='09:31')
    run_daily(before_close,       time='14:50')

    log.info('热点板块策略 V2.0 初始化完成 [板块类型=%s 代码=%s]'
             % (Cfg.SECTOR_TYPE, Cfg.SECTOR_CODE or Cfg.SECTOR_CUSTOM))


# ==================================================================
# 三、盘前
# ==================================================================
def before_market_open(context):
    g.bar += 1
    for code in list(g.cooldown.keys()):
        g.cooldown[code] -= 1
        if g.cooldown[code] <= 0:
            del g.cooldown[code]
    if g.warn_left > 0:
        g.warn_left -= 1
    g.divergence_today = False
    update_sector_members(context)


def update_sector_members(context):
    """每日刷新热点板块成分股。如需每日切换热点，在此重写 g.sector_members。"""
    prev = context.previous_date
    try:
        if Cfg.SECTOR_TYPE == 'concept':
            code = _resolve_concept_code(Cfg.SECTOR_NAME) or Cfg.SECTOR_CODE
            g.sector_members = list(get_concept_stocks(code, date=prev)) if code else []
        elif Cfg.SECTOR_TYPE == 'industry':
            code = _resolve_industry_code(Cfg.SECTOR_NAME) or Cfg.SECTOR_CODE
            g.sector_members = list(get_industry_stocks(code, date=prev)) if code else []
        elif Cfg.SECTOR_TYPE == 'custom':
            g.sector_members = list(Cfg.SECTOR_CUSTOM)
        else:
            g.sector_members = []
    except Exception as e:
        log.warn('板块成分股获取失败：%s' % e)
        g.sector_members = []
    if not g.sector_members:
        log.warn('板块成分股为空：请检查 Cfg.SECTOR_NAME / SECTOR_CODE / SECTOR_CUSTOM。')


def _resolve_concept_code(name):
    """按概念名称查代码（get_concepts 的 index 即概念代码）。名称需完全匹配；结果缓存。"""
    if not name:
        return None
    cache = g.sector_code_cache
    key = ('concept', name)
    if key in cache:
        return cache[key]
    code = None
    try:
        cons = get_concepts()
        hit = cons[cons['name'] == name]
        if not hit.empty:
            code = hit.index[0]
        else:
            log.warn('未找到概念「%s」。可用 get_concepts() 查看全部名称。' % name)
    except Exception as e:
        log.warn('概念代码解析失败：%s' % e)
    cache[key] = code
    return code


def _resolve_industry_code(name):
    """按行业名称查代码，依次扫描各分类体系（申万三级/聚宽二级/证监会）；结果缓存。"""
    if not name:
        return None
    cache = g.sector_code_cache
    key = ('industry', name)
    if key in cache:
        return cache[key]
    code = None
    for scheme in ('sw_l1', 'sw_l2', 'sw_l3', 'jq_l1', 'jq_l2', 'zjw'):
        try:
            inds = get_industries(name=scheme)
            hit = inds[inds['name'] == name]
            if not hit.empty:
                code = hit.index[0]
                break
        except Exception:
            continue
    if code is None:
        log.warn('未找到行业「%s」。可用 get_industries(name="sw_l1") 等查看名称。' % name)
    cache[key] = code
    return code


# ==================================================================
# 四、开盘：隔夜止损 + 执行昨日登记的卖出
# ==================================================================
def market_open(context):
    cd = get_current_data()
    # 1) 隔夜止损：涨停次日低开 >3% → 卖 50%（开盘价成交）
    _overnight_stop(context, cd)
    # 2) 执行昨日登记的卖出（顶背离减半 / 退潮 / 破位 / 止损）
    _execute_pending(context, cd)


def _overnight_stop(context, cd):
    for code in list(g.holdings.keys()):
        if _pending_has(code):
            continue
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            continue
        closes = _hist_close(code, context.previous_date, 2)
        if closes is None or len(closes) < 2:
            continue
        prev_close, prev_prev = float(closes[-1]), float(closes[-2])
        lp = _limit_up_pct(code)
        was_limit = prev_close >= prev_prev * (1 + lp - 0.005)   # 前一日涨停
        day_open = cd[code].day_open
        if was_limit and day_open > 0 and day_open <= prev_close * (1 - Cfg.OVERNIGHT_GAP):
            target = _round100(pos.total_amount * 0.5)
            order_target(code, target)
            h = g.holdings.get(code)
            if h and h['stage'] == 'full':
                h['stage'] = 'half'
                h['half_source'] = 'probe'
            log.info('[隔夜止损] %s 涨停次日低开%.1f%% 卖至%d股'
                     % (code, (day_open / prev_close - 1) * 100, target))


def _execute_pending(context, cd):
    if not g.pending_sells:
        return
    retry = []
    for od in g.pending_sells:
        code, action = od['code'], od['action']
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            g.holdings.pop(code, None)
            if action == 'clear':
                g.cooldown[code] = Cfg.COOLDOWN_DAYS
            continue
        before = pos.total_amount
        target = 0 if action == 'clear' else _round100(before * 0.5)   # reduce=减至当前一半
        order_target(code, target)
        pos2 = context.portfolio.positions.get(code)
        remaining = pos2.total_amount if pos2 is not None else 0
        if action == 'reduce':
            if remaining < before:
                h = g.holdings.get(code)
                if h:
                    h['stage'] = 'half'
                    if od.get('source'):
                        h['half_source'] = od['source']
                log.info('[减仓成交] %s %d→%d（%s）' % (code, before, remaining, od.get('reason', '')))
            else:
                retry.append(od)
                log.warn('[减仓未成交] %s（疑跌停/停牌），次日重试' % code)
        else:  # clear
            if remaining <= 0:
                g.holdings.pop(code, None)
                g.cooldown[code] = Cfg.COOLDOWN_DAYS
                log.info('[清仓成交] %s（%s）' % (code, od.get('reason', '')))
            else:
                retry.append(od)
                log.warn('[清仓未成交] %s 仍持%d股，次日重试' % (code, remaining))
    g.pending_sells = retry


# ==================================================================
# 五、收盘前主流程（14:50 决策）
# ==================================================================
def before_close(context):
    cd = get_current_data()
    prev = context.previous_date

    _update_holdings_highest(context, cd)
    _reconcile(context, cd)

    # ---- 大盘数据 ----
    idx_hist = _hist_close(Cfg.BENCHMARK, prev, 25)
    idx_today = _cur_index(Cfg.BENCHMARK, context.current_dt)
    idx_ok = idx_hist is not None and len(idx_hist) >= 11 and idx_today is not None

    # ---- 板块快照 + 成分股统计 ----
    members = list(g.sector_members)
    sec, mem = None, None
    if members:
        try:
            sec, mem = _sector_snapshot(context, cd, members, prev)
        except Exception as e:
            log.warn('板块快照计算异常：%s' % e)
    if sec is None:
        log.warn('板块数据不足：本日仅执行卖出/止损，跳过买入。')

    # ---- 股票类型元数据（申万一级 + 市值 → tech/blue/mid）----
    if members:
        _update_sector_meta(context, members, prev)

    # ---- 1) 大盘顶背离（=冷静期触发）----
    divergence = False
    if idx_ok:
        try:
            divergence = _check_divergence(context, cd, idx_hist[-12:], idx_today)
        except Exception as e:
            log.warn('顶背离计算异常：%s' % e)
    g.divergence_today = divergence
    if divergence:
        g.warn_left = Cfg.WARN_BARS
        log.info('[顶背离] 触发：启动%d交易日冷静期，禁一切买入/加仓' % Cfg.WARN_BARS)

    # ---- 2) 卖出评估（登记，次日开盘执行）----
    idx_weak = _index_weak(idx_hist, idx_today) if idx_ok else False
    _evaluate_sells(context, cd, sec, mem, divergence)

    # ---- 板块生命周期阶段 ----
    phase = _phase(sec, g.sector_retreat_stage)
    if Cfg.DEBUG_SECTOR and sec:
        log.info('[板块] 阶段=%s 近5日=%.1f%% 单日=%.1f%% 站上MA5=%s 撤退阶段=%d'
                 % (phase, sec['ret5'] * 100, sec['ret1'] * 100, sec['above_ma5'], g.sector_retreat_stage))

    if not idx_ok:
        log.warn('上证数据不足：本日跳过买入。')
        return

    # ---- 3) 冷静期内禁买 ----
    if g.warn_left > 0:
        return

    # ---- 4) 买入 / 加仓（当日收盘价执行）----
    idx_ma5 = (np.asarray(idx_hist[-4:], dtype=float).sum() + idx_today) / 5.0
    try:
        _rebalance(context, cd, members, sec, mem, idx_hist, idx_today, idx_ma5, idx_weak, phase)
    except Exception as e:
        import traceback
        log.error('[买入调仓异常] %s\n%s' % (e, traceback.format_exc()))

    # ---- 撤退阶段复位：空仓且板块重新站上MA5 ----
    if g.sector_retreat_stage > 0 and not g.holdings and sec and sec['above_ma5']:
        g.sector_retreat_stage = 0
        g.retreat_ref = None


# ==================================================================
# 六、卖出评估（优先级：退潮 > 顶背离 > 破位 > 特殊止损 > 基础止损）
# ==================================================================
def _evaluate_sells(context, cd, sec, mem, divergence):
    if not g.holdings:
        return

    # --- 第一优先级：退潮信号 D1-D4 + 三阶段撤退（板块级）---
    if sec is not None and mem is not None:
        try:
            _apply_retreat(context, cd, sec, mem, divergence)
        except Exception as e:
            log.warn('退潮撤退评估异常：%s' % e)

    # --- 板块崩盘止损：板块单日跌 >5% → 全部清仓（属特殊止损，但灾难级优先执行）---
    if sec is not None and sec['ret1'] <= -Cfg.SECTOR_CRASH:
        for code in list(g.holdings.keys()):
            if not _pending_has(code):
                pos = context.portfolio.positions.get(code)
                if pos:
                    _queue_clear(code, '板块崩盘(单日跌%.1f%%)' % (sec['ret1'] * 100))
        return

    # --- 逐股：顶背离 / 破位 / 特殊止损 / 基础止损 ---
    for code in list(g.holdings.keys()):
        if _pending_has(code):
            continue
        try:
            _decide_one_sell(context, cd, code, sec, divergence)
        except Exception as e:
            log.warn('持仓 %s 卖出评估异常：%s' % (code, e))


def _apply_retreat(context, cd, sec, mem, divergence):
    """四重退潮信号 + 三阶段撤退。"""
    count = _retreat_count(cd, sec, mem)
    stage = g.sector_retreat_stage
    held = [c for c in g.holdings if not _pending_has(c)]

    # ≥2 个退潮信号同时触发 → 直接清仓
    if count >= 2:
        for code in held:
            _queue_clear(code, '退潮多重信号(D触发%d个)' % count)
        g.sector_retreat_stage = 3
        return

    # 首次触发（阶段一）：卖50%，止损容忍收紧至10%
    if stage == 0 and count >= 1:
        g.sector_retreat_stage = 1
        g.retreat_ref = sec['level_today']
        for code in held:
            g.holdings[code]['retreat_stage'] = 1
            _queue_reduce(code, '退潮撤退阶段一(卖50%)', source='probe')
        log.info('[退潮] 阶段一触发(D%d)：持仓减半，止损容忍→10%%' % count)
        return

    # 已在撤退中：检查阶段三清仓 / 阶段二再减
    if stage >= 1:
        below20 = sec['ma20'] is not None and sec['level_today'] < sec['ma20']
        stock_below20 = _any_holding_below_ma20(cd, held)
        if below20 or stock_below20 or divergence:
            for code in held:
                _queue_clear(code, '退潮撤退阶段三清仓')
            g.sector_retreat_stage = 3
            log.info('[退潮] 阶段三触发：板块/个股破MA20 或 顶背离 → 清仓')
            return
        if stage == 1 and g.retreat_ref and sec['level_today'] <= g.retreat_ref * (1 - Cfg.RETREAT_STAGE2_DROP):
            g.sector_retreat_stage = 2
            for code in held:
                g.holdings[code]['retreat_stage'] = 2
                _queue_reduce(code, '退潮撤退阶段二(再卖50%)', source='probe')
            log.info('[退潮] 阶段二触发：板块继续下跌，仓位降至约25%%')


def _retreat_count(cd, sec, mem):
    """返回 D1-D4 中触发的数量。"""
    n = 0
    today, closes = mem['today'], mem['closes']

    # D1：板块龙头(近10日涨幅第一)单日跌>5% 且跌破MA10
    leader = mem.get('leader')
    if leader is not None and leader in today.index:
        prev_c = closes[leader].dropna()
        if len(prev_c) >= 10:
            drop = today[leader] / prev_c.values[-1] - 1
            ma10 = (np.append(prev_c.values[-9:], today[leader])).mean()
            if drop <= -Cfg.D1_LEADER_DROP and today[leader] < ma10:
                n += 1

    # D2：涨幅前5的个股中 ≥3 只当日跌 >3%
    top5 = mem.get('top5', [])
    hit = 0
    for c in top5:
        pc = closes[c].dropna()
        if len(pc) >= 1 and (today[c] / pc.values[-1] - 1) <= -Cfg.D2_DROP:
            hit += 1
    if hit >= Cfg.D2_COUNT:
        n += 1

    # D3：板块近5日日均成交额较前5日萎缩 >30%
    if sec.get('amt_shrink') is not None and sec['amt_shrink'] > Cfg.D3_SHRINK:
        n += 1

    # D4：板块指数连续2日收于MA5下方
    if sec.get('below_ma5_days', 0) >= Cfg.D4_BELOW_DAYS:
        n += 1

    return n


def _decide_one_sell(context, cd, code, sec, divergence):
    pos = context.portfolio.positions.get(code)
    if pos is None or pos.total_amount <= 0:
        g.holdings.pop(code, None)
        return
    h = g.holdings[code]
    cur = cd[code].last_price
    avg = _avg_cost(h)
    highest = h['highest']

    closes = _hist_close(code, context.previous_date, 12)
    if closes is None or len(closes) < 11:
        # 数据不足，只做基础止损
        if cur <= max(avg, highest) * (1 - Cfg.BASE_STOP):
            _queue_clear(code, '基础回撤止损')
        return
    ser = np.append(closes, cur)
    ma5_t, ma10_t = ser[-5:].mean(), ser[-10:].mean()
    ma5_y, ma10_y = closes[-5:].mean(), closes[-10:].mean()
    ret1 = cur / closes[-1] - 1

    # (2) 顶背离：full → 减至半仓（25%）；启动冷静期已在主流程处理
    if divergence and h['stage'] == 'full':
        _queue_reduce(code, '顶背离减半', source='divergence')
        return

    # (3) 个股破位
    #   G2：MA5 下穿 MA10（死叉）→ 清仓剩余
    if ma5_y > ma10_y and ma5_t < ma10_t:
        _queue_clear(code, 'G2:MA5死叉MA10')
        return
    #   G1：收盘跌破 MA10 → 卖50%
    if cur < ma10_t:
        _queue_reduce(code, 'G1:跌破MA10', source='probe')
        return
    #   G3：单日跌>7% 且放量>2倍 → 卖50%
    if ret1 <= -Cfg.G3_DROP:
        vols = _hist_field(code, None, context.previous_date, 'volume', count=6)
        vr = 1.0
        if vols is not None and len(vols) >= 6 and np.mean(vols[-6:-1]) > 0:
            vr = vols[-1] / np.mean(vols[-6:-1])
        if vr >= Cfg.G3_VOL_MULT:
            _queue_reduce(code, 'G3:单日跌%.1f%%放量' % (ret1 * 100), source='probe')
            return

    # (4) 特殊止损
    #   时间止损：买入满5个交易日仍未盈利
    if _bars_held(h, context.current_dt.date()) >= Cfg.TIME_STOP_BARS and cur < avg:
        _queue_clear(code, '时间止损(5日未盈利)')
        return
    #   加速止损：退潮期(撤退中)回撤 >10%
    retreat_active = g.sector_retreat_stage > 0 or h.get('retreat_stage', 0) > 0
    if retreat_active and cur <= max(avg, highest) * (1 - Cfg.ACCEL_STOP):
        _queue_clear(code, '加速止损(退潮期回撤10%)')
        return

    # (5) 基础止损：回撤 20%
    if cur <= max(avg, highest) * (1 - Cfg.BASE_STOP):
        _queue_clear(code, '基础回撤止损(20%)')
        return


def _queue_reduce(code, reason, source='probe'):
    """登记减半：目标股数在 _execute_pending 按次日开盘持仓量的一半计算。"""
    g.pending_sells.append({'code': code, 'action': 'reduce', 'reason': reason, 'source': source})
    log.info('[登记减半] %s（%s）' % (code, reason))


def _queue_clear(code, reason):
    g.pending_sells.append({'code': code, 'action': 'clear', 'reason': reason})
    log.info('[登记清仓] %s（%s）' % (code, reason))


# ==================================================================
# 七、买入 / 加仓引擎
# ==================================================================
def _rebalance(context, cd, members, sec, mem, idx_hist, idx_today, idx_ma5, idx_weak, phase):
    total = context.portfolio.total_value
    idx_above = idx_today > idx_ma5

    # ---- Step 1：持仓半仓(试探) → 全仓 升级（确认加仓）----
    if phase != '高潮':
        for code in list(g.holdings.keys()):
            h = g.holdings[code]
            if h['stage'] == 'half' and h.get('half_source') == 'probe' and not _pending_has(code):
                if _confirm_add(context, cd, code, sec, phase):
                    order_target_value(code, Cfg.FULL_RATIO * total)
                    p = cd[code].last_price
                    h['stage'] = 'full'
                    h['full_price'] = p
                    log.info('[确认加仓] %s 升级至50%% @%.3f' % (code, p))

    # ---- 买入前置门槛 ----
    if sec is None or mem is None:
        return
    if phase in ('高潮', '退潮', '冰点'):
        return                       # 高潮/退潮/冰点禁止新开仓
    if not idx_above:
        return                       # 上证需站上MA5
    if not sec['above_ma5']:
        return                       # 板块需站上MA5

    slots = Cfg.MAX_HOLDINGS - len(g.holdings)
    if slots <= 0:
        return

    # ---- 总仓位上限：加速期 ≤50%（高潮/退潮已禁开仓）----
    invested = context.portfolio.positions_value / total if total > 0 else 1.0
    exposure_cap = 0.50 if phase == '加速' else 1.0
    if invested >= exposure_cap:
        return

    # ---- Step 2：候选评分排序，取前 TOP_N ----
    cands = _rank_candidates(context, cd, members, mem, idx_hist, idx_today, idx_weak)
    if not cands:
        return
    if Cfg.DEBUG_SCORE:
        for c in cands[:5]:
            log.info('[评分] %s 类型=%s 总分=%.1f 趋势=%.1f 量能=%.1f 位置=%.1f 热点=%.1f 共振=%.1f'
                     % (c['code'], c['type'], c['score'], c['trend'], c['vol'],
                        c['pos'], c['hot'], c['reso']))
    cands = cands[:Cfg.TOP_N]

    # ---- Step 3：试探建仓（正常25% / 加速15%）----
    probe_ratio = Cfg.PROBE_ACCEL if phase == '加速' else Cfg.PROBE_NORMAL
    for c in cands:
        if slots <= 0:
            break
        code = c['code']
        if code in g.holdings or code in g.cooldown or _pending_has(code):
            continue
        order_target_value(code, probe_ratio * total)
        pos = context.portfolio.positions.get(code)
        if pos is None or pos.total_amount <= 0:
            log.warn('[试探买入未成交] %s（资金≈0 或涨停）' % code)
            continue
        p = cd[code].last_price
        g.holdings[code] = {
            'stage': 'half', 'half_source': 'probe',
            'half_price': p, 'full_price': None,
            'buy_date': context.current_dt.date(), 'buy_bar': g.bar,
            'highest': p, 'retreat_stage': 0,
        }
        log.info('[试探买入] %s @%.3f（%.0f%%，阶段=%s）' % (code, p, probe_ratio * 100, phase))
        slots -= 1


def _confirm_add(context, cd, code, sec, phase):
    """确认加仓条件：MA5>MA10 且 连续2日收盘>MA10 且 板块未跌破MA5；
    加速期额外要求 近3日无单日涨幅>7%。"""
    if sec is None or not sec['above_ma5']:
        return False
    closes = _hist_close(code, context.previous_date, 12)
    if closes is None or len(closes) < 11:
        return False
    cur = cd[code].last_price
    ser = np.append(closes, cur)
    ma5_t, ma10_t = ser[-5:].mean(), ser[-10:].mean()
    if not (ma5_t > ma10_t):
        return False
    # 连续2日(今日、昨日)收盘 > 各自 MA10
    ma10_today = ma10_t
    ma10_yest = closes[-10:].mean()
    if not (cur > ma10_today and closes[-1] > ma10_yest):
        return False
    if phase == '加速':
        rets = ser[1:] / ser[:-1] - 1
        if len(rets) >= 3 and np.nanmax(rets[-3:]) > Cfg.ACCEL_NO_SURGE:
            return False               # 近3日出现单日涨幅>7%的透支信号
    return True


# ==================================================================
# 八、五维评分体系
# ==================================================================
def _rank_candidates(context, cd, members, mem, idx_hist, idx_today, idx_weak):
    prev = context.previous_date
    # 硬过滤
    base = _prefilter(context, cd, members, mem, prev)
    base = [c for c in base if c not in g.holdings and c not in g.cooldown and not _pending_has(c)]
    idx_ret = _index_returns(idx_hist, idx_today)
    out = []
    for code in base:
        try:
            s = _score_stock(code, cd, mem, idx_ret, idx_weak)
        except Exception as e:
            log.warn('[评分异常] %s: %s' % (code, e))
            s = None
        if s is None or not s['eligible']:      # 未站上MA5 不入候选
            continue
        out.append(s)
    # 总分降序；同分按 热点强度 → 趋势 → 市值小
    out.sort(key=lambda s: (-s['score'], -s['hot'], -s['trend'], s['mktcap']))
    return out


def _prefilter(context, cd, members, mem, prev):
    """硬过滤：非ST/退市 + 上市≥60日 + 近5日日均换手率≥3% + 排除一字涨停。"""
    info = get_all_securities(['stock'], date=prev)
    cutoff = get_trade_days(end_date=prev, count=Cfg.MIN_LIST_DAYS)[0]
    turnover = _turnover5(members, prev)
    out = []
    for code in members:
        if code not in info.index:
            continue
        if info.loc[code, 'start_date'] > cutoff:
            continue
        c = cd[code]
        if c.is_st or c.paused:
            continue
        name = c.name if hasattr(c, 'name') else ''
        if 'ST' in name or '退' in name:
            continue
        if turnover.get(code, 0.0) < Cfg.TURNOVER_MIN:
            continue
        if Cfg.ONE_WORD_LIMIT_SKIP and _is_one_word_limit(code, cd):
            continue
        out.append(code)
    return out


def _score_stock(code, cd, mem, idx_ret, idx_weak):
    today = mem['today']
    if code not in today.index or code not in mem['closes'].columns:
        return None
    closes = mem['closes'][code].dropna().values
    if len(closes) < 10:
        return None
    cur = today[code]
    if cur is None or np.isnan(cur):
        return None
    ser = np.append(closes, cur)
    ma5 = ser[-5:].mean()
    ma10 = ser[-10:].mean()
    ma20 = ser[-20:].mean() if len(ser) >= 20 else ma10
    ma5_y = closes[-5:].mean()
    ma10_y = closes[-10:].mean()
    eligible = cur > ma5
    ret1 = cur / closes[-1] - 1

    # ---- 维度一：趋势强度（25）----
    if ma5 > ma10 > ma20:      s_arr = 15
    elif ma5 > ma10:           s_arr = 9
    else:                      s_arr = 3
    dev = cur / ma20 - 1 if ma20 else 0.0
    if dev < 0:                s_dev = 5
    elif dev <= 0.10:          s_dev = 10
    elif dev <= 0.15:          s_dev = 8
    elif dev <= 0.25:          s_dev = 4
    else:                      s_dev = 2       # 偏离>15%~25%及以上，追高风险
    trend = s_arr + s_dev

    # ---- 维度二：量能配合（20）----
    vpanel = mem.get('vols')
    vols = vpanel[code].dropna().values if (vpanel is not None and code in vpanel.columns) else None
    vr, vol_up = 1.0, False
    if vols is not None and len(vols) >= 6 and np.mean(vols[-6:-1]) > 0:
        vr = vols[-1] / np.mean(vols[-6:-1])
        vol_up = vols[-1] > vols[-2]
    if vr >= 1.5:    s_vr = 12
    elif vr >= 1.2:  s_vr = 8
    elif vr >= 1.0:  s_vr = 5
    else:            s_vr = 2
    if (ret1 > 0 and vol_up) or (ret1 <= 0 and not vol_up):
        s_pv = 8                                # 价涨量增 / 价跌量缩
    elif ret1 > 0 and not vol_up:
        s_pv = 2                                # 价涨量缩(背离)
    else:
        s_pv = 4                                # 价跌量增
    vol = s_vr + s_pv

    # ---- 维度三：位置安全（20）----
    p20 = mem['ret20_pct'].get(code, 0.5)
    if 0.30 <= p20 <= 0.70:     s_rank = 10     # 涨幅中位
    elif 0.15 <= p20 <= 0.85:   s_rank = 6
    else:                       s_rank = 3
    max20 = ser[-20:].max() if len(ser) >= 20 else ser.max()
    dd = 1 - cur / max20 if max20 else 0.0
    if dd < 0.03:               s_dd = 6        # 贴近最高(略透支)
    elif dd <= 0.15:            s_dd = 10       # 健康回撤
    elif dd <= 0.30:            s_dd = 6
    else:                       s_dd = 2
    pos = s_rank + s_dd

    # ---- 维度四：热点强度（20；科技/小市值 → 35）----
    ret5 = mem['ret5'].get(code, 0.0)
    is_top30 = ret5 >= mem['top30_thr']
    is_max = (code == mem.get('ret5_max'))
    p5 = mem['ret5_pct'].get(code, 0.5)
    if is_top30 and not is_max: s_hrank = 12    # 龙头溢价、非末班车
    elif is_max:                s_hrank = 8
    elif p5 >= 0.40:            s_hrank = 7
    else:                       s_hrank = 3
    lu = mem['limitups'].get(code, 0)
    if 1 <= lu <= 3:            s_lu = 5
    elif lu == 0:              s_lu = 2
    else:                       s_lu = 3        # ≥4 次，过热
    if 0 < ret5 <= 0.20:       s_c5 = 3
    else:                       s_c5 = 1
    hot = s_hrank + s_lu + s_c5

    # ---- 维度五：大盘共振（15；差异化）----
    stype = g.sector_meta.get(code, {}).get('type', 'mid')
    reso = 0.0
    if stype == 'tech':
        hot = hot * (35.0 / 20.0)               # 权重并入热点强度
        reso = 0.0
    elif stype == 'blue':
        corr = _corr10(ser, idx_ret)
        if corr > Cfg.CORR_HIGH:    reso = 15.0
        elif corr >= Cfg.CORR_LOW:  reso = 7.5
        else:                       reso = 0.0
    else:  # mid：宽松考核，仅大盘单边下跌时扣分
        reso = 0.0 if idx_weak else 15.0

    score = trend + vol + pos + hot + reso
    return {'code': code, 'eligible': eligible, 'score': float(score),
            'trend': float(trend), 'vol': float(vol), 'pos': float(pos),
            'hot': float(hot), 'reso': float(reso), 'type': stype,
            'mktcap': g.sector_meta.get(code, {}).get('mktcap', 1e6)}


def _corr10(ser, idx_ret):
    if len(ser) < 6 or idx_ret is None or len(idx_ret) < 5:
        return 0.0
    sret = ser[1:] / ser[:-1] - 1
    n = min(10, len(sret), len(idx_ret))
    a, b = sret[-n:], np.asarray(idx_ret[-n:], dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    try:
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return 0.0


# ==================================================================
# 九、板块快照 + 成分股统计
# ==================================================================
def _sector_snapshot(context, cd, members, prev):
    """返回 (sec, mem)。sec=板块级指标；mem=成分股级统计。"""
    closes = _hist_close_panel(members, prev, 25)     # index=time, cols=code（至昨日）
    if closes is None or closes.shape[0] < 6:
        return None, None
    closes = closes.dropna(axis=1, how='all')
    cols = list(closes.columns)
    if not cols:
        return None, None
    today = pd.Series({c: cd[c].last_price for c in cols}, dtype=float)

    # ---- 板块合成指数（成分股等权归一化均值）----
    first = closes.iloc[0]
    norm = closes.divide(first, axis=1)
    sect = norm.mean(axis=1)                          # 历史序列（至昨日）
    today_norm = (today / first).replace([np.inf, -np.inf], np.nan).mean()
    if np.isnan(today_norm) or sect.isna().all():
        return None, None

    level_today = float(today_norm)
    prev_level = float(sect.iloc[-1])
    ret1 = level_today / prev_level - 1 if prev_level else 0.0
    ret5 = level_today / float(sect.iloc[-5]) - 1 if len(sect) >= 5 else 0.0
    ma5 = (float(sect.iloc[-4:].sum()) + level_today) / 5.0
    above_ma5 = level_today > ma5
    ma20 = ((float(sect.iloc[-19:].sum()) + level_today) / 20.0) if len(sect) >= 19 else None

    # D4：连续多少日收于 MA5 下方（用历史 sect）
    ma5_hist = sect.rolling(Cfg.MA_SHORT).mean()
    below = (sect < ma5_hist)
    below_days = 0
    for v in below.values[::-1]:
        if bool(v):
            below_days += 1
        else:
            break

    # D3：近5日均额 vs 前5日均额
    amt_shrink = _amount_shrink(members, prev)

    sec = {'level_today': level_today, 'ret1': ret1, 'ret5': ret5,
           'ma5': ma5, 'above_ma5': above_ma5, 'ma20': ma20,
           'below_ma5_days': below_days, 'amt_shrink': amt_shrink}

    # ---- 成分股统计 ----
    ret5_map, ret10_map, ret20_map, lu_map = {}, {}, {}, {}
    for c in cols:
        cs = closes[c].dropna().values
        t = today[c]
        if len(cs) < 1 or np.isnan(t):
            continue
        ret5_map[c] = t / cs[-5] - 1 if len(cs) >= 5 else t / cs[0] - 1
        ret10_map[c] = t / cs[-10] - 1 if len(cs) >= 10 else t / cs[0] - 1
        ret20_map[c] = t / cs[-20] - 1 if len(cs) >= 20 else t / cs[0] - 1
        daily = cs[1:] / cs[:-1] - 1 if len(cs) >= 2 else np.array([])
        lu_map[c] = int(np.sum(daily[-5:] >= (_limit_up_pct(c) - 0.005))) if len(daily) else 0

    ret5_pct = _pctile_map(ret5_map)
    ret20_pct = _pctile_map(ret20_map)
    ret5_max = max(ret5_map, key=ret5_map.get) if ret5_map else None
    leader = max(ret10_map, key=ret10_map.get) if ret10_map else None
    top5 = sorted(ret5_map, key=ret5_map.get, reverse=True)[:5]
    top30_thr = np.percentile(list(ret5_map.values()), 70) if ret5_map else 0.0

    # 量能面板（量比用）
    vols = _hist_field_panel(members, prev, 'volume', 6)

    mem = {'closes': closes, 'today': today, 'vols': vols,
           'ret5': ret5_map, 'ret10': ret10_map, 'ret20': ret20_map,
           'ret5_pct': ret5_pct, 'ret20_pct': ret20_pct, 'limitups': lu_map,
           'ret5_max': ret5_max, 'leader': leader, 'top5': top5, 'top30_thr': float(top30_thr)}

    # 持仓跌出前 WARN_RANK 名预警
    _warn_rank(ret5_map)
    return sec, mem


def _warn_rank(ret5_map):
    if not ret5_map or not g.holdings:
        return
    ranked = sorted(ret5_map, key=ret5_map.get, reverse=True)
    topn = set(ranked[:Cfg.WARN_RANK])
    for code in g.holdings:
        if code in ret5_map and code not in topn:
            rank = ranked.index(code) + 1
            log.info('[持仓预警] %s 跌出板块前%d名(当前第%d)，进入重点关注' % (code, Cfg.WARN_RANK, rank))


def _amount_shrink(members, prev):
    """近5日均成交额 较 前5日 萎缩比例（0.30 = 萎缩30%）。"""
    money = _hist_field_panel(members, prev, 'money', 10)
    if money is None or money.shape[0] < 10:
        return None
    total = money.sum(axis=1)
    near5 = float(total.iloc[-5:].mean())
    prior5 = float(total.iloc[-10:-5].mean())
    if prior5 <= 0:
        return None
    return 1 - near5 / prior5


def _update_sector_meta(context, members, prev):
    """成分股：申万一级行业 + 总市值 → 分类 tech/blue/mid（每日刷新）。"""
    try:
        ind = get_industry(members, date=prev)
    except Exception:
        ind = {}
    mktcap = {}
    try:
        df = get_valuation(members, end_date=prev, count=1, fields=['market_cap'])
        if df is not None and len(df):
            for _, row in df.iterrows():
                mktcap[row['code']] = float(row['market_cap'])
    except Exception:
        pass
    meta = {}
    for code in members:
        name = ''
        try:
            name = ind.get(code, {}).get('sw_l1', {}).get('industry_name', '')
        except Exception:
            name = ''
        cap = mktcap.get(code, None)
        meta[code] = {'ind': name, 'mktcap': cap if cap is not None else 1e6,
                      'type': _classify(name, cap)}
    g.sector_meta = meta


def _classify(industry_name, mktcap):
    """tech(科技/小市值) / blue(蓝筹/大市值) / mid(中间)。"""
    if industry_name in Cfg.TECH_INDUSTRIES:
        return 'tech'
    if industry_name in Cfg.BLUE_INDUSTRIES:
        return 'blue'
    if mktcap is not None:
        if mktcap < Cfg.TECH_MKTCAP_MAX:
            return 'tech'
        if mktcap > Cfg.BLUE_MKTCAP_MIN:
            return 'blue'
    return 'mid'


def _phase(sec, retreat_stage):
    """板块生命周期：启动/扩散/加速/高潮/退潮/冰点。"""
    if sec is None:
        return '未知'
    if sec['ma20'] is not None and sec['level_today'] < sec['ma20'] and sec['ret5'] < 0:
        return '冰点'
    if retreat_stage > 0:
        return '退潮'
    if sec['ret5'] > Cfg.SECTOR_HIGH:
        return '高潮'
    if sec['ret5'] >= Cfg.SECTOR_ACCEL_LOW:
        return '加速'
    return '启动/扩散'


# ==================================================================
# 十、大盘顶背离四条件（沿用基础策略）
# ==================================================================
def _check_divergence(context, cd, idx_hist, idx_today):
    c1 = idx_today > idx_hist[-1]
    c2 = idx_today > idx_hist[-(Cfg.NEW_HIGH_DAYS - 1):].max()
    if not (c1 and c2):
        return False

    uni = _breadth_universe(context)
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

    result = c1 and c2 and c3 and c4
    if result:
        log.info('★ 顶背离触发：弱势广度=%.1f%% 累计下跌广度=%.1f%%' % (weak * 100, down * 100))
    return result


def _index_weak(idx_hist, idx_today):
    """上证是否连续 IDX_WEAK_DAYS 日收于 MA5 下方（大盘单边下跌）。"""
    if idx_hist is None or len(idx_hist) < Cfg.MA_SHORT + Cfg.IDX_WEAK_DAYS:
        return False
    ser = np.append(np.asarray(idx_hist, dtype=float), float(idx_today))
    ma5 = pd.Series(ser).rolling(Cfg.MA_SHORT).mean().values
    # 最近 IDX_WEAK_DAYS 个交易日(含今日)是否均收于各自 MA5 下方
    for k in range(1, Cfg.IDX_WEAK_DAYS + 1):
        if np.isnan(ma5[-k]) or ser[-k] >= ma5[-k]:
            return False
    return True


def _index_returns(idx_hist, idx_today):
    if idx_hist is None:
        return None
    ser = np.append(np.asarray(idx_hist, dtype=float), float(idx_today))
    return ser[1:] / ser[:-1] - 1


# ==================================================================
# 十一、辅助函数
# ==================================================================
def _reconcile(context, cd):
    """未跟踪的实际持仓纳入跟踪，确保止损覆盖。"""
    for code, pos in context.portfolio.positions.items():
        if pos.total_amount <= 0 or code in g.holdings or _pending_has(code):
            continue
        ac = pos.avg_cost if getattr(pos, 'avg_cost', 0) else cd[code].last_price
        g.holdings[code] = {
            'stage': 'full', 'half_source': 'probe',
            'half_price': ac, 'full_price': ac,
            'buy_date': context.previous_date, 'buy_bar': g.bar,
            'highest': max(ac, cd[code].last_price), 'retreat_stage': 0,
        }
        log.warn('[持仓校准] 未跟踪持仓 %s 纳入(成本 %.3f)' % (code, ac))


def _update_holdings_highest(context, cd):
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


def _any_holding_below_ma20(cd, codes):
    for code in codes:
        closes = _hist_close(code, None, 20)
        if closes is None or len(closes) < 20:
            continue
        ser = np.append(closes[-19:], cd[code].last_price)
        if cd[code].last_price < ser.mean():
            return True
    return False


def _avg_cost(h):
    if h.get('full_price') is not None:
        return (h['half_price'] + h['full_price']) / 2.0
    return h['half_price']


def _bars_held(h, cur_date):
    bd = h.get('buy_date')
    if not bd:
        return 0
    try:
        return len(get_trade_days(start_date=bd, end_date=cur_date)) - 1
    except Exception:
        return 0


def _limit_up_pct(code):
    """涨跌停幅度：科创/创业 20%，其余 10%（北交所暂按30%近似）。"""
    if code.startswith('688') or code.startswith('689'):
        return 0.20
    if code.startswith('300') or code.startswith('301'):
        return 0.20
    if code.startswith('8') or code.startswith('4'):
        return 0.30
    return 0.10


def _is_one_word_limit(code, cd):
    """一字涨停：今开=最高 且 涨停（无法买入）。"""
    c = cd[code]
    hl = getattr(c, 'high_limit', None)
    op = getattr(c, 'day_open', None)
    hi = getattr(c, 'high', None)
    lp = getattr(c, 'last_price', None)
    if hl is None or op is None:
        return False
    try:
        if op >= hl - 0.001 and lp >= hl - 0.001:
            if hi is None or hi <= op + 0.001:
                return True
    except Exception:
        return False
    return False


def _turnover5(members, prev):
    """近5日日均换手率(%)。"""
    out = {}
    df = None
    try:
        df = get_valuation(members, end_date=prev, count=5, fields=['turnover_ratio'])
    except Exception as e:
        log.warn('[换手率取值异常] %s' % e)
    if df is None or len(df) == 0:
        return {c: 999.0 for c in members}      # 取不到则不因换手率误杀
    for code in members:
        sub = df[df['code'] == code]['turnover_ratio'] if 'code' in df.columns else None
        if sub is not None and len(sub):
            out[code] = float(np.nanmean(sub.values))
        else:
            out[code] = 999.0
    return out


def _pctile_map(d):
    """把 {code:value} 映射为 {code: 分位数(0~1)}。"""
    if not d:
        return {}
    items = sorted(d.items(), key=lambda kv: kv[1])
    n = len(items)
    out = {}
    for i, (code, _) in enumerate(items):
        out[code] = (i + 0.5) / n if n else 0.5
    return out


def _breadth_universe(context):
    stocks = get_all_securities(['stock'], date=context.previous_date)
    stocks = stocks[(stocks.index.str.endswith('.XSHG') | stocks.index.str.endswith('.XSHE'))]
    cutoff = get_trade_days(end_date=context.previous_date, count=Cfg.MIN_LIST_DAYS)[0]
    stocks = stocks[stocks['start_date'] <= cutoff]
    return list(stocks.index)


def _hist_close(code, end_date, count):
    try:
        kw = {'count': count} if end_date is None else {'end_date': end_date, 'count': count}
        df = get_price(code, frequency='daily', fields=['close'],
                       skip_paused=False, fq='pre', panel=False, **kw)
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


def _hist_close_panel(codes, end_date, count, batch=800):
    return _hist_field_panel(codes, end_date, 'close', count, batch)


def _hist_field_panel(codes, end_date, field, count, batch=800):
    codes = list(codes)
    if not codes:
        return None
    frames = []
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        try:
            df = get_price(chunk, end_date=end_date, count=count, frequency='daily',
                           fields=[field], skip_paused=False, fq='pre', panel=False)
            if df is not None and len(df) > 0:
                frames.append(df)
        except Exception as e:
            log.warn('取%s面板分批失败(%d): %s' % (field, i, e))
    if not frames:
        return None
    try:
        df = pd.concat(frames, ignore_index=True)
        return df.pivot(index='time', columns='code', values=field).sort_index()
    except Exception as e:
        log.warn('%s面板拼接失败: %s' % (field, e))
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
