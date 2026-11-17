# -*- coding: utf-8 -*-
"""
动量策略 Adaptive Momentum（一般版）
========================================================
策略类型：横截面动量 / 行业轮动（月度调仓，中长期视角）
平台：聚宽 JoinQuant

对应《动量策略_Adaptive_Momentum.md》。双层架构 + 三层崩溃防护：
  第一层 行业动量：31 个申万一级行业，过去 J 个月(剔除最近 L 个月)夏普比率，取前 10 行业
  第二层 残差动量：候选行业成分股，日超额收益对 FF3(市场/规模/价值)做 OLS 回归，
                    以截距 α 的 t 统计量为残差动量，取前 20 只
  第三层 等权配置：20 只等权(各 5%)，每月最后一个交易日收盘再平衡

动量崩溃防护：
  4.1 波动率熔断：组合近 5 日年化波动率 > 40% → 减至 50%(保留残差前 10)；
                  回落 < 25% 且持续 5 日 → 恢复满仓
  4.2 相关性熔断：赢家(前 20)与输家(后 20)近 20 日收益相关性 > 0.85 → 暂停当月调仓；
                  回落 < 0.70 → 恢复调仓
  4.3 市场状态过滤：全市场指数 MA60 向下 且 近 5 日累计反弹 > 8% → 清仓等待；
                    MA60 走平/向上 → 恢复

执行口径：每交易日 14:45 做风控检查(熔断/崩溃)，月末交易日 14:50 做再平衡，均按当日收盘价成交。
FF3 因子在每次再平衡时由中证800 universe 自建(市场=全市场指数超额；SMB=市值分组；HML=B/M 分组)。
"""

import numpy as np
import pandas as pd
from jqdata import *


# ==================================================================
# 一、参数配置
# ==================================================================
class Cfg:
    BENCHMARK      = '000906.XSHG'     # 中证800（业绩基准 & 选股范围）
    UNIVERSE_INDEX = '000906.XSHG'     # 选股范围：中证800成分股
    MARKET_INDEX   = '000985.XSHG'     # 全市场指数（万得全A 代理：中证全指）

    # ---- 动量形成/持有窗口（交易日）----
    J_DAYS   = 126                     # 形成期 6 个月
    L_DAYS   = 21                      # 剔除期 1 个月（消除短期反转）
    HOLD_N   = 20                      # 持仓数量
    TOP_IND  = 10                      # 候选行业数
    WEIGHT   = 0.05                    # 单票权重（等权 5%）
    RF_ANNUAL = 0.02                   # 无风险年利率

    # ---- 4.1 波动率熔断 ----
    VOL_WIN          = 5               # 组合波动率回看
    VOL_HIGH         = 0.40            # 年化波动 > 40% 触发减仓
    VOL_LOW          = 0.25            # 回落 < 25%
    VOL_RECOVER_DAYS = 5               # 且持续 5 个交易日恢复
    VOL_REDUCE_N     = 10              # 减仓时保留残差前 10

    # ---- 4.2 相关性熔断 ----
    CORR_WIN  = 20                     # 赢家/输家相关性回看
    CORR_HIGH = 0.85                   # > 0.85 暂停调仓
    CORR_LOW  = 0.70                   # < 0.70 恢复调仓

    # ---- 4.3 市场状态过滤 ----
    MKT_MA          = 60               # 全市场 MA60
    MKT_SLOPE_WIN   = 5               # MA60 斜率判定窗口
    MKT_REBOUND_WIN = 5              # 反弹累计窗口
    MKT_REBOUND     = 0.08          # 近 5 日累计反弹 > 8%

    # ---- 因子分组 ----
    HML_QUANTILE = 0.30               # B/M 高/低各取 30%

    # ---- 调试 ----
    DEBUG = False


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

    g.last_ranking = []          # 上次再平衡的残差动量排序（code 从高到低）
    g.vol_reduced = False        # 是否处于波动率减仓状态
    g.low_vol_days = 0           # 低波动持续天数（用于恢复满仓）
    g.corr_suspended = False     # 是否处于相关性暂停调仓状态
    g.crash_cleared = False      # 是否处于市场崩溃清仓状态

    run_daily(risk_check,   time='14:45')     # 每日风控（熔断/崩溃）
    run_daily(rebalance,    time='14:50')     # 月末再平衡

    log.info('动量策略 Adaptive Momentum 初始化完成 [BUILD=momentum-v1] 范围=%s' % Cfg.UNIVERSE_INDEX)


# ==================================================================
# 三、每日风控（4.1 波动率熔断 + 4.3 市场状态过滤）
# ==================================================================
def risk_check(context):
    prev = context.previous_date

    # --- 4.3 市场状态过滤（最高优先级）---
    state = _market_state(prev)
    if state == 'crash':
        if not g.crash_cleared:
            _clear_all(context, '市场崩溃(MA60向下+急反弹>8%)')
            g.crash_cleared = True
            g.vol_reduced = False
            g.low_vol_days = 0
        return
    elif state == 'ok' and g.crash_cleared:
        g.crash_cleared = False
        log.info('[市场恢复] MA60 走平/向上，解除清仓，等待月末再平衡')

    if g.crash_cleared:
        return

    # --- 4.1 波动率熔断 ---
    holds = _current_holdings(context)
    if not holds:
        return
    vol = _portfolio_vol(context, holds)
    if vol is None:
        return

    if not g.vol_reduced:
        if vol > Cfg.VOL_HIGH:
            keep = [c for c in g.last_ranking if c in holds][:Cfg.VOL_REDUCE_N]
            if not keep:
                keep = list(holds)[:Cfg.VOL_REDUCE_N]
            _rebalance_to(context, keep, Cfg.WEIGHT)
            g.vol_reduced = True
            g.low_vol_days = 0
            log.info('[波动率熔断] 年化波动=%.1f%% > %.0f%%，减仓至前%d只(≈50%%)'
                     % (vol * 100, Cfg.VOL_HIGH * 100, len(keep)))
    else:
        if vol < Cfg.VOL_LOW:
            g.low_vol_days += 1
        else:
            g.low_vol_days = 0
        if g.low_vol_days >= Cfg.VOL_RECOVER_DAYS:
            top = [c for c in g.last_ranking][:Cfg.HOLD_N]
            if top:
                _rebalance_to(context, top, Cfg.WEIGHT)
            g.vol_reduced = False
            g.low_vol_days = 0
            log.info('[波动率恢复] 波动持续 %d 日 < %.0f%%，恢复满仓(%d只)'
                     % (Cfg.VOL_RECOVER_DAYS, Cfg.VOL_LOW * 100, len(top)))


# ==================================================================
# 四、月末再平衡（含 4.2 相关性熔断）
# ==================================================================
def rebalance(context):
    if not _is_month_end(context):
        return
    if g.crash_cleared:
        log.info('[再平衡跳过] 市场崩溃清仓状态，维持空仓')
        return

    ranked, tmap = _rank_stocks(context)
    if not ranked:
        log.warn('[再平衡跳过] 残差动量排序为空（数据不足）')
        return

    top20 = ranked[:Cfg.HOLD_N]
    bottom20 = ranked[-Cfg.HOLD_N:]

    # --- 4.2 相关性熔断 ---
    corr = _winner_loser_corr(context, top20, bottom20)
    if corr is not None:
        if not g.corr_suspended and corr > Cfg.CORR_HIGH:
            g.corr_suspended = True
            log.info('[相关性熔断] 赢家/输家相关性=%.2f > %.2f，暂停本月调仓' % (corr, Cfg.CORR_HIGH))
        elif g.corr_suspended and corr < Cfg.CORR_LOW:
            g.corr_suspended = False
            log.info('[相关性恢复] 相关性=%.2f < %.2f，恢复调仓' % (corr, Cfg.CORR_LOW))

    g.last_ranking = ranked        # 更新排序（供波动率减仓/恢复使用）

    if g.corr_suspended:
        log.info('[再平衡跳过] 相关性熔断中，维持现有仓位')
        return

    target = top20[:Cfg.VOL_REDUCE_N] if g.vol_reduced else top20
    _rebalance_to(context, target, Cfg.WEIGHT)
    log.info('[月末再平衡] 目标 %d 只等权(各%.0f%%)%s'
             % (len(target), Cfg.WEIGHT * 100, '（波动率减仓中）' if g.vol_reduced else ''))
    if Cfg.DEBUG:
        for c in target[:10]:
            log.info('   %s t(α)=%.2f' % (c, tmap.get(c, float('nan'))))


# ==================================================================
# 五、选股：行业夏普动量 + 个股残差动量
# ==================================================================
def _rank_stocks(context):
    """返回 (残差动量从高到低排序的 code 列表, {code: t(α)})。"""
    prev = context.previous_date
    universe = get_index_stocks(Cfg.UNIVERSE_INDEX, date=prev)
    universe = [c for c in universe if _tradable(c, prev)]
    if not universe:
        return [], {}

    # ---- 取形成期日线收盘（J+L 窗口）----
    close = _batch_close(universe, prev, Cfg.J_DAYS + Cfg.L_DAYS + 10)
    if close is None or close.shape[0] < Cfg.J_DAYS + Cfg.L_DAYS:
        log.warn('[选股] universe 收盘数据不足')
        return [], {}
    rets = close.pct_change()
    # 形成期：J 个交易日，结束于 L 个交易日之前
    form = rets.iloc[-(Cfg.J_DAYS + Cfg.L_DAYS):-Cfg.L_DAYS]
    if form.shape[0] < 60:
        return [], {}
    fdates = form.index

    # ---- 第一步：行业夏普动量，取前 TOP_IND 行业 ----
    # 用 universe 个股按申万一级行业分组、等权合成行业收益（绕开申万指数取价不可用问题）
    ind_map = _industry_map(universe, prev)
    if not ind_map:
        log.warn('[选股] 行业映射为空')
        return [], {}
    top_inds = _industry_sharpe(form, ind_map)
    if not top_inds:
        log.warn('[选股] 行业夏普计算为空')
        return [], {}

    cand = set()
    for ind in top_inds:
        for s in ind_map.get(ind, []):
            if s in form.columns:
                cand.add(s)
    cand = [c for c in cand if c in universe]
    if Cfg.DEBUG:
        log.info('[选股] 行业数=%d 前%d行业候选股=%d' % (len(ind_map), Cfg.TOP_IND, len(cand)))
    if not cand:
        return [], {}

    # ---- 第二步：构建 FF3 因子并对候选股做 OLS，取 α 的 t 值 ----
    factors = _build_ff3(context, universe, close, form, fdates)
    if factors is None:
        return [], {}
    rf_daily = Cfg.RF_ANNUAL / 252.0

    scores = {}
    Fmat_base = factors  # DataFrame: columns MKT/SMB/HML, index=fdates
    for code in cand:
        if code not in form.columns:
            continue
        y = form[code] - rf_daily
        df = pd.concat([y.rename('y'), Fmat_base], axis=1).dropna()
        if len(df) < 60:
            continue
        X = np.column_stack([np.ones(len(df)),
                             df['MKT'].values, df['SMB'].values, df['HML'].values])
        t_alpha = _ols_alpha_t(df['y'].values, X)
        if t_alpha is not None and np.isfinite(t_alpha):
            scores[code] = float(t_alpha)

    if not scores:
        return [], {}
    ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return ranked, scores


def _industry_map(codes, date, batch=500):
    """返回 {申万一级行业代码: [成分股...]}，仅含 universe 内个股。"""
    m = {}
    codes = list(codes)
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        try:
            d = get_industry(chunk, date=date)
        except Exception as e:
            log.warn('[取行业分类失败] %d: %s' % (i, e))
            continue
        for c in chunk:
            info = d.get(c, {})
            sw = info.get('sw_l1')
            if sw and sw.get('industry_code'):
                m.setdefault(sw['industry_code'], []).append(c)
    return m


def _industry_sharpe(form, ind_map):
    """按行业等权合成收益，算过去 J 个月(已剔除 L)夏普比率，返回前 TOP_IND 行业代码。"""
    rf_daily = Cfg.RF_ANNUAL / 252.0
    n = form.shape[0]
    rf_period = rf_daily * n
    sharpe = {}
    for ind, stocks in ind_map.items():
        cols = [s for s in stocks if s in form.columns]
        if len(cols) < 3:                    # 行业内至少 3 只，避免噪声
            continue
        r = form[cols].mean(axis=1).dropna()  # 行业等权日收益
        if len(r) < 60:
            continue
        cum = float(np.prod(1.0 + r.values) - 1.0)
        ann_std = float(r.std(ddof=0) * np.sqrt(252))
        if ann_std <= 0:
            continue
        sharpe[ind] = (cum - rf_period) / ann_std
    if not sharpe:
        return []
    ordered = sorted(sharpe.keys(), key=lambda i: sharpe[i], reverse=True)
    return ordered[:Cfg.TOP_IND]


def _build_ff3(context, universe, close, form, fdates):
    """自建 FF3 日频因子：MKT(全市场超额)、SMB(市值)、HML(B/M)。返回 DataFrame(index=fdates)。"""
    rf_daily = Cfg.RF_ANNUAL / 252.0

    # --- MKT：全市场指数日收益 - 无风险 ---
    mkt_px = _index_close(Cfg.MARKET_INDEX, fdates[-1], Cfg.J_DAYS + Cfg.L_DAYS + 10)
    if mkt_px is None:
        return None
    mkt = (mkt_px.pct_change().reindex(fdates) - rf_daily).rename('MKT')

    # --- 分组依据：形成期起点的市值与 PB ---
    start_date = fdates[0]
    if hasattr(start_date, 'date'):
        start_date = start_date.date()
    try:
        val = get_valuation(universe, end_date=start_date, count=1,
                            fields=['market_cap', 'pb_ratio'])
    except Exception as e:
        log.warn('[取估值失败] %s' % e)
        return None
    if val is None or len(val) == 0:
        return None
    val = val.dropna(subset=['market_cap', 'pb_ratio'])
    val = val[val['pb_ratio'] > 0]
    val = val.groupby('code').last()
    val['bm'] = 1.0 / val['pb_ratio']            # 账面市值比 = 1 / PB

    cols = [c for c in val.index if c in form.columns]
    if len(cols) < 20:
        return None
    mcap = val.loc[cols, 'market_cap']
    bm = val.loc[cols, 'bm']

    # --- SMB：小市值组 - 大市值组（按中位数二分）---
    med = mcap.median()
    small = [c for c in cols if mcap[c] <= med]
    big = [c for c in cols if mcap[c] > med]
    smb = (form[small].mean(axis=1) - form[big].mean(axis=1)).rename('SMB')

    # --- HML：高 B/M 组 - 低 B/M 组（各取 30%）---
    hi_thr = bm.quantile(1 - Cfg.HML_QUANTILE)
    lo_thr = bm.quantile(Cfg.HML_QUANTILE)
    high = [c for c in cols if bm[c] >= hi_thr]
    low = [c for c in cols if bm[c] <= lo_thr]
    hml = (form[high].mean(axis=1) - form[low].mean(axis=1)).rename('HML')

    factors = pd.concat([mkt, smb, hml], axis=1).reindex(fdates)
    return factors


def _ols_alpha_t(y, X):
    """OLS 回归，返回截距项 α 的 t 统计量。"""
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X.dot(beta)
        n, k = X.shape
        dof = n - k
        if dof <= 0:
            return None
        sigma2 = float(resid.dot(resid) / dof)
        xtx_inv = np.linalg.inv(X.T.dot(X))
        se_alpha = np.sqrt(sigma2 * xtx_inv[0, 0])
        if se_alpha <= 0:
            return None
        return beta[0] / se_alpha
    except Exception:
        return None


# ==================================================================
# 六、崩溃防护辅助
# ==================================================================
def _market_state(prev):
    """返回 'crash'(熊市急反弹) / 'ok'(MA60走平或向上) / 'neutral'。"""
    px = _index_close(Cfg.MARKET_INDEX, prev, Cfg.MKT_MA + Cfg.MKT_SLOPE_WIN + 5)
    if px is None or len(px) < Cfg.MKT_MA + Cfg.MKT_SLOPE_WIN:
        return 'neutral'
    ma = px.rolling(Cfg.MKT_MA).mean().dropna().values
    if len(ma) < Cfg.MKT_SLOPE_WIN + 1:
        return 'neutral'
    slope_down = ma[-1] < ma[-1 - Cfg.MKT_SLOPE_WIN]
    closes = px.values
    rebound = closes[-1] / closes[-1 - Cfg.MKT_REBOUND_WIN] - 1.0
    if slope_down and rebound > Cfg.MKT_REBOUND:
        return 'crash'
    if not slope_down:
        return 'ok'
    return 'neutral'


def _portfolio_vol(context, holds):
    """持仓组合近 VOL_WIN 日等权年化波动率。"""
    close = _batch_close(holds, context.previous_date, Cfg.VOL_WIN + 3)
    if close is None or close.shape[0] < Cfg.VOL_WIN + 1:
        return None
    rets = close.pct_change().dropna(how='all').tail(Cfg.VOL_WIN)
    if rets.shape[0] < 2:
        return None
    port = rets.mean(axis=1)                       # 等权组合日收益
    return float(port.std(ddof=0) * np.sqrt(252))


def _winner_loser_corr(context, winner, loser):
    """赢家(前20)与输家(后20)组合近 CORR_WIN 日收益相关性。"""
    both = list(set(winner) | set(loser))
    close = _batch_close(both, context.previous_date, Cfg.CORR_WIN + 3)
    if close is None or close.shape[0] < Cfg.CORR_WIN:
        return None
    rets = close.pct_change().tail(Cfg.CORR_WIN)
    w = [c for c in winner if c in rets.columns]
    l = [c for c in loser if c in rets.columns]
    if not w or not l:
        return None
    port_w = rets[w].mean(axis=1)
    port_l = rets[l].mean(axis=1)
    df = pd.concat([port_w, port_l], axis=1).dropna()
    if len(df) < 5:
        return None
    c = df.iloc[:, 0].corr(df.iloc[:, 1])
    return float(c) if pd.notna(c) else None


# ==================================================================
# 七、交易执行
# ==================================================================
def _rebalance_to(context, codes, weight):
    """把持仓调整为 codes 等权(各 weight)，其余清仓。当日收盘价成交。"""
    target = set(codes)
    for code in list(context.portfolio.positions.keys()):
        pos = context.portfolio.positions[code]
        if pos.total_amount > 0 and code not in target:
            order_target(code, 0)
    total = context.portfolio.total_value
    for code in codes:
        order_target_value(code, weight * total)


def _clear_all(context, reason):
    for code in list(context.portfolio.positions.keys()):
        if context.portfolio.positions[code].total_amount > 0:
            order_target(code, 0)
    log.info('[清仓] %s' % reason)


def _current_holdings(context):
    return [c for c, p in context.portfolio.positions.items() if p.total_amount > 0]


# ==================================================================
# 八、数据与通用辅助
# ==================================================================
def _tradable(code, date):
    """非停牌、非ST、上市满形成期。"""
    cd = get_current_data()
    c = cd[code]
    if c.paused or c.is_st:
        return False
    name = getattr(c, 'name', '') or ''
    if 'ST' in name or '退' in name:
        return False
    return True


def _batch_close(codes, end_date, count, batch=500):
    """批量取收盘，返回 DataFrame(index=time, columns=code)。"""
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
            log.warn('[批量取收盘失败] %d: %s' % (i, e))
    if not frames:
        return None
    try:
        alldf = pd.concat(frames, ignore_index=True)
        return alldf.pivot(index='time', columns='code', values='close').sort_index()
    except Exception as e:
        log.warn('[收盘面板拼接失败] %s' % e)
        return None


def _index_close(code, end_date, count):
    """取单一指数/证券收盘 Series（index=time）。"""
    try:
        df = get_price(code, end_date=end_date, count=count, frequency='daily',
                       fields=['close'], skip_paused=False, fq='pre', panel=False)
        if df is None or len(df) == 0:
            return None
        s = df['close']
        s.index = df.index
        return s
    except Exception as e:
        if Cfg.DEBUG:
            log.warn('[取指数收盘失败] %s: %s' % (code, e))
        return None


def _is_month_end(context):
    """今日是否为本月最后一个交易日。"""
    today = context.current_dt.date()
    try:
        days = get_trade_days(start_date=today, count=2)
        if len(days) >= 2:
            d0 = days[0].date() if hasattr(days[0], 'date') else days[0]
            d1 = days[1].date() if hasattr(days[1], 'date') else days[1]
            return d0 == today and d1.month != today.month
    except Exception:
        pass
    return False
