# -*- coding: utf-8 -*-
"""
全行情切换组合 V1.3 诊断 · 仅板块块（ETF）—— 聚宽回测代码（验证板块块用，不是交易策略）

规则：中小盘走强（中证1000 近 20 日净涨幅 ≥ 来回波动的 20% 且高于 20 日均价）并且大盘在上涨期
      （沪深300 高于半年均价、半年均价上行、一半以上申万一级行业站上 60 日均价）时，100% 资金给板块块；否则空仓（货币 ETF）。
板块块：申万二级行业指数（成分股按流通市值合成）里，走势顺（ER20>0.2 且 >20 日均价）、60 日涨幅前 3 的行业，
        各买一只近 120 日日收益相关系数最高（≥0.85）的行业 / 主题 ETF；每 10 个交易日重排，行业走弱当天卖。
其余代码与「全行情切换组合 V1.3」完全相同。

本地对照（2023-01-03 至 2026-11-13，单边 0.08%）：年化 +9.4%、最大回撤 -20.5%；行业用前一日收盘 +8.5%
    逐年 2023 +2%、2024 +1%、2025 +16%、2026 +17%
看日志：「行业→ETF 匹配」一行显示找到对应 ETF 的行业数（本地约 51–67 个）；「板块块持有」一行显示选中的行业和 ETF。
聚宽设置：2023-01-01 至 2026-11-15，¥1,000,000，分钟频率
"""
import builtins
from jqdata import *
import numpy as np
import pandas as pd
sum, min, max, abs, any, all, round = (builtins.sum, builtins.min, builtins.max, builtins.abs,
                                       builtins.any, builtins.all, builtins.round)

VERSION = "全行情切换组合 V1.3 诊断·仅板块块（ETF）"
MODE = "板块+稳健"   # 只用于决定是否计算大盘上涨期


class Cfg:
    TRADE_TIME = "14:50"
    CASH_ETF = "511880.XSHG"     # 货币 ETF（银华日利）
    CASH_BUFFER = 0.01
    TOL = 0.25                   # 已持有股票偏离目标市值超过 25% 才调整
    # ---- 开关 ----
    GATE_INDEX = "000852.XSHG"
    REG_INDEX = "000300.XSHG"
    ER_N, ER_MIN, MA_N = 20, 0.20, 20
    REG_MA, REG_SLOPE, BR_MA, BR_MIN = 120, 20, 60, 0.50
    # ---- 稳健块（V1.1）----
    A_POOL = 800; A_LIST_DAYS = 365; A_ROE_MIN = 0.05; A_HIGH_N = 250; A_VOL_N = 60
    A_HOLD = 10; A_BUFFER = 20; A_REBAL = 5; A_IND_MAX = 3; A_STOP = 0.10
    A_VOL_TARGET = 0.15; A_VOL_WIN = 40; A_EXPO_BAND = 0.10; A_DD_HALF = 0.15; A_DD_OUT = 0.25; A_DD_LOOK = 250
    # ---- 个股块 ----
    S_LIST_DAYS = 365; S_AMT_N = 20; S_AMT_DROP = 0.30; S_REV_N = 20; S_HOLD = 10; S_HOLD_MIX = 5; S_BUFFER = 20; S_REBAL = 5
    # ---- 板块块 ----
    B_MOM_N = 60; B_IND_N = 3; B_REBAL = 10; B_PER_IND = 1; B_PER_IND_MIX = 1
    B_CORR_N = 120; B_CORR_MIN = 0.85; B_MAP_DAYS = 20; B_ETF_DAYS = 120; B_ETF_MONEY = 3e7   # 行业→ETF 匹配


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type="stock")
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.00025, close_commission=0.00025,
                             min_commission=5), type="fund")
    set_slippage(PriceRelatedSlippage(0.002), type="stock")
    set_slippage(PriceRelatedSlippage(0.0005), type="fund")
    log.set_level("order", "error")
    g.day = 0
    # 稳健块
    g.a_picks = []; g.a_sleeve = []; g.a_expo = 0.0; g.a_tv = {}
    # 个股块
    g.s_on = False; g.s_picks = []
    # 板块块
    g.b_on = False; g.b_inds = []; g.b_stocks = {}
    g.last_alloc = None
    g.l2_mem = {}; g.l2_mem_day = -999; g.l2_cache = None; g.l2_cache_day = None   # 二级行业成分股 / 合成指数缓存
    g.b_map = {}; g.b_map_day = -999                                                   # 行业→ETF 匹配
    run_daily(trade, time=Cfg.TRADE_TIME)
    run_daily(after_close, time="after_close")
    log.info(VERSION)


# ================= 通用工具 =================
def can_buy(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price < d.high_limit - 1e-6


def can_sell(cd, s):
    d = cd[s]
    return (not d.paused) and d.last_price > d.low_limit + 1e-6


def place(context, cd, s, target_value):
    """带保护价的市价单 + 按手取整。target_value = 0 表示清仓。"""
    d = cd[s]
    price = d.last_price
    if price is None or price <= 0 or d.paused:
        return
    pos = context.portfolio.positions
    cur = pos[s].total_amount if s in pos else 0
    if target_value <= 0:
        if cur > 0 and can_sell(cd, s):
            order_target(s, 0, MarketOrderStyle(max(price * 0.98, d.low_limit)))
        return
    tgt = int(target_value / price / 100) * 100
    if s.startswith("688") and cur == 0 and tgt < 200:
        return
    delta = tgt - cur
    if delta >= 100 and can_buy(cd, s):
        lim = min(price * 1.02, d.high_limit)
        afford = int(context.portfolio.available_cash * 0.995 / (lim * 1.0003) / 100) * 100   # 按可用资金封顶（含佣金余量）
        delta = min(delta, afford)
        if s.startswith("688") and cur == 0 and delta < 200:
            return
        if delta >= 100:
            order(s, delta, MarketOrderStyle(lim))
    elif delta <= -100 and can_sell(cd, s):
        sell = min(-delta, pos[s].closeable_amount)
        sell = sell if sell == cur else int(sell / 100) * 100
        if sell > 0:
            order(s, -sell, MarketOrderStyle(max(price * 0.98, d.low_limit)))


def er_ma_ok(s):
    s = np.asarray(s, dtype=float)
    if len(s) < max(Cfg.ER_N + 1, Cfg.MA_N) or np.isnan(s[-(Cfg.ER_N + 1):]).any():
        return False
    path = np.abs(np.diff(s[-(Cfg.ER_N + 1):])).sum()
    er = (s[-1] - s[-(Cfg.ER_N + 1)]) / path if path > 0 else 0.0
    return er > Cfg.ER_MIN and s[-1] > s[-Cfg.MA_N:].mean()


def sw_close(codes, day, cal_days):
    """申万指数日收盘（截至 day）DataFrame[日期 × 代码]，分批查询"""
    start = day - pd.Timedelta(days=cal_days)
    out = []
    for i in range(0, len(codes), 25):
        q = query(finance.SW1_DAILY_PRICE.date, finance.SW1_DAILY_PRICE.code, finance.SW1_DAILY_PRICE.close
                  ).filter(finance.SW1_DAILY_PRICE.code.in_(codes[i:i + 25]),
                           finance.SW1_DAILY_PRICE.date >= start, finance.SW1_DAILY_PRICE.date <= day)
        out.append(finance.run_query(q))
    df = pd.concat(out) if out else pd.DataFrame()
    if len(df) == 0:
        return pd.DataFrame()
    return df.pivot_table(index="date", columns="code", values="close").sort_index()


def base_stocks(context, cd, list_days):
    day = context.previous_date
    secs = get_all_securities(["stock"], date=day)
    secs = secs[secs.start_date <= day - pd.Timedelta(days=list_days)]
    ss = [s for s in secs.index if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
    return [s for s in ss if (not cd[s].is_st) and (not cd[s].paused) and "退" not in cd[s].name]


def sell_cash_etf(context, cd, value):
    """卖出约 value 元的货币 ETF：按 100 份向上取整；卖完剩余不足 100 份就全部卖出"""
    e = Cfg.CASH_ETF
    pos = context.portfolio.positions
    if e not in pos or pos[e].closeable_amount <= 0 or cd[e].paused:
        return
    price = cd[e].last_price
    if price is None or price <= 0:
        return
    have = pos[e].closeable_amount
    n = int(np.ceil(value / price / 100.0)) * 100
    if have - n < 100:
        n = have
    if n > 0:
        order(e, -min(n, have))


# ================= 开关 =================
def gate_1000(context):
    n = max(Cfg.ER_N + 1, Cfg.MA_N)
    h = attribute_history(Cfg.GATE_INDEX, n, "1d", ["close"], skip_paused=True, df=False)["close"]
    b = get_price(Cfg.GATE_INDEX, end_date=context.current_dt, frequency="1m", fields=["close"], count=1)
    if len(h) < n or len(b) == 0:
        return False
    return er_ma_ok(np.append(h, float(b["close"].iloc[-1])))


def uptrend(context):
    day = context.previous_date
    c = attribute_history(Cfg.REG_INDEX, Cfg.REG_MA + Cfg.REG_SLOPE, "1d", ["close"], skip_paused=True, df=False)["close"]
    if len(c) < Cfg.REG_MA + Cfg.REG_SLOPE:
        return False
    ma_now = c[-Cfg.REG_MA:].mean(); ma_old = c[-(Cfg.REG_MA + Cfg.REG_SLOPE):-Cfg.REG_SLOPE].mean()
    if not (c[-1] > ma_now and ma_now > ma_old):
        return False
    px = sw_close(list(get_industries("sw_l1", date=day).index), day, 130)
    px = px.loc[:, px.notna().sum() >= Cfg.BR_MA] if len(px) else px
    if px.shape[1] == 0:
        return False
    return (px.iloc[-1] > px.iloc[-Cfg.BR_MA:].mean()).mean() >= Cfg.BR_MIN


def allocation(g1, g2):
    """诊断版：只让板块块工作，其余时间空仓（资金放货币 ETF）"""
    return (0.0, 0.0, 1.0) if g2 else (0.0, 0.0, 0.0)
    if MODE == "个股+稳健":
        return (0.0, 1.0, 0.0) if g1 else (1.0, 0.0, 0.0)
    if MODE == "板块+稳健":
        return (0.0, 0.0, 1.0) if g2 else (1.0, 0.0, 0.0)
    if g2:
        return (0.0, 0.5, 0.5)
    return (0.0, 1.0, 0.0) if g1 else (1.0, 0.0, 0.0)


# ================= 稳健块（V1.1 规则）=================
def a_exposure():
    r = np.array(g.a_sleeve[-Cfg.A_VOL_WIN:])
    if len(r) < 20:
        e = 1.0
    else:
        vol = np.std(r, ddof=1) * np.sqrt(252)
        e = 1.0 if vol <= 0 else min(1.0, Cfg.A_VOL_TARGET / vol)
    if len(g.a_sleeve) > 0:
        nav = np.cumprod(1 + np.array(g.a_sleeve)); dd = nav[-1] / np.max(nav[-Cfg.A_DD_LOOK:]) - 1
        if dd < -Cfg.A_DD_OUT:
            e = 0.0
        elif dd < -Cfg.A_DD_HALF:
            e *= 0.5
    return e


def a_select(context, cd):
    day = context.previous_date
    stocks = base_stocks(context, cd, Cfg.A_LIST_DAYS)
    q = query(valuation.code, valuation.circulating_market_cap, indicator.roe, income.net_profit).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date=day).dropna(subset=["circulating_market_cap"])
    df = df.sort_values("circulating_market_cap", ascending=False).head(Cfg.A_POOL)
    pool = list(df[(df.net_profit > 0) & (df.roe * 4 / 100 > Cfg.A_ROE_MIN)].code)
    if len(pool) < Cfg.A_HOLD:
        return g.a_picks
    px = get_price(pool, end_date=day, count=Cfg.A_HIGH_N - 1, fields=["close"], fq="pre", panel=False
                   ).pivot(index="time", columns="code", values="close")
    cur = pd.Series({s: cd[s].last_price for s in px.columns})
    px = pd.concat([px, cur.to_frame().T]); px = px.loc[:, px.notna().sum() >= 200]
    sc = ((px.iloc[-1] / px.max()).rank(pct=True) + (-px.pct_change().iloc[-Cfg.A_VOL_N:].std()).rank(pct=True)).dropna()
    codes = list(sc.sort_values(ascending=False).index); rank = {s: i for i, s in enumerate(codes)}
    ind = get_industry(codes + [s for s in g.a_picks if s not in rank], date=day)
    def ind_of(s): return ind.get(s, {}).get("sw_l2", {}).get("industry_name", "未知")
    cnt, pick = {}, []
    def add(s): pick.append(s); cnt[ind_of(s)] = cnt.get(ind_of(s), 0) + 1
    for s in sorted(g.a_picks, key=lambda x: rank.get(x, 10 ** 9)):
        if rank.get(s, 10 ** 9) < Cfg.A_BUFFER and len(pick) < Cfg.A_HOLD and cnt.get(ind_of(s), 0) < Cfg.A_IND_MAX:
            add(s)
    for s in codes:
        if len(pick) >= Cfg.A_HOLD:
            break
        if s not in pick and cnt.get(ind_of(s), 0) < Cfg.A_IND_MAX:
            add(s)
    return pick


def a_update(context, cd, share):
    """维护稳健块：止损、调仓日选股、总仓位；返回该块每只股票的目标市值"""
    pos = context.portfolio.positions
    # 单只止损（只对稳健块持有、且不在上行块目标里的股票）
    up = set(g.s_picks) | set(s for v in g.b_stocks.values() for s in v)
    for s in list(g.a_picks):
        if s in pos and pos[s].total_amount > 0 and s not in up and pos[s].avg_cost > 0 \
                and cd[s].last_price / pos[s].avg_cost - 1 < -Cfg.A_STOP and can_sell(cd, s):
            g.a_picks.remove(s); g.a_tv.pop(s, None)
            log.info("稳健块止损 %s %s" % (s, cd[s].name))
    expo = a_exposure()
    rebal = g.day % Cfg.A_REBAL == 0
    if rebal:
        g.a_picks = a_select(context, cd)          # 不持仓时也照常选股，影子净值连续
    base = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) * share
    if rebal or abs(expo - g.a_expo) >= Cfg.A_EXPO_BAND or (expo == 0 and g.a_expo > 0) \
            or share != getattr(g, "a_share", None):
        g.a_expo = expo; g.a_share = share
        tv = base * expo / Cfg.A_HOLD
        g.a_tv = {s: tv for s in g.a_picks} if share > 0 and expo > 0 else {}
    return dict(g.a_tv)


# ================= 个股块 =================
def s_select(context, cd, n):
    day = context.previous_date
    stocks = base_stocks(context, cd, Cfg.S_LIST_DAYS)
    amt = get_price(stocks, end_date=day, count=Cfg.S_AMT_N, fields=["money"], panel=False
                    ).pivot(index="time", columns="code", values="money").mean()
    amt = amt[amt > 0]; pool = list(amt[amt.rank(pct=True) > Cfg.S_AMT_DROP].index)
    cap = get_fundamentals(query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(pool)),
                           date=day).dropna().set_index("code")["circulating_market_cap"]
    cap = cap[cap > 0]
    px = get_price(list(cap.index), end_date=day, count=Cfg.S_REV_N, fields=["close"], fq="pre", panel=False
                   ).pivot(index="time", columns="code", values="close")
    cur = pd.Series({s: cd[s].last_price for s in px.columns})
    ret = (cur / px.iloc[0] - 1).replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({"cap": cap, "ret": ret}).dropna()
    sc = ((-np.log(df["cap"])).rank(pct=True) + (-df["ret"]).rank(pct=True)).sort_values(ascending=False)
    codes = list(sc.index); rank = {s: i for i, s in enumerate(codes)}
    pick = [s for s in sorted(g.s_picks, key=lambda x: rank.get(x, 10 ** 9)) if rank.get(s, 10 ** 9) < Cfg.S_BUFFER][:n]
    for s in codes:
        if len(pick) >= n:
            break
        if s not in pick:
            pick.append(s)
    return pick


def s_update(context, cd, share, n):
    if share <= 0:
        g.s_on = False; g.s_picks = []
        return {}
    if (not g.s_on) or g.day % Cfg.S_REBAL == 0 or not g.s_picks or len(g.s_picks) != n:
        g.s_picks = s_select(context, cd, n)
    g.s_on = True
    tv = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) * share / n
    return {s: tv for s in g.s_picks}


# ================= 板块块（V1.3：申万二级行业指数挑行业，买走势最像它的 ETF）=================
ETF_EXCLUDE = ["300", "500", "1000", "2000", "50", "180", "100", "800", "200", "A股", "A50", "A100", "A500", "中证A", "创业", "科创50",
               "科创100", "科创200", "科创板", "科综", "双创", "深证", "上证", "深成", "综指", "中小", "国债", "债", "货币", "现金", "日利",
               "添益", "快线", "快钱", "财富宝", "添利", "天天", "日日", "短融", "黄金", "金ETF", "白银", "豆粕", "原油", "能源化工", "期货",
               "纳指", "纳斯达克", "标普", "日经", "日本", "德国", "法国", "恒生", "港股", "中概", "H股", "沪港深", "香港", "美国", "亚太",
               "东南亚", "沙特", "巴西", "MSCI", "道琼斯", "海外", "全球", "中韩", "AH", "等权", "央企", "国企", "民企", "ESG", "红利", "价值",
               "成长", "质量", "低波", "基本面", "动量", "现金流", "股息", "大盘", "中盘", "小盘", "微盘", "龙头", "核心", "宽基", "长三角",
               "大湾区", "粤港澳", "湾创", "湖北", "浙江", "浙商", "上海", "北京", "四川", "江苏", "福建", "山东", "安徽", "湖南", "广东",
               "京津冀", "成渝", "杭州湾", "张江", "G60", "长江保护", "中创400", "一带一路", "央视", "中关村", "新兴", "战略", "产业升级",
               "新经济", "高端装备", "科技100", "科技50", "TMT", "互联网+"]


def l2_close(context, codes, need):
    """申万二级行业指数日收盘（截至前一交易日，need 个交易日）。
    聚宽 finance.SW1_DAILY_PRICE 查不到二级行业，用成分股按流通市值加权的日收益合成；成分股每 20 个交易日更新一次。"""
    day = context.previous_date
    if g.l2_cache_day == day and g.l2_cache is not None and len(g.l2_cache) >= need:
        return g.l2_cache
    if g.day - g.l2_mem_day >= 20 or not g.l2_mem:
        g.l2_mem = {c: [s for s in get_industry_stocks(c, date=day) if not s.startswith(("4", "8", "92")) and not s.endswith(".BJ")]
                    for c in codes}
        g.l2_mem_day = g.day
    allst = sorted(set(s for v in g.l2_mem.values() for s in v))
    if not allst:
        return pd.DataFrame()
    p = get_price(allst, end_date=day, count=need + 1, fields=["close"], fq="pre", panel=False
                  ).pivot(index="time", columns="code", values="close")
    r = p.pct_change().iloc[1:]
    cap = get_fundamentals(query(valuation.code, valuation.circulating_market_cap).filter(valuation.code.in_(allst)),
                           date=day).dropna().set_index("code")["circulating_market_cap"]
    syn = {}
    for c, ss in g.l2_mem.items():
        ss = [s for s in ss if s in r.columns and s in cap.index]
        if len(ss) < 3:
            continue
        rr = r[ss]
        ww = rr.notna().mul(cap[ss], axis=1)
        ret = (rr.fillna(0) * ww).sum(axis=1) / ww.sum(axis=1).replace(0, np.nan)
        syn[c] = (1 + ret.fillna(0)).cumprod()
    out = pd.DataFrame(syn)
    out.index = pd.to_datetime(out.index)
    g.l2_cache, g.l2_cache_day = out, day
    return out


def etf_pool(context):
    """行业 / 主题 ETF：名称不含宽基、风格、跨境、债券、货币、商品等关键字；上市满 120 天；近 20 日日均成交额 ≥ 3000 万"""
    day = context.previous_date
    df = get_all_securities(["etf"], date=day)
    df = df[df.start_date <= day - pd.Timedelta(days=Cfg.B_ETF_DAYS)]
    cand = [c for c, n in zip(df.index, df.display_name) if not any(k in n for k in ETF_EXCLUDE)]
    if not cand:
        return []
    m = get_price(cand, end_date=day, count=20, fields=["money"], panel=False).pivot(index="time", columns="code", values="money").mean()
    return list(m[m >= Cfg.B_ETF_MONEY].index)


def b_mapping(context, ind_px):
    """每个二级行业找近 120 日日收益相关系数最高、且 ≥ 0.85 的 ETF；返回 {行业: (ETF, 相关系数)}"""
    day = context.previous_date
    etfs = etf_pool(context)
    if not etfs or len(ind_px) < Cfg.B_CORR_N + 1:
        return {}
    ep = get_price(etfs, end_date=day, count=Cfg.B_CORR_N + 1, fields=["close"], panel=False
                   ).pivot(index="time", columns="code", values="close")
    er = ep.pct_change().iloc[1:]
    er.index = pd.to_datetime(er.index)
    er = er.loc[:, er.notna().sum() >= Cfg.B_CORR_N * 0.8]
    ir = ind_px.pct_change().iloc[-Cfg.B_CORR_N:]
    ir, er = ir.align(er, join="inner", axis=0)
    if len(ir) < Cfg.B_CORR_N * 0.8 or er.shape[1] == 0:
        return {}
    out = {}
    for c in ir.columns:
        cc = er.corrwith(ir[c]).dropna()
        if len(cc) and cc.max() >= Cfg.B_CORR_MIN:
            out[c] = (cc.idxmax(), float(cc.max()))
    log.info("行业→ETF 匹配：%d 个二级行业找到相关系数 ≥ %.2f 的 ETF（可选 ETF %d 只）" % (len(out), Cfg.B_CORR_MIN, er.shape[1]))
    return out


def b_update(context, cd, share, k=None):
    """返回板块块每只 ETF 的目标市值；g.b_stocks = {行业代码: [ETF]}（供其它块识别）"""
    if share <= 0:
        g.b_on = False; g.b_inds = []; g.b_stocks = {}
        return {}
    day = context.previous_date
    inds = get_industries("sw_l2", date=day)
    inds = inds[pd.to_datetime(inds.start_date) <= pd.Timestamp(day)]   # 统一成时间戳比较，避免 pandas 警告刷屏
    need = max(Cfg.B_CORR_N, Cfg.B_MOM_N + Cfg.ER_N) + 5
    px = l2_close(context, list(inds.index), need)
    if px is None or len(px) < Cfg.B_MOM_N + 1:
        log.warn("二级行业行情不足，板块块今天不操作")
        return {}
    ok, mom = set(), {}
    for c in px.columns:
        s = px[c].dropna()
        if len(s) >= Cfg.B_MOM_N + 1 and er_ma_ok(s.values):
            ok.add(c)
            mom[c] = s.iloc[-1] / s.iloc[-(Cfg.B_MOM_N + 1)] - 1
    def nm(c):
        return inds.loc[c, "name"] if c in inds.index else c
    for ind in list(g.b_inds):                       # 行业自己走弱，当天卖掉它的 ETF
        if ind not in ok:
            g.b_inds.remove(ind); g.b_stocks.pop(ind, None)
            log.info("板块块：%s 走弱，卖出对应 ETF" % nm(ind))
    if (not g.b_on) or g.day % Cfg.B_REBAL == 0:
        if g.day - g.b_map_day >= Cfg.B_MAP_DAYS or not g.b_map:
            g.b_map = b_mapping(context, px)
            g.b_map_day = g.day
        order_ = sorted(mom, key=lambda c: -mom[c])
        pick, used = [], set()
        for c in order_:
            if len(pick) >= Cfg.B_IND_N:
                break
            if c in g.b_map and g.b_map[c][0] not in used:
                pick.append(c); used.add(g.b_map[c][0])
        g.b_inds = pick
        g.b_stocks = {c: [g.b_map[c][0]] for c in pick}
        log.info("板块块持有：%s" % ("、".join("%s→%s" % (nm(c), get_security_info(g.b_map[c][0]).display_name) for c in pick) or "无"))
    g.b_on = True
    tv = context.portfolio.total_value * (1 - Cfg.CASH_BUFFER) * share / Cfg.B_IND_N
    out = {}
    for c in g.b_inds:
        for e in g.b_stocks.get(c, []):
            out[e] = out.get(e, 0.0) + tv
    return out


# ================= 主流程 =================
def trade(context):
    cd = get_current_data()
    pos = context.portfolio.positions
    g1 = gate_1000(context)
    g2 = g1 and MODE != "个股+稳健" and uptrend(context)
    a_sh, s_sh, b_sh = allocation(g1, g2)
    if (a_sh, s_sh, b_sh) != g.last_alloc:
        log.info("资金分配 → 稳健 %.0f%% · 个股 %.0f%% · 板块 %.0f%%（中证1000趋势=%s 上行期=%s）"
                 % (a_sh * 100, s_sh * 100, b_sh * 100, g1, g2))
        g.last_alloc = (a_sh, s_sh, b_sh)
    tgt = {}
    mix = s_sh > 0 and b_sh > 0                     # 个股、板块同时持有时各自减少只数，合计 ≤ 10
    n_s = Cfg.S_HOLD_MIX if mix else Cfg.S_HOLD
    k_b = Cfg.B_PER_IND_MIX if mix else Cfg.B_PER_IND
    for part in (s_update(context, cd, s_sh, n_s), b_update(context, cd, b_sh, k_b), a_update(context, cd, a_sh)):
        for s, v in part.items():
            tgt[s] = tgt.get(s, 0.0) + v
    # 1. 卖：不在目标里的清仓，超出目标太多的减仓
    for s in list(pos.keys()):
        if s == Cfg.CASH_ETF or pos[s].total_amount == 0:
            continue
        if s not in tgt:
            place(context, cd, s, 0)
        elif pos[s].value > tgt[s] * (1 + Cfg.TOL):
            place(context, cd, s, tgt[s])
    # 2. 需要买入时先卖货币 ETF 腾钱
    need = sum(max(0.0, v - (pos[s].value if s in pos else 0)) for s, v in tgt.items()
               if (pos[s].value if s in pos else 0) < v * (1 - Cfg.TOL))
    e = Cfg.CASH_ETF
    if need > 0 and e in pos and pos[e].total_amount > 0 and not cd[e].paused:
        pf = context.portfolio
        if need > pf.available_cash:
            sell_cash_etf(context, cd, (need - pf.available_cash) * 1.02)
    # 3. 买
    for s, v in tgt.items():
        cur = pos[s].value if s in pos else 0
        if cur < v * (1 - Cfg.TOL):
            place(context, cd, s, v)
    # 4. 剩余现金买货币 ETF
    pf = context.portfolio
    spare = pf.available_cash - pf.total_value * Cfg.CASH_BUFFER
    if spare > 20000 and not cd[e].paused:
        order_value(e, spare)
    record(稳健块=a_sh * 100, 个股块=s_sh * 100, 板块块=b_sh * 100)
    g.day += 1


def after_close(context):
    """稳健块影子净值：选股组合当日等权收益（不管当天是否实际持有）"""
    if not g.a_picks:
        g.a_sleeve.append(0.0); return
    px = get_price(g.a_picks, end_date=context.current_dt.date(), count=2, fields=["close"], fq="pre", panel=False
                   ).pivot(index="time", columns="code", values="close")
    if len(px) < 2:
        g.a_sleeve.append(0.0); return
    r = (px.iloc[-1] / px.iloc[-2] - 1).dropna()
    g.a_sleeve.append(float(r.mean()) if len(r) else 0.0)
    g.a_sleeve = g.a_sleeve[-400:]
