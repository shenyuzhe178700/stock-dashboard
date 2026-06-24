# -*- coding: utf-8 -*-
"""增强版个股分析数据生成 - 含短线/中线/长线预测 + K线引用"""
import sys, os, json
sys.path.insert(0, r"D:\quant_a_stock")
import pandas as pd, numpy as np
from config import OUTPUT_DIR, DATA_DIR

def safe(v, default=0, ndigits=4):
    try:
        x = float(v)
        return round(0 if pd.isna(x) else x, ndigits)
    except: return default

def compute_macd_signal(dif, dea):
    """MACD金叉死叉信号"""
    if dif > dea: return "金叉(看多)"
    return "死叉(看空)"

def compute_kdj_signal(k, d, j):
    """KDJ超买超卖"""
    if j < 0: return "严重超卖(强反弹)"
    if j < 20: return "超卖区(反弹机会)"
    if j > 100: return "严重超买(回调压力)"
    if j > 80: return "超买区(注意风险)"
    return "正常区间"

def compute_support_resistance(df):
    """计算支撑位和阻力位"""
    close = df["close"].values
    if len(close) < 20: return 0, 0
    recent_high = df["high"].tail(20).max()
    recent_low = df["low"].tail(20).min()
    return round(recent_low, 2), round(recent_high, 2)

def compute_volume_analysis(sd):
    """成交量分析"""
    if len(sd) < 20: return "数据不足"
    vol = sd["volume"].values
    avg_vol_20 = np.mean(vol[-20:])
    avg_vol_5 = np.mean(vol[-5:])
    ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else 1
    if ratio > 1.5: return "放量上攻"
    if ratio > 1.1: return "温和放量"
    if ratio < 0.5: return "极度缩量"
    if ratio < 0.7: return "缩量整理"
    return "量能平稳"

def compute_trend_strength(df):
    """趋势强度 (ADX简化)"""
    if len(df) < 14: return 0
    highs = df["high"].values[-14:]
    lows = df["low"].values[-14:]
    closes = df["close"].values[-14:]
    tr = np.maximum(highs[-1] - lows[-1], 
                    np.maximum(np.abs(highs[-1] - closes[-2] if len(closes)>1 else 0),
                               np.abs(lows[-1] - closes[-2] if len(closes)>1 else 0)))
    atr = np.mean([np.maximum(highs[i]-lows[i], 
                    np.maximum(np.abs(highs[i]-closes[i-1] if i>0 else closes[i]),
                               np.abs(lows[i]-closes[i-1] if i>0 else closes[i])))
                   for i in range(len(highs))])
    return round(atr / closes[-1] * 100, 2) if closes[-1] > 0 else 0

def compute_fibonacci(df):
    """斐波那契回调位"""
    if len(df) < 30: return {}
    high_30 = df["high"].tail(30).max()
    low_30 = df["low"].tail(30).min()
    diff = high_30 - low_30
    if diff <= 0: return {}
    return {
        "fib_0": round(low_30, 2),
        "fib_382": round(low_30 + diff * 0.382, 2),
        "fib_50": round(low_30 + diff * 0.5, 2),
        "fib_618": round(low_30 + diff * 0.618, 2),
        "fib_100": round(high_30, 2),
    }

def compute():
    from data.factors import FactorEngine
    
    # 尝试加载数据
    cache_files = [
        "daily_bs_300_20230101_20260601.pkl",
        "daily_bs_300_20251224_20260622.pkl",
    ]
    
    daily = None
    for fn in cache_files:
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            print(f"Load: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")
            daily = pd.read_pickle(path)
            break
    
    if daily is None:
        print("ERROR: No data cache found")
        return
    
    codes = sorted(daily["code"].unique())
    engine = FactorEngine(daily)
    f = engine.compute_all_factors()
    
    f["date_dt"] = pd.to_datetime(f["date"])
    latest = f["date_dt"].max()
    cutoff = latest - pd.Timedelta(days=120)
    recent = f[f["date_dt"] >= cutoff]
    
    # 名字映射
    name_map = {}
    name_path = os.path.join(DATA_DIR, "stock_names.json")
    if os.path.exists(name_path):
        with open(name_path, "r", encoding="utf-8") as fn:
            name_map = json.load(fn)
    
    results = []
    for i, code in enumerate(codes):
        sd = recent[recent["code"] == code].copy()
        if len(sd) < 10:
            continue
        
        sd = sd.sort_values("date_dt")
        sd["ret_1d"] = sd["close"].pct_change()
        sd["ret_5d"] = sd["close"].pct_change(5)
        sd["ret_20d"] = sd["close"].pct_change(20)
        sd["ret_60d"] = sd["close"].pct_change(60)
        
        row = sd.iloc[-1]
        
        # === 基础数据 ===
        ret5 = safe(row.get("ret_5d"))
        ret20 = safe(row.get("ret_20d"))
        ret60 = safe(row.get("ret_60d"))
        rsi14 = safe(row.get("rsi_14"), 50)
        vr = safe(row.get("volume_ratio_5d"), 1)
        bias20 = safe(row.get("bias_20"))
        amp = safe(row.get("amplitude"))
        close = safe(row.get("close"))
        turnover = safe(row.get("turnover_rate"))
        
        # === MACD ===
        dif = safe(row.get("dif"))
        dea = safe(row.get("dea"))
        macd_bar = safe(row.get("macd_bar"))
        
        # === KDJ ===
        k_val = safe(row.get("k"), 50)
        d_val = safe(row.get("d"), 50)
        j_val = safe(row.get("j"), 50)
        
        # === 成交量分析 ===
        vol_analysis = compute_volume_analysis(sd)
        atr = compute_trend_strength(sd)
        
        # === 支撑阻力 ===
        support, resistance = compute_support_resistance(sd)
        
        # === 斐波那契 ===
        fib = compute_fibonacci(sd)
        
        # === 短线评分 (5日) ===
        short_score = 0
        short_factors = []
        if ret5 > 0.03: short_score += 20; short_factors.append("强势上涨")
        elif ret5 > 0.01: short_score += 12; short_factors.append("温和上涨")
        elif ret5 > -0.01: short_score += 5; short_factors.append("横盘整理")
        else: short_score -= 5; short_factors.append("短线走弱")
        
        if 35 <= rsi14 <= 70: short_score += 10
        elif rsi14 < 30: short_score += 8; short_factors.append("超卖反弹机会")
        else: short_score -= 3
        
        if vr > 1.3: short_score += 8; short_factors.append("放量配合")
        elif vr > 0.8: short_score += 3
        
        if dif > dea: short_score += 6; short_factors.append("MACD金叉")
        else: short_score -= 2
        
        if j_val < 20: short_score += 8; short_factors.append("KDJ超卖")
        elif j_val > 80: short_score -= 4; short_factors.append("KDJ超买")
        
        # === 中线评分 (1月) ===
        mid_score = 0
        mid_factors = []
        if ret20 > 0.05: mid_score += 25; mid_factors.append("月线强势")
        elif ret20 > 0.02: mid_score += 15; mid_factors.append("中线偏强")
        elif ret20 > -0.03: mid_score += 5; mid_factors.append("中线横盘")
        else: mid_score -= 10; mid_factors.append("中线走弱")
        
        if 40 <= rsi14 <= 65: mid_score += 10
        elif rsi14 > 75: mid_score -= 5
        
        if -8 < bias20 < 8: mid_score += 5
        elif bias20 < -12: mid_score += 8; mid_factors.append("中线超跌")
        
        # === 长线评分 (3月) ===
        long_score = 0
        long_factors = []
        if ret60 > 0.1: long_score += 30; long_factors.append("强势长牛")
        elif ret60 > 0.05: long_score += 20; long_factors.append("稳健上涨")
        elif ret60 > -0.05: long_score += 8; long_factors.append("长期横盘")
        else: long_score -= 10; long_factors.append("长期弱势")
        
        # 均线多头排列
        ma5 = safe(row.get("ma_5"))
        ma10 = safe(row.get("ma_10"))
        ma20 = safe(row.get("ma_20"))
        ma60 = safe(row.get("ma_60"))
        if ma5 > ma10 > ma20 > ma60 > 0:
            long_score += 15; long_factors.append("均线多头排列")
        elif close > ma60 > 0:
            long_score += 5
        
        # === 综合评价 ===
        total_short = min(max(round(short_score, 1), -20), 40)
        total_mid = min(max(round(mid_score, 1), -15), 45)
        total_long = min(max(round(long_score, 1), -15), 50)
        overall = round(total_short * 0.4 + total_mid * 0.35 + total_long * 0.25, 1)
        
        # 评级
        if overall > 30: rating = "强烈推荐"; rating_cls = "hot"
        elif overall > 22: rating = "推荐关注"; rating_cls = "warm"
        elif overall > 14: rating = "中性观察"; rating_cls = "mild"
        elif overall > 5: rating = "谨慎观望"; rating_cls = "cold"
        else: rating = "暂时回避"; rating_cls = "cold"
        
        # 短线趋势描述
        if ret5 > 0.05: short_trend = "爆发上攻"
        elif ret5 > 0.03: short_trend = "强势上涨"
        elif ret5 > 0.01: short_trend = "温和上涨"
        elif ret5 > -0.01: short_trend = "横盘整理"
        elif rsi14 < 35: short_trend = "超跌待反弹"
        elif rsi14 < 25: short_trend = "深度超跌"
        else: short_trend = "弱势下跌"
        
        # 中线趋势
        if ret20 > 0.08 and dif > dea: mid_trend = "加速上行"
        elif ret20 > 0.03: mid_trend = "上升通道"
        elif ret20 > 0: mid_trend = "震荡偏多"
        elif ret20 > -0.05: mid_trend = "区间震荡"
        else: mid_trend = "下行趋势"
        
        # 长线趋势
        if ret60 > 0.2 and ma5 > ma60: long_trend = "长牛格局"
        elif ret60 > 0.1: long_trend = "稳步上行"
        elif ret60 > 0: long_trend = "缓慢攀升"
        elif ret60 > -0.1: long_trend = "低位盘整"
        else: long_trend = "长期走弱"
        
        # 信号
        sigs = []
        if rsi14 < 30: sigs.append("RSI超卖")
        if rsi14 > 80: sigs.append("RSI超买")
        if bias20 < -8: sigs.append("严重超跌")
        if vr > 1.8: sigs.append("异常放量")
        if dif > dea and macd_bar > 0: sigs.append("MACD多头")
        if j_val < 0: sigs.append("KDJ底背离")
        if j_val > 100: sigs.append("KDJ顶背离")
        if support > 0 and close < support * 1.02: sigs.append("接近支撑位")
        if resistance > 0 and close > resistance * 0.98: sigs.append("接近阻力位")
        if not sigs: sigs.append("无明显信号")
        
        info = {
            "code": str(code),
            "name": name_map.get(str(code), str(code)),
            # 基础
            "close": close,
            "ret_5d": ret5, "ret_20d": ret20, "ret_60d": ret60,
            "vol_ratio_5d": vr,
            "rsi_14": rsi14,
            "bias_20": bias20,
            "amplitude": amp,
            "turnover_rate": turnover,
            # 技术指标
            "macd_dif": dif, "macd_dea": dea, "macd_bar": macd_bar,
            "macd_signal": compute_macd_signal(dif, dea),
            "kdj_k": k_val, "kdj_d": d_val, "kdj_j": j_val,
            "kdj_signal": compute_kdj_signal(k_val, d_val, j_val),
            # 均线
            "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
            # 支撑阻力
            "support": support, "resistance": resistance,
            # 成交量
            "vol_analysis": vol_analysis,
            "atr": atr,
            # 斐波那契
            "fibonacci": fib,
            # 评分
            "short_score": total_short,
            "mid_score": total_mid,
            "long_score": total_long,
            "total_score": overall,
            # 趋势
            "short_trend": short_trend,
            "mid_trend": mid_trend,
            "long_trend": long_trend,
            # 因子
            "short_factors": short_factors,
            "mid_factors": mid_factors,
            "long_factors": long_factors,
            # 评级
            "rating": rating,
            "rating_cls": rating_cls,
            # 信号
            "signals": sigs,
            "update_time": str(latest.date()),
            "has_kline": True,
        }
        
        results.append(info)
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(codes)}")
    
    # 保存
    out_path = os.path.join(OUTPUT_DIR, "stock_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    
    kb = os.path.getsize(out_path) / 1024
    print(f"\nDone! {len(results)} stocks -> {out_path} ({kb:.0f}KB)")
    
    # 统计
    df = pd.DataFrame(results)
    if "rating" in df.columns:
        print("Rating distribution:")
        for k, v in df["rating"].value_counts().items():
            print(f"  {k}: {v}")
    
    print(f"Short score: {df['short_score'].mean():.1f} avg")
    print(f"Mid score:   {df['mid_score'].mean():.1f} avg")
    print(f"Long score:  {df['long_score'].mean():.1f} avg")

if __name__ == "__main__":
    compute()