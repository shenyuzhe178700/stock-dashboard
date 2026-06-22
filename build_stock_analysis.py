# -*- coding: utf-8 -*-
"""批量预计算HS300个股分析数据"""
import sys, os
sys.path.insert(0, r"D:\quant_a_stock")
import pandas as pd, numpy as np, json
from config import OUTPUT_DIR, DATA_DIR

def safe(v, d=0):
    try:
        x = float(v)
        return round(0 if pd.isna(x) else x, 4)
    except: return d

def compute():
    from data.factors import FactorEngine
    
    cache = os.path.join(DATA_DIR, "daily_bs_300_20230101_20260601.pkl")
    print(f"Load: {cache}")
    daily = pd.read_pickle(cache)
    codes = sorted(daily["code"].unique())
    
    engine = FactorEngine(daily)
    f = engine.compute_all_factors()
    
    # Handle date
    f["date_dt"] = pd.to_datetime(f["date"])
    latest = f["date_dt"].max()
    cutoff = latest - pd.Timedelta(days=15)
    recent = f[f["date_dt"] >= cutoff]
    
    results = []
    for i, code in enumerate(codes):
        sd = recent[recent["code"] == code].copy()
        if len(sd) < 5: continue
        
        sd["ret_1d"] = sd["close"].pct_change()
        sd["ret_5d"] = sd["close"].pct_change(5)
        
        row = sd.iloc[-1]
        
        ret5 = safe(row.get("ret_5d"))
        rsi14 = safe(row.get("rsi_14"), 50)
        vr = safe(row.get("volume_ratio_5d"), 1)
        bias20 = safe(row.get("bias_20"))
        amp = safe(row.get("amplitude"))
        ret1m = safe(row.get("ret_1m"))
        close = safe(row.get("close"))
        turnover = safe(row.get("turnover_rate"))
        to_chg = safe(row.get("turnover_change"))
        
        info = {
            "code": str(code),
            "ret_1d": safe(row.get("ret_1d")),
            "ret_5d": ret5, "ret_1m": ret1m,
            "vol_ratio_5d": vr, "rsi_14": rsi14,
            "bias_20": bias20, "amplitude": amp,
            "close": close, "turnover_rate": turnover,
            "turnover_change": to_chg,
            "ma_20": safe(row.get("ma_20")),
        }
        
        score = 0
        if ret5 > 0: score += min(ret5 * 40, 20)
        else: score += max(ret5 * 20, -12)
        if 35 <= rsi14 <= 70: score += 10
        elif rsi14 < 30: score += 8
        if vr > 1.3: score += 8
        elif vr > 0.8: score += 3
        if -5 < bias20 < 5: score += 5
        if amp > 0.03: score += 3
        if to_chg > 0.5: score += 4
        
        info["total_score"] = round(score, 1)
        if score > 25: info["rating"] = "强烈推荐"
        elif score > 18: info["rating"] = "推荐关注"
        elif score > 10: info["rating"] = "中性观察"
        elif score > 3: info["rating"] = "谨慎观望"
        else: info["rating"] = "暂时回避"
        
        if ret5 > 0.03 and vr > 1: info["short_trend"] = "强势上攻"
        elif ret5 > 0.01: info["short_trend"] = "温和上涨"
        elif ret5 > -0.01: info["short_trend"] = "横盘整理"
        elif rsi14 < 35: info["short_trend"] = "超跌待反弹"
        else: info["short_trend"] = "弱势下跌"
        
        if ret1m > 0.05 and rsi14 > 55: info["mid_trend"] = "上升通道"
        elif ret1m > 0.02: info["mid_trend"] = "震荡偏多"
        elif ret1m > -0.05: info["mid_trend"] = "区间震荡"
        else: info["mid_trend"] = "下行趋势"
        
        sigs = []
        if rsi14 < 30: sigs.append("RSI超卖")
        if rsi14 > 80: sigs.append("RSI超买")
        if bias20 < -8: sigs.append("严重超跌")
        if bias20 < -3: sigs.append("偏离均线")
        if vr > 1.8: sigs.append("异常放量")
        if vr < 0.4: sigs.append("极度缩量")
        info["signals"] = sigs if sigs else ["无明显信号"]
        info["update_time"] = str(latest.date())
        
        results.append(info)
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(codes)}")
    
    path = os.path.join(OUTPUT_DIR, "stock_analysis.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    
    kb = os.path.getsize(path)/1024
    print(f"\n{len(results)} stocks | {path} ({kb:.0f} KB)")
    df = pd.DataFrame(results)
    if "rating" in df.columns:
        for k, v in df["rating"].value_counts().items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    compute()