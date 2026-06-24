# -*- coding: utf-8 -*-
"""Generate K-line data JSON for ECharts candlestick display"""
import sys, os, json
sys.path.insert(0, r"D:\quant_a_stock")
import pandas as pd, numpy as np
from config import OUTPUT_DIR, DATA_DIR

def safe(v, default=0):
    try: return default if pd.isna(v) else v
    except: return default

def calc_macd(df):
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar

def calc_kdj(df, n=9):
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j

def build_klines(daily, codes, max_codes=100, tail_days=120):
    results = {}
    daily = daily.sort_values(["code", "date"])
    codes_to_use = codes[:max_codes]
    for i, code in enumerate(codes_to_use):
        sd = daily[daily["code"] == str(code)].copy()
        if len(sd) < 20: continue
        sd = sd.tail(tail_days).reset_index(drop=True)
        dif, dea, macd_bar = calc_macd(sd)
        k, d, j = calc_kdj(sd)
        ma5 = sd["close"].rolling(5).mean()
        ma10 = sd["close"].rolling(10).mean()
        ma20 = sd["close"].rolling(20).mean()
        ma60 = sd["close"].rolling(60).mean()
        
        kline_data = []
        for idx, row in sd.iterrows():
            vol = safe(row.get("volume", 0))
            try: vol = int(float(vol))
            except: vol = 0
            kline_data.append([
                str(row["date"])[:10],
                round(float(safe(row["open"])), 2),
                round(float(safe(row["close"])), 2),
                round(float(safe(row["low"])), 2),
                round(float(safe(row["high"])), 2),
                vol,
                round(float(safe(dif.iloc[idx])), 3),
                round(float(safe(dea.iloc[idx])), 3),
                round(float(safe(macd_bar.iloc[idx])), 3),
                round(float(safe(k.iloc[idx], 50)), 1),
                round(float(safe(d.iloc[idx], 50)), 1),
                round(float(safe(j.iloc[idx], 50)), 1),
                round(float(safe(ma20.iloc[idx] * 1.05 if pd.notna(ma20.iloc[idx]) else 0)), 2),
                round(float(safe(ma20.iloc[idx])), 2),
                round(float(safe(ma20.iloc[idx] * 0.95 if pd.notna(ma20.iloc[idx]) else 0)), 2),
                round(float(safe(ma5.iloc[idx])), 2),
                round(float(safe(ma10.iloc[idx])), 2),
                round(float(safe(ma20.iloc[idx])), 2),
                round(float(safe(ma60.iloc[idx])), 2),
            ])
        
        results[str(code)] = {
            "code": str(code),
            "name": str(code),
            "count": len(kline_data),
            "last_close": round(float(safe(sd.iloc[-1]["close"])), 2),
            "data": kline_data,
        }
        
        if (i+1) % 20 == 0: print(f"  K-line: {i+1}/{min(len(codes), max_codes)}")
    return results

def filter_top_klines(out_path, top_n=50):
    analysis_path = out_path.replace("kline_data.json", "stock_analysis.json")
    if not os.path.exists(analysis_path): return
    with open(analysis_path, "r", encoding="utf-8") as f:
        stocks = json.load(f)
    stocks.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    top_codes = set(s["code"] for s in stocks[:top_n])
    with open(out_path, "r", encoding="utf-8") as f:
        klines = json.load(f)
    filtered = {c: klines[c] for c in top_codes if c in klines}
    for c in filtered:
        filtered[c]["data"] = filtered[c]["data"][-60:]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False)
    print(f"  Auto-filter: {len(filtered)} stocks, {os.path.getsize(out_path)/1024:.0f}KB")

def main():
    print("=" * 50)
    print("  Build K-line Chart Data")
    print("=" * 50)
    
    cache_files = [
        ("daily_bs_300_20230101_20260601.pkl", "HS300"),
        ("daily_bs_300_20251224_20260622.pkl", "HS300 recent"),
    ]
    daily = None
    for fn, label in cache_files:
        path = os.path.join(DATA_DIR, fn)
        if os.path.exists(path):
            print(f"Load: {path} ({os.path.getsize(path)/1024/1024:.1f}MB)")
            try:
                daily = pd.read_pickle(path)
                print(f"  OK: {len(daily)} rows, {daily['code'].nunique()} stocks")
                break
            except Exception as e: print(f"  Failed: {e}")
    
    if daily is None:
        print("ERROR: No cache file!")
        return
    
    codes = sorted(daily["code"].unique())
    klines = build_klines(daily, codes, max_codes=200)
    
    out_path = os.path.join(OUTPUT_DIR, "kline_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(klines, f, ensure_ascii=False)
    
    kb = os.path.getsize(out_path) / 1024
    print(f"Done! {len(klines)} stocks -> {out_path} ({kb:.0f}KB)")
    filter_top_klines(out_path)

if __name__ == "__main__":
    main()