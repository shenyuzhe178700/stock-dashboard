# -*- coding: utf-8 -*-
"""GitHub Actions CI 流水线 - 纯akshare实时数据, 无需本地缓存"""
import sys, os, json, time
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from datetime import datetime

OUTPUT_DIR = "outputs"
DATA_DIR = "data/cache"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

os.environ.setdefault("NO_PROXY", "*")

now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

def run_step(name):
    print(f"\n{'='*50}\n  [{name}]\n{'='*50}")

# ===== 1. 获取实时全A股行情 =====
run_step("1. 实时行情 (akshare)")
import akshare as ak
try:
    spot = ak.stock_zh_a_spot_em()
    spot = spot.rename(columns={
        "代码": "code", "名称": "name", "最新价": "close", "涨跌幅": "pct_chg",
        "换手率": "turnover_rate", "量比": "volume_ratio", "总市值": "total_mv",
        "市盈率-动态": "pe_ttm", "市净率": "pb"
    })
    spot["code"] = spot["code"].astype(str).str.strip()
    spot = spot[~spot["name"].str.contains("ST|退", na=False)]
    spot = spot[spot["close"] > 0]
    spot.to_csv(f"{OUTPUT_DIR}/realtime_spot.csv", index=False)
    print(f"  全A股: {len(spot)} 只")
except Exception as e:
    print(f"  FAIL: {e}")
    spot = pd.DataFrame()

# ===== 2. 板块行情 =====
run_step("2. 板块行情")
try:
    sectors = ak.stock_board_industry_name_em()
    sectors = sectors.rename(columns={
        "板块名称": "sector_name", "涨跌幅": "pct_chg",
        "上涨家数": "up_count", "下跌家数": "down_count",
        "领涨股票": "lead_stock"
    })
    if "up_count" in sectors.columns:
        sectors["total_count"] = sectors["up_count"] + sectors["down_count"]
        sectors["up_ratio"] = (sectors["up_count"] / sectors["total_count"].replace(0,1)).round(3)
    sectors.to_csv(f"{OUTPUT_DIR}/realtime_sectors.csv", index=False)
    print(f"  板块: {len(sectors)} 个")
except Exception as e:
    print(f"  FAIL: {e}")
    sectors = pd.DataFrame()

# ===== 3. 资金流向 =====
run_step("3. 资金流向")
try:
    fund = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流向")
    fund = fund.rename(columns={
        "名称": "sector_name", "今日涨跌幅": "pct_chg",
        "主力净流入-净额": "main_net_inflow",
        "主力净流入-净占比": "main_net_ratio"
    })
    for col in ["main_net_inflow", "pct_chg"]:
        if col in fund.columns:
            fund[f"{col}_rank"] = fund[col].rank(ascending=False, pct=True)
    score_cols = [c for c in ["main_net_inflow_rank", "pct_chg_rank"] if c in fund.columns]
    if score_cols:
        fund["capital_score"] = fund[score_cols].mean(axis=1)
        fund = fund.sort_values("capital_score", ascending=False)
    fund.to_csv(f"{OUTPUT_DIR}/realtime_capital_flow.csv", index=False)
    print(f"  数据: {len(fund)} 条")
except Exception as e:
    print(f"  FAIL: {e}")
    fund = pd.DataFrame()

# ===== 4. 综合热度板块 =====
run_step("4. 综合热度")
try:
    if not sectors.empty:
        hot = sectors.copy()
        if "pct_chg" in hot.columns:
            hot["score_pct"] = hot["pct_chg"].rank(pct=True)
        if "up_ratio" in hot.columns:
            hot["score_up"] = hot["up_ratio"].rank(pct=True)
        score_cols = [c for c in ["score_pct", "score_up"] if c in hot.columns]
        if score_cols:
            hot["hot_score"] = hot[score_cols].mean(axis=1)
            hot = hot.sort_values("hot_score", ascending=False)
        hot.to_csv(f"{OUTPUT_DIR}/realtime_hot_sectors.csv", index=False)
        print(f"  板块: {len(hot)} 个")
except Exception as e:
    print(f"  FAIL: {e}")

# ===== 5. 涨幅榜 =====
run_step("5. 涨幅榜")
try:
    if not spot.empty and "pct_chg" in spot.columns:
        top = spot.nlargest(30, "pct_chg")
        cols = [c for c in ["code","name","close","pct_chg","turnover_rate","volume_ratio","total_mv"] if c in top.columns]
        top[cols].to_csv(f"{OUTPUT_DIR}/realtime_top_stocks.csv", index=False)
        print(f"  Top30 已保存")
except Exception as e:
    print(f"  FAIL: {e}")

# ===== 6. 个股分析(简化版) =====
run_step("6. 个股快速分析")
try:
    if not spot.empty:
        results = []
        top_codes = spot.nlargest(20, "pct_chg")["code"].tolist() if "pct_chg" in spot.columns else spot["code"].head(100).tolist()
        
        for i, code in enumerate(top_codes):
            try:
                hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
                if hist is None or len(hist) < 10:
                    continue
                hist = hist.tail(60)
                close = hist["收盘"].values
                ret_5d = (close[-1] / close[-6] - 1) if len(close) >= 6 else 0
                ret_20d = (close[-1] / close[-21] - 1) if len(close) >= 21 else 0
                ret_60d = (close[-1] / close[-1] - 1)  # simplified
                
                # RSI
                delta = pd.Series(close).diff()
                gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
                loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
                rsi = 100 - 100/(1 + gain/loss) if loss > 0 else 50
                
                # Score
                short_score = min(15 + ret_5d * 200, 35) if ret_5d > -0.1 else 5
                mid_score = min(12 + ret_20d * 100, 30)
                overall = short_score * 0.5 + mid_score * 0.3 + 10
                
                results.append({
                    "code": code,
                    "name": str(spot[spot["code"]==code]["name"].values[0]) if not spot[spot["code"]==code].empty else code,
                    "close": round(float(close[-1]), 2),
                    "ret_5d": round(ret_5d, 4), "ret_20d": round(ret_20d, 4),
                    "rsi_14": round(float(rsi), 1),
                    "short_score": round(short_score, 1), "mid_score": round(mid_score, 1),
                    "total_score": round(overall, 1),
                    "short_trend": "强势上涨" if ret_5d > 0.03 else "温和上涨" if ret_5d > 0.01 else "横盘" if ret_5d > -0.01 else "弱势",
                    "mid_trend": "上升通道" if ret_20d > 0.05 else "震荡偏多" if ret_20d > 0.02 else "区间震荡",
                    "long_trend": "N/A",
                    "rating": "强烈推荐" if overall > 25 else "推荐关注" if overall > 18 else "中性观察" if overall > 12 else "谨慎观望",
                    "rating_cls": "hot" if overall > 25 else "warm" if overall > 18 else "mild",
                    "signals": ["实时数据"],
                    "update_time": now_str,
                })
            except:
                pass
            
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(top_codes)}")
        
        with open(f"{OUTPUT_DIR}/stock_analysis.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"  分析: {len(results)} 只")
except Exception as e:
    print(f"  FAIL: {e}")

# ===== 7. 生成K线数据(简化) =====
run_step("7. K线数据")
try:
    if not spot.empty:
        klines = {}
        top_codes = spot.nlargest(20, "pct_chg")["code"].tolist() if "pct_chg" in spot.columns else spot["code"].head(30).tolist()
        for code in top_codes:
            try:
                hist = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
                if hist is None or len(hist) < 10:
                    continue
                hist = hist.tail(60)
                data = []
                for _, row in hist.iterrows():
                    data.append([
                        str(row["日期"])[:10],
                        round(float(row["开盘"]), 2),
                        round(float(row["收盘"]), 2),
                        round(float(row["最低"]), 2),
                        round(float(row["最高"]), 2),
                        int(row.get("成交量", 0)),
                        0,0,0,50,50,50,0,0,0,0,0,0,0
                    ])
                klines[code] = {"code": code, "name": code, "count": len(data), "last_close": round(float(hist["收盘"].iloc[-1]), 2), "data": data}
            except: pass
        with open(f"{OUTPUT_DIR}/kline_data.json", "w", encoding="utf-8") as f:
            json.dump(klines, f, ensure_ascii=False)
        print(f"  K线: {len(klines)} 只")
except Exception as e:
    print(f"  FAIL: {e}")

print(f"\n{'='*50}")
print(f"  CI Pipeline Done! {now_str}")
print(f"{'='*50}")