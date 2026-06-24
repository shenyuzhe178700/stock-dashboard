# -*- coding: utf-8 -*-
"""生成量化看板HTML - v3 支持名称+丰富月度选股"""
import sys, os, json
sys.path.insert(0, r"D:\quant_a_stock")
import pandas as pd, base64
from config import OUTPUT_DIR, DATA_DIR
from datetime import datetime

USE_REALTIME = "--realtime" in sys.argv

name_map = {}
name_path = os.path.join(DATA_DIR, "stock_names.json")
if os.path.exists(name_path):
    with open(name_path, "r", encoding="utf-8") as f:
        name_map = json.load(f)
if not name_map:
    try:
        from build_stock_names import HS300_NAMES
        name_map = HS300_NAMES
    except: pass

def load_csv(name):
    p = os.path.join(OUTPUT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None

def img_to_b64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def fmt_pct(v, color=True):
    try:
        v = float(v)
        s = f"{v*100:+.2f}%" if abs(v) < 100 else f"{v:+.2f}%"
        if color and v > 0: return f'<span class="up">{s}</span>'
        if color and v < 0: return f'<span class="down">{s}</span>'
        return s
    except: return str(v)

def fmt_num(v):
    try: return f"{float(v):.4f}"
    except: return str(v)

def score_bar(score, max_score=1.0):
    try:
        pct = min(max(float(score)/max_score*100, 0), 100)
        color = "#22c55e" if pct > 50 else "#eab308" if pct > 30 else "#ef4444"
        return f'<div class="bar-wrap"><div class="bar" style="width:{int(pct)}%;background:{color}"></div><span>{int(pct)}</span></div>'
    except: return ""

def heat_badge(score):
    try:
        s = float(score)
        if s > 0.7: return '<span class="badge hot">极热</span>'
        if s > 0.5: return '<span class="badge warm">热</span>'
        if s > 0.3: return '<span class="badge mild">温</span>'
        return '<span class="badge cold">冷</span>'
    except: return ""

def get_name(code):
    return name_map.get(str(code), str(code))

now = datetime.now().strftime("%Y-%m-%d %H:%M")
date_short = datetime.now().strftime("%m/%d")

realtime_spot = load_csv("realtime_spot.csv")
realtime_sectors = load_csv("realtime_sectors.csv")
realtime_fund_flow = load_csv("realtime_fund_flow.csv")
realtime_hot_sectors = load_csv("realtime_hot_sectors.csv")
realtime_top = load_csv("realtime_top_stocks.csv")
realtime_capital = load_csv("realtime_capital_flow.csv")

stock_picks = load_csv("stock_picks.csv")
hot_picks = load_csv("hot_sector_picks.csv")
sector_heat = load_csv("sector_heat.csv")
cap_sectors = load_csv("capital_flow_sectors.csv")
cap_picks = load_csv("capital_flow_picks.csv")
backtest_b64 = img_to_b64(os.path.join(OUTPUT_DIR, "backtest_result.png"))

# Add names to all DataFrames
for df in [stock_picks, hot_picks, cap_picks]:
    if df is not None and not df.empty and "code" in df.columns:
        if "name" not in df.columns:
            df["name"] = df["code"].apply(get_name)

data_source = "akshare+东方财富(实时)" if USE_REALTIME else "baostock(历史)"

def build_realtime_spot_table(df, n=15):
    if df is None or df.empty: return '<div class="empty">暂无实时数据</div>'
    top = df.nlargest(n, "pct_chg") if "pct_chg" in df.columns else df.head(n)
    rows = ""
    for i, (_, r) in enumerate(top.iterrows()):
        rows += f'<tr><td class="rank">{i+1}</td><td class="code">{r.get("code","?")}</td><td class="name">{get_name(r.get("code","?"))}</td><td>{r.get("close",0)}</td><td>{fmt_pct(r.get("pct_chg",0)/100)}</td><td>{r.get("turnover_rate",0)}%</td></tr>'
    return '<table class="table"><thead><tr><th>#</th><th>代码</th><th>名称</th><th>现价</th><th>涨跌幅</th><th>换手率</th></tr></thead><tbody>' + rows + '</tbody></table>'

def build_realtime_sector_table(df, n=12):
    if df is None or df.empty: return '<div class="empty">暂无板块数据</div>'
    top = df.nlargest(n, "pct_chg") if "pct_chg" in df.columns else df.head(n)
    rows = ""
    for i, (_, r) in enumerate(top.iterrows()):
        rows += f'<tr><td class="rank">{i+1}</td><td class="name">{r.get("sector_name","?")}</td><td>{fmt_pct(r.get("pct_chg",0)/100)}</td><td>{r.get("up_count",0)}/{int(r.get("up_count",0))+int(r.get("down_count",0))}</td><td>{r.get("lead_stock","")}</td><td>{score_bar(r.get("hot_score",0))}</td></tr>'
    return '<table class="table"><thead><tr><th>#</th><th>板块</th><th>涨跌幅</th><th>涨/跌</th><th>领涨股</th><th>热度</th></tr></thead><tbody>' + rows + '</tbody></table>'

def build_realtime_capital_table(df, n=12):
    if df is None or df.empty: return '<div class="empty">暂无资金数据</div>'
    top = df.nlargest(n, "capital_score") if "capital_score" in df.columns else df.head(n)
    rows = ""
    for i, (_, r) in enumerate(top.iterrows()):
        inflow = r.get("main_net_inflow", 0)
        inflow_str = f"{inflow/1e8:.1f}亿" if inflow else ""
        rows += f'<tr><td class="rank">{i+1}</td><td class="name">{r.get("sector_name","?")}</td><td>{fmt_pct(r.get("pct_chg",0)/100)}</td><td>{inflow_str}</td><td>{r.get("main_net_ratio",0)}</td><td>{score_bar(r.get("capital_score",0))}</td></tr>'
    return '<table class="table"><thead><tr><th>#</th><th>板块</th><th>涨跌幅</th><th>主力净流</th><th>净占比</th><th>认可度</th></tr></thead><tbody>' + rows + '</tbody></table>'

def build_table(df, cols, labels, max_rows=30, col_formatters=None):
    if df is None: return '<div class="empty">暂无数据</div>'
    if col_formatters is None: col_formatters = {}
    rows = ""
    for i, (_, r) in enumerate(df.head(max_rows).iterrows()):
        row = f'<tr><td class="rank">{i+1}</td>'
        for c in cols:
            val = r.get(c, "")
            if c in col_formatters:
                val = col_formatters[c](val)
            else:
                try: val = str(val)
                except: val = ""
            cls = "code" if c == "code" else ("name" if c == "name" else "")
            row += f'<td class="{cls}">{val}</td>'
        row += '</tr>'
        rows += row
    header = '<tr>' + ''.join(f'<th>{l}</th>' for l in labels) + '</tr>'
    return '<table class="table"><thead>' + header + '</thead><tbody>' + rows + '</tbody></table>'

def build_sector_table(df):
    if df is None: return '<div class="empty">暂无数据</div>'
    rows = ""
    for i, (_, r) in enumerate(df.head(12).iterrows()):
        rows += f'<tr><td class="rank">{i+1}</td><td class="name">{r.get("industry","")}</td><td>{fmt_pct(r.get("sector_ret_5d",0))}</td><td>{r.get("stock_count","")}只</td><td>{score_bar(r.get("heat_score",0))}</td><td>{heat_badge(r.get("heat_score",0))}</td></tr>'
    return '<table class="table"><thead><tr><th>#</th><th>板块</th><th>5日涨幅</th><th>成分股</th><th>热度</th><th>状态</th></tr></thead><tbody>' + rows + '</tbody></table>'

def build_hot_table(df):
    if df is None: return ""
    rows = ""
    for i, (_, r) in enumerate(df.head(30).iterrows()):
        code = str(r.get("code", ""))
        name = get_name(code)
        rows += f'<tr><td class="rank">{i+1}</td><td class="code">{code}</td><td class="name">{name}</td><td>{r.get("industry","")}</td><td>{fmt_pct(r.get("ret_5d",0))}</td><td>{fmt_num(r.get("vol_ratio_5d",0))}</td><td>{fmt_num(r.get("rsi_6",0))}</td><td>{fmt_pct(r.get("gap_from_5d_high",0))}</td><td>{score_bar(r.get("score",0))}</td></tr>'
    return '<table class="table"><thead><tr><th>#</th><th>代码</th><th>名称</th><th>板块</th><th>5日涨幅</th><th>量比</th><th>RSI</th><th>距高</th><th>评分</th></tr></thead><tbody>' + rows + '</tbody></table>'

def build_stock_picks_table(df):
    """丰富版月度选股表格"""
    if df is None: return '<div class="empty">暂无选股数据（请先运行月度多因子模型）</div>'
    # Try to load stock_analysis for richer data
    analysis = {}
    apath = os.path.join(OUTPUT_DIR, "stock_analysis.json")
    if os.path.exists(apath):
        with open(apath, "r", encoding="utf-8") as f:
            for d in json.load(f):
                analysis[d["code"]] = d
    
    rows = ""
    for i, (_, r) in enumerate(df.head(30).iterrows()):
        code = str(r.get("code", ""))
        name = get_name(code)
        score_val = r.get("score", 0)
        # Enrich from analysis
        extra = analysis.get(code, {})
        ret5 = extra.get("ret_5d", 0)
        ret20 = extra.get("ret_20d", 0)
        short_trend = extra.get("short_trend", "N/A")
        mid_trend = extra.get("mid_trend", "N/A")
        rating = extra.get("rating", "N/A")
        rating_cls = extra.get("rating_cls", "mild")
        
        rows += f'<tr><td class="rank">{i+1}</td><td class="code">{code}</td><td class="name">{name}</td><td>{fmt_pct(ret5)}</td><td style="color:#38bdf8">{short_trend}</td><td style="color:#818cf8">{mid_trend}</td><td><span class="badge {rating_cls}">{rating}</span></td><td>{score_bar(score_val, 1.0)}</td></tr>'
    
    return '<table class="table"><thead><tr><th>#</th><th>代码</th><th>名称</th><th>5日涨跌</th><th>短线趋势</th><th>中线趋势</th><th>综合评级</th><th>评分</th></tr></thead><tbody>' + rows + '</tbody></table>'

hot_count = len(hot_picks) if hot_picks is not None else len(realtime_top or [])
cap_count = len(cap_picks) if cap_picks is not None else len(realtime_capital or [])
stock_count = len(stock_picks) if stock_picks is not None else 0
realtime_spot_count = len(realtime_spot) if realtime_spot is not None else 0

if USE_REALTIME and realtime_spot is not None and not realtime_spot.empty:
    sector_html = build_realtime_sector_table(realtime_hot_sectors or realtime_sectors)
    hot_html = build_realtime_spot_table(realtime_top or realtime_spot, n=30)
    cap_sector_html = build_realtime_capital_table(realtime_capital or realtime_fund_flow)
    cap_pick_html = build_realtime_spot_table(realtime_spot, n=24)
else:
    sector_html = build_sector_table(sector_heat)
    hot_html = build_hot_table(hot_picks)
    cap_sector_html = build_table(cap_sectors,
        ["industry","sector_ret_5d","avg_vol_ratio","avg_net_capital_flow","capital_score"],
        ["#","板块","5日涨跌","量比","主力净流","评分"], 12,
        {"sector_ret_5d": fmt_pct, "capital_score": score_bar})
    cap_pick_html = build_table(cap_picks,
        ["code","name","industry","ret_5d","stock_upside","net_capital_flow","stock_score"],
        ["#","代码","名称","板块","5日涨跌","上涨空间","主力净流","评分"], 24,
        {"ret_5d": fmt_pct, "stock_upside": fmt_pct, "stock_score": score_bar})

repl = {
    "{{NOW}}": now,
    "{{DATE_SHORT}}": date_short,
    "{{HOT_COUNT}}": str(hot_count),
    "{{CAP_COUNT}}": str(cap_count),
    "{{STOCK_COUNT}}": str(stock_count),
    "{{SECTOR_TABLE}}": sector_html,
    "{{HOT_TABLE}}": hot_html,
    "{{CAP_SECTOR_TABLE}}": cap_sector_html,
    "{{CAP_PICK_TABLE}}": cap_pick_html,
    "{{STOCK_PICKS_TABLE}}": build_stock_picks_table(stock_picks),
    "{{REPORT_TXT}}": f"数据源: {data_source} | 全A股: {realtime_spot_count}只 | 更新: {now}",
    "{{BACKTEST_IMG}}": f'<img class="chart" src="data:image/png;base64,{backtest_b64}" alt="回测曲线">' if backtest_b64 else '<div class="empty">暂无回测图片</div>',
}

tmpl_path = os.path.join(os.path.dirname(__file__), "template.html")
with open(tmpl_path, "r", encoding="utf-8") as f:
    html = f.read()

for k, v in repl.items():
    html = html.replace(k, v)

for name in ["quant_dashboard.html", "index.html"]:
    out_path = os.path.join(OUTPUT_DIR, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

print(f"OK {os.path.join(OUTPUT_DIR, 'quant_dashboard.html')}")
print(f"SIZE {os.path.getsize(os.path.join(OUTPUT_DIR, 'quant_dashboard.html'))}")
print(f"MODE {'realtime' if USE_REALTIME else 'historical'}")