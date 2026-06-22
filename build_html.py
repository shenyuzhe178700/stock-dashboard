# -*- coding: utf-8 -*-
"""生成量化看板HTML"""
import sys, os, json
sys.path.insert(0, r"D:\quant_a_stock")
import pandas as pd, base64
from config import OUTPUT_DIR
from datetime import datetime

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
        s = f"{v*100:+.2f}%"
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
        return '<div class="bar-wrap"><div class="bar" style="width:%d%%;background:%s"></div><span>%d</span></div>' % (int(pct), color, int(pct))
    except: return ""

def heat_badge(score):
    try:
        s = float(score)
        if s > 0.7: return '<span class="badge hot">极热</span>'
        if s > 0.5: return '<span class="badge warm">热</span>'
        if s > 0.3: return '<span class="badge mild">温</span>'
        return '<span class="badge cold">冷</span>'
    except: return ""

now = datetime.now().strftime("%Y-%m-%d %H:%M")
date_short = datetime.now().strftime("%m/%d")

stock_picks = load_csv("stock_picks.csv")
hot_picks = load_csv("hot_sector_picks.csv")
sector_heat = load_csv("sector_heat.csv")
cap_sectors = load_csv("capital_flow_sectors.csv")
cap_picks = load_csv("capital_flow_picks.csv")
backtest_b64 = img_to_b64(os.path.join(OUTPUT_DIR, "backtest_result.png"))

txt_path = os.path.join(OUTPUT_DIR, "stock_picks.txt")
report_txt = ""
if os.path.exists(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        report_txt = f.read()

def build_table(df, cols, labels, max_rows=30, col_formatters=None):
    if df is None: return '<div class="empty">暂无数据</div>'
    if col_formatters is None: col_formatters = {}
    rows = ""
    for i, r in df.head(max_rows).iterrows():
        row = '<tr><td class="rank">%d</td>' % (i+1)
        for c in cols:
            val = r.get(c, "")
            if c in col_formatters:
                val = col_formatters[c](val)
            else:
                try: val = str(val)
                except: val = ""
            cls = "code" if c == "code" else ("name" if c in ("industry",) else "")
            row += '<td class="%s">%s</td>' % (cls, val)
        row += '</tr>'
        rows += row
    header = '<tr>' + ''.join('<th>%s</th>' % l for l in labels) + '</tr>'
    return '<table class="table"><thead>%s</thead><tbody>%s</tbody></table>' % (header, rows)

def build_sector_table(df):
    if df is None: return '<div class="empty">暂无数据</div>'
    rows = ""
    for i, r in df.head(12).iterrows():
        rows += '<tr><td class="rank">%d</td><td class="name">%s</td><td>%s</td><td>%s只</td><td>%s</td><td>%s</td></tr>' % (
            i+1, r.get('industry',''), fmt_pct(r.get('sector_ret_5d',0)),
            r.get('stock_count',''), score_bar(r.get('heat_score',0)), heat_badge(r.get('heat_score',0)))
    return '<table class="table"><thead><tr><th>#</th><th>板块</th><th>5日涨幅</th><th>成分股</th><th>热度评分</th><th>热度</th></tr></thead><tbody>%s</tbody></table>' % rows

def build_hot_table(df):
    if df is None: return ""
    rows = ""
    for i, r in df.head(30).iterrows():
        rows += '<tr><td class="rank">%d</td><td class="code">%s</td><td class="name">%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            i+1, r.get('code',''), r.get('industry',''), fmt_pct(r.get('ret_5d',0)),
            fmt_num(r.get('vol_ratio_5d',0)), fmt_num(r.get('rsi_6',0)),
            fmt_pct(r.get('gap_from_5d_high',0)), score_bar(r.get('score',0)))
    return '<table class="table"><thead><tr><th>#</th><th>代码</th><th>板块</th><th>5日涨幅</th><th>量比</th><th>RSI(6)</th><th>距5日高</th><th>评分</th></tr></thead><tbody>%s</tbody></table>' % rows

hot_count = len(hot_picks) if hot_picks is not None else 0
cap_count = len(cap_picks) if cap_picks is not None else 0
stock_count = len(stock_picks) if stock_picks is not None else 0

# Build replacement dict
repl = {
    "{{NOW}}": now,
    "{{DATE_SHORT}}": date_short,
    "{{HOT_COUNT}}": str(hot_count),
    "{{CAP_COUNT}}": str(cap_count),
    "{{STOCK_COUNT}}": str(stock_count),
    "{{SECTOR_TABLE}}": build_sector_table(sector_heat),
    "{{HOT_TABLE}}": build_hot_table(hot_picks),
    "{{CAP_SECTOR_TABLE}}": build_table(cap_sectors,
        ["industry","sector_ret_5d","avg_vol_ratio","avg_net_capital_flow","capital_score"],
        ["#","板块","5日涨跌","量比","主力净流","资金评分"], 12,
        {"sector_ret_5d": fmt_pct, "capital_score": score_bar}),
    "{{CAP_PICK_TABLE}}": build_table(cap_picks,
        ["code","industry","ret_5d","stock_upside","net_capital_flow","stock_score"],
        ["#","代码","板块","5日涨跌","上涨空间","主力净流","评分"], 24,
        {"ret_5d": fmt_pct, "stock_upside": fmt_pct, "stock_score": score_bar}),
    "{{STOCK_PICKS_TABLE}}": build_table(stock_picks,
        ["code","score","rank"],
        ["#","代码","得分","排名分位"], 30,
        {"score": fmt_num}),
    "{{REPORT_TXT}}": report_txt if report_txt else "暂无报告",
    "{{BACKTEST_IMG}}": '<img class="chart" src="data:image/png;base64,%s" alt="回测曲线" style="width:100%%">' % backtest_b64 if backtest_b64 else '<div class="empty">暂无回测图片</div>',
}

# Read template and apply replacements
tmpl_path = os.path.join(os.path.dirname(__file__), "template.html")
html = ""
with open(tmpl_path, "r", encoding="utf-8") as f:
    html = f.read()

for k, v in repl.items():
    html = html.replace(k, v)

out_path = os.path.join(OUTPUT_DIR, "quant_dashboard.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
idx_path = os.path.join(OUTPUT_DIR, "index.html")
with open(idx_path, "w", encoding="utf-8") as f:
    f.write(html)

print("OK " + out_path)
print("SIZE " + str(os.path.getsize(out_path)))
