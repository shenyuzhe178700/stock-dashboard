# -*- coding: utf-8 -*-
"""Inject stock_analysis.json + kline_data.json into HTML"""
import json, os

HTML_PATH = os.path.join("outputs", "index.html")
KV_PATH = os.path.join("outputs", "stock_analysis.json")
KLINE_PATH = os.path.join("outputs", "kline_data.json")

if not os.path.exists(HTML_PATH):
    print(f"SKIP: {HTML_PATH} not found")
    exit(0)

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# Stock analysis (optional)
stock_json = "[]"
if os.path.exists(KV_PATH):
    with open(KV_PATH, "r", encoding="utf-8") as f:
        stock_json = f.read()
    print(f"  stock_analysis: {len(stock_json)} chars")
else:
    print("  stock_analysis: NOT FOUND, using []")

html = html.replace("__STOCK_DATA_PLACEHOLDER__", stock_json)

# K-line (optional)
kline_json = "{}"
if os.path.exists(KLINE_PATH):
    with open(KLINE_PATH, "r", encoding="utf-8") as f:
        kline_json = f.read()
    print(f"  kline_data: {len(kline_json)} chars")
else:
    print("  kline_data: NOT FOUND, using {}")

html = html.replace("__KLINE_DATA_PLACEHOLDER__", kline_json)

for name in ["index.html", "quant_dashboard.html"]:
    out = os.path.join("outputs", name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

print(f"OK -> {os.path.getsize(HTML_PATH)//1024}KB")