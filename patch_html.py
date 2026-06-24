# -*- coding: utf-8 -*-
"""Inject stock_analysis.json + kline_data.json into HTML"""
import json, os

HTML_PATH = os.path.join("outputs", "index.html")
KV_PATH = os.path.join("outputs", "stock_analysis.json")
KLINE_PATH = os.path.join("outputs", "kline_data.json")

for path in [HTML_PATH, KV_PATH]:
    if not os.path.exists(path):
        print(f"SKIP: {path} not found")
        exit(1)

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

with open(KV_PATH, "r", encoding="utf-8") as f:
    stock_json = f.read()
html = html.replace("__STOCK_DATA_PLACEHOLDER__", stock_json if stock_json.strip() else "[]")

kline_json = "{}"
if os.path.exists(KLINE_PATH):
    with open(KLINE_PATH, "r", encoding="utf-8") as f:
        kline_json = f.read()
html = html.replace("__KLINE_DATA_PLACEHOLDER__", kline_json)

for name in ["index.html", "quant_dashboard.html"]:
    out = os.path.join("outputs", name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

print(f"OK patched: stock={len(stock_json)} kline={len(kline_json)} -> {os.path.getsize(HTML_PATH)//1024}KB")