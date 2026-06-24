# -*- coding: utf-8 -*-
"""Inject stock_analysis.json + kline_data.json into HTML"""
import json, os

HTML_PATH = r"D:\quant_a_stock\outputs\index.html"
KV_PATH = r"D:\quant_a_stock\outputs\stock_analysis.json"
KLINE_PATH = r"D:\quant_a_stock\outputs\kline_data.json"

for path in [HTML_PATH, KV_PATH]:
    if not os.path.exists(path):
        print(f"SKIP: {path} not found")
        exit(1)

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# Inject stock analysis
with open(KV_PATH, "r", encoding="utf-8") as f:
    stock_json = f.read()

html = html.replace("__STOCK_DATA_PLACEHOLDER__", stock_json if stock_json.strip() else "[]")

# Inject K-line data (or empty)
kline_json = "{}"
if os.path.exists(KLINE_PATH):
    with open(KLINE_PATH, "r", encoding="utf-8") as f:
        kline_json = f.read()
    if os.path.getsize(KLINE_PATH) > 5 * 1024 * 1024:
        print("[WARN] K-line data > 5MB, using empty to save load time")
        kline_json = "{}"

html = html.replace("__KLINE_DATA_PLACEHOLDER__", kline_json)

# Save both output files
for name in ["index.html", "quant_dashboard.html"]:
    out = os.path.join(r"D:\quant_a_stock\outputs", name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

print(f"[OK] Patched: stock={len(stock_json)} chars, kline={len(kline_json)} chars -> {os.path.getsize(HTML_PATH)//1024}KB")