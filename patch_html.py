import json, re, os

# Read stock data
with open(r"D:\quant_a_stock\outputs\stock_analysis.json", "r", encoding="utf-8") as f:
    stock_data = json.load(f)

stock_json = json.dumps(stock_data, ensure_ascii=False)

# Read build_html.py
with open(r"D:\quant_a_stock\build_html.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add tab button
old_btn = '<button class=\"tab-btn\" data-tab=\"about\">'
new_btn = '<button class=\"tab-btn\" data-tab=\"search\">🔍 个股分析</button>\n  <button class=\"tab-btn\" data-tab=\"about\">'
content = content.replace(old_btn, new_btn)

# 2. Add stat card
old_card = '<div class=\"stat-card\"><div class=\"label\">📦 股票池</div>'
new_card = '<div class=\"stat-card\"><div class=\"label\">🔍 可查个股</div><div class=\"value\">300</div><div class=\"desc\">预分析股票数</div></div>\n  <div class=\"stat-card\"><div class=\"label\">📦 股票池</div>'
content = content.replace(old_card, new_card)

# 3. Add search tab content
search_tab = '''
  <div class="tab-content" id="tab-search">
    <div class="summary">🔍 输入股票代码 (如 600519) 查询短线/中线走势分析 | 基于HS300实时因子评分</div>
    <div class="card">
      <h3>🎯 个股走势分析</h3>
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap">
        <input type="text" id="stockInput" placeholder="输入股票代码, 如 600519" 
          style="flex:1;min-width:200px;max-width:300px;padding:10px 16px;background:#0b1625;border:1px solid #1e3a5f;border-radius:8px;color:#e2e8f0;font-size:14px;font-family:monospace"
          onkeyup="searchStock()">
        <select id="stockSelect" onchange="selectStock()" 
          style="padding:10px 12px;background:#0b1625;border:1px solid #1e3a5f;border-radius:8px;color:#e2e8f0;font-size:13px;max-width:240px">
          <option value="">热门股票...</option>
        </select>
      </div>
      <div id="stockResult" style="min-height:200px">
        <div class="empty">输入股票代码查看分析结果</div>
      </div>
    </div>
    <div class="card">
      <h3>📋 强烈推荐个股</h3>
      <div id="topStocksTable" class="table-wrap"></div>
    </div>
  </div>
'''

old_about = '  <div class="tab-content" id="tab-about">'
content = content.replace(old_about, search_tab + '\n' + old_about)

# 4. Add JavaScript
search_js = f'''
var STOCK_DATA = {stock_json};
var currentStock = null;

function searchStock() {{
  var input = document.getElementById("stockInput").value.trim().toUpperCase();
  var result = document.getElementById("stockResult");
  if (!input) {{ showTopPicks(); return; }}
  var found = null;
  for (var i = 0; i < STOCK_DATA.length; i++) {{
    var d = STOCK_DATA[i];
    if (d.code === input || d.code.indexOf(input) === 0) {{ found = d; break; }}
  }}
  if (found) {{
    document.getElementById("stockSelect").value = found.code;
    showAnalysis(found);
  }} else {{
    result.innerHTML = '<div class="empty">未找到股票: ' + input + '</div>';
  }}
}}

function selectStock() {{
  var code = document.getElementById("stockSelect").value;
  if (!code) return;
  document.getElementById("stockInput").value = code;
  searchStock();
}}

function showAnalysis(d) {{
  var score = d.total_score;
  var sc = score > 25 ? "#22c55e" : score > 18 ? "#38bdf8" : score > 10 ? "#f59e0b" : score > 3 ? "#f97316" : "#ef4444";
  var ret5c = d.ret_5d >= 0 ? "up" : "down";
  var rsi = d.rsi_14;
  var rsiColor = rsi > 70 ? "#ef4444" : rsi < 30 ? "#22c55e" : "#38bdf8";
  
  var sigs = (d.signals || []).map(function(s) {{ 
    return "<span class='badge " + (s.indexOf("超卖")>=0||s.indexOf("超跌")>=0?"mild":s.indexOf("超买")>=0?"hot":"cold") + "'>" + s + "</span>";
  }}).join(" ");
  
  document.getElementById("stockResult").innerHTML = 
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px">' +
    '<div class="stat-card"><div class="label">综合评分</div><div class="value" style="color:' + sc + '">' + score + '</div><div class="desc">' + d.rating + '</div></div>' +
    '<div class="stat-card"><div class="label">5日涨跌</div><div class="value ' + ret5c + '">' + (d.ret_5d*100).toFixed(2) + '%</div><div class="desc">' + d.short_trend + '</div></div>' +
    '<div class="stat-card"><div class="label">中线趋势</div><div class="value" style="color:' + (d.ret_1m>=0?"#22c55e":"#ef4444") + '">' + (d.ret_1m*100).toFixed(1) + '%</div><div class="desc">月涨幅 | ' + d.mid_trend + '</div></div>' +
    '<div class="stat-card"><div class="label">RSI(14)</div><div class="value" style="color:' + rsiColor + '">' + rsi.toFixed(1) + '</div><div class="desc">' + (rsi>70?"超买区":rsi<30?"超卖区":"正常区") + '</div></div>' +
    '</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-top:14px">' +
    '<div class="stat-card"><div class="label">量比(5日)</div><div class="value" style="font-size:16px;color:#38bdf8">' + d.vol_ratio_5d.toFixed(2) + '</div></div>' +
    '<div class="stat-card"><div class="label">距20日均线</div><div class="value" style="font-size:16px;color:' + (d.bias_20>=0?"#22c55e":"#ef4444") + '">' + (d.bias_20*100).toFixed(1) + '%</div></div>' +
    '<div class="stat-card"><div class="label">振幅</div><div class="value" style="font-size:16px;color:#f59e0b">' + (d.amplitude*100).toFixed(1) + '%</div></div>' +
    '<div class="stat-card"><div class="label">换手率</div><div class="value" style="font-size:16px;color:#94a3b8">' + d.turnover_rate.toFixed(2) + '%</div></div>' +
    '</div>' +
    '<div style="margin-top:14px;padding:12px;background:#0b1625;border-radius:8px;border:1px solid #1e3a5f">' +
    '<span style="color:#94a3b8;font-size:12px">技术信号: </span>' + sigs +
    '<span style="color:#64748b;font-size:11px;margin-left:12px">更新: ' + d.update_time + '</span></div>';
}}

function showTopPicks() {{
  var top = [];
  for (var i = 0; i < STOCK_DATA.length; i++) {{
    if (STOCK_DATA[i].total_score > 20) top.push(STOCK_DATA[i]);
  }}
  top.sort(function(a,b) {{ return b.total_score - a.total_score; }});
  
  var rows = "";
  for (var i = 0; i < Math.min(top.length, 30); i++) {{
    var d = top[i];
    rows += "<tr onclick=\"showStock('" + d.code + "')\" style=\"cursor:pointer\">" +
      "<td class=\"rank\">" + (i+1) + "</td>" +
      "<td class=\"code\">" + d.code + "</td>" +
      "<td class=\"" + (d.ret_5d>=0?"up":"down") + "\">" + (d.ret_5d*100).toFixed(2) + "%</td>" +
      "<td>" + d.short_trend + "</td>" +
      "<td>" + d.mid_trend + "</td>" +
      "<td><span class=\"badge " + (d.rating==="强烈推荐"?"hot":"warm") + "\">" + d.rating + "</span></td>" +
      "<td><div class=\"bar-wrap\"><div class=\"bar\" style=\"width:" + Math.min(d.total_score/30*100,100) + "%;background:#22c55e\"></div><span>" + d.total_score + "</span></div></td>" +
      "</tr>";
  }}
  
  document.getElementById("topStocksTable").innerHTML = "<table class=\"table\"><thead><tr>" +
    "<th>#</th><th>代码</th><th>5日涨幅</th><th>短线趋势</th><th>中线趋势</th><th>评级</th><th>评分</th>" +
    "</tr></thead><tbody>" + rows + "</tbody></table>";
  
  var sel = document.getElementById("stockSelect");
  sel.innerHTML = "<option value=\"\">热门股票...</option>";
  for (var i = 0; i < Math.min(top.length, 20); i++) {{
    sel.innerHTML += "<option value=\"" + top[i].code + "\">" + top[i].code + "</option>";
  }}
}}

function showStock(code) {{
  document.getElementById("stockInput").value = code;
  searchStock();
}}

showTopPicks();
'''

# Insert before tab switching
old_js = 'document.querySelectorAll(".tab-btn")'
content = content.replace(old_js, search_js + '\n\n' + old_js)

# Write back
with open(r"D:\quant_a_stock\build_html.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ build_html.py updated with stock search tab")
print(f"   Stock data embedded: {len(stock_json)} chars")