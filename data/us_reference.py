# -*- coding: utf-8 -*-
# 中美科技股联动映射表
# 中国股票 -> (美股参考代码, 公司名, 关联逻辑)

US_TECH_MAP = {
    # 光纤通信
    "600105": [("GLW", "Corning", "光纤光缆龙头"), ("LITE", "Lumentum", "光器件")],
    "600487": [("GLW", "Corning", "光纤光缆"), ("LITE", "Lumentum", "光模块上游")],
    "300308": [("LITE", "Lumentum", "光模块"), ("COHR", "Coherent", "激光器")],
    "300502": [("LITE", "Lumentum", "光模块"), ("GLW", "Corning", "光纤")],
    "300394": [("LITE", "Lumentum", "光器件"), ("COHR", "Coherent", "光学")],

    # 通信设备
    "000063": [("CSCO", "Cisco", "通信设备"), ("ERIC", "Ericsson", "5G设备")],
    "600498": [("CSCO", "Cisco", "通信设备"), ("JNPR", "Juniper", "网络设备")],

    # 半导体设备
    "002371": [("AMAT", "Applied Materials", "半导体设备"), ("LRCX", "Lam Research", "刻蚀设备")],
    "688012": [("AMAT", "Applied Materials", "半导体设备"), ("LRCX", "Lam Research", "刻蚀")],
    "688981": [("TSM", "TSMC", "晶圆代工"), ("INTC", "Intel", "芯片制造")],
    "688041": [("AMD", "AMD", "芯片设计"), ("NVDA", "NVIDIA", "GPU/AI")],

    # 消费电子
    "002475": [("AAPL", "Apple", "消费电子"), ("MSFT", "Microsoft", "科技")],
    "601138": [("AAPL", "Apple", "供应链"), ("DELL", "Dell", "服务器")],
    "603501": [("QCOM", "Qualcomm", "芯片"), ("AVGO", "Broadcom", "射频")],

    # AI/软件
    "002230": [("MSFT", "Microsoft", "AI"), ("GOOGL", "Google", "AI")],
    "688111": [("MSFT", "Microsoft", "办公软件"), ("ADBE", "Adobe", "软件")],

    # 新能源汽车
    "002594": [("TSLA", "Tesla", "电动车"), ("RIVN", "Rivian", "电动车")],
    "300750": [("TSLA", "Tesla", "电池客户"), ("LCID", "Lucid", "电动车")],

    # 互联网
    "00700": [("META", "Meta", "社交"), ("GOOGL", "Google", "互联网")],
    "09988": [("AMZN", "Amazon", "电商云"), ("MSFT", "Microsoft", "云服务")],
}

# 美股ETF参考 (行业级别)
US_SECTOR_ETF = {
    "半导体": "SMH",      # VanEck Semiconductor ETF
    "通信": "XLC",        # Communication Services
    "电子": "XLK",        # Technology Select Sector
    "AI": "QQQ",          # Nasdaq 100 (tech heavy)
    "新能源": "TAN",      # Invesco Solar ETF
    "消费电子": "XLK",    # Tech
    "军工": "ITA",        # Aerospace & Defense
    "医药": "XLV",        # Health Care
    "金融": "XLF",        # Financials
}
