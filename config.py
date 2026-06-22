"""
A股多因子量化选股系统 - 配置文件
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCK_POOL = "hs300"

# 时间范围 - 2023年起以适配 baostock 串行下载速度
START_DATE = "20230101"
END_DATE = "20260601"

FACTOR_GROUPS = {
    "value": ["pe_ttm", "pb", "ps_ttm", "pcf_ttm"],
    "quality": ["roe", "roa", "gross_profit_margin", "debt_to_asset"],
    "momentum": ["ret_1m", "ret_3m", "ret_6m", "ret_12m"],
    "volatility": ["vol_20d", "vol_60d", "max_drawdown_60d"],
    "liquidity": ["turnover_rate", "volume_ratio_5d", "amount_20d"],
    "growth": ["revenue_yoy", "profit_yoy", "eps_yoy"],
}

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

BACKTEST_PARAMS = {
    "top_n": 30,
    "rebalance_freq": "M",
    "commission": 0.0003,
    "slippage": 0.001,
    "benchmark": "000300",
}

# === ?????????? ===
HOT_SECTOR_PARAMS = {
    "lookback_days": 5,
    "forward_days": 5,
    "top_sectors": 5,
    "top_n_per_sector": 6,
    "min_stocks_per_sector": 3,
    "sector_weight": 0.4,
    "stock_weight": 0.6,
}

HOT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

HOT_FACTORS = [
    # 动量 (适度动量, 不追高)
    "ret_1d", "ret_3d", "ret_5d",
    # 成交量
    "vol_ratio_5d", "vol_ratio_10d",
    # 价格位置
    "gap_from_5d_high", "gap_from_10d_high",
    "bias_ma5", "bias_ma10",
    # 换手率变化
    "turnover_change_5d",
    # 波动
    "amplitude_ratio_5d", "vol_convergence",
    # 技术指标
    "rsi_6", "rsi_sweet_spot",
    "consecutive_up", "money_flow_pos",
    # 板块因子
    "sector_ret_5d", "sector_vol_ratio", "sector_rank_pct",
    # 上涨空间因子 (补涨逻辑)
    "sector_rel_ret_5d", "upside_potential",
]
