"""
多因子计算模块
价值 / 动量 / 质量 / 波动率 / 流动性 / 成长因子
"""
import pandas as pd
import numpy as np
from config import FACTOR_GROUPS

class FactorEngine:
    def __init__(self, daily_data):
        self.data = daily_data.copy()
        self.data = self.data.sort_values(["code", "date"]).reset_index(drop=True)
    
    def compute_all_factors(self):
        """计算所有因子，返回截面因子数据"""
        print("[Factors] 计算因子...")
        df = self.data.copy()
        
        # 按股票分组计算
        grouped = df.groupby("code")
        
        # --- 动量因子 ---
        df["ret_1m"] = grouped["close"].pct_change(20)
        df["ret_3m"] = grouped["close"].pct_change(60)
        df["ret_6m"] = grouped["close"].pct_change(120)
        df["ret_12m"] = grouped["close"].pct_change(240)
        
        # --- 波动率因子 ---
        df["vol_20d"] = grouped["close"].pct_change().rolling(20).std().values
        df["vol_60d"] = grouped["close"].pct_change().rolling(60).std().values
        df["max_drawdown_60d"] = grouped["close"].rolling(60).apply(
            lambda x: (x / x.cummax() - 1).min()
        ).values
        
        # --- 流动性因子 ---
        df["turnover_rate"] = df.get("turnover_rate", np.nan)
        df["volume_ratio_5d"] = grouped["volume"].transform(
            lambda x: x / x.rolling(5).mean()
        )
        df["amount_20d"] = grouped["amount"].transform(
            lambda x: x.rolling(20).mean()
        )
        
        # --- 均线偏离 ---
        df["ma_20"] = grouped["close"].transform(lambda x: x.rolling(20).mean())
        df["ma_60"] = grouped["close"].transform(lambda x: x.rolling(60).mean())
        df["bias_20"] = df["close"] / df["ma_20"] - 1
        df["bias_60"] = df["close"] / df["ma_60"] - 1
        
        # --- 换手率变化 ---
        df["turnover_change"] = grouped["turnover_rate"].pct_change(20)
        
        # --- 振幅 ---
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["amplitude_20d"] = grouped["amplitude"].transform(lambda x: x.rolling(20).mean())
        
        # --- RSI ---
        delta = grouped["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        df["rsi_14"] = df.groupby("code")["rsi_14"].transform(lambda x: x)
        
        # 清理
        df = df.replace([np.inf, -np.inf], np.nan)
        
        self.factor_df = df
        print(f"  因子计算完成: {df.shape[0]} 行 x {df.shape[1]} 列")
        return df
    
    def prepare_labels(self, forward_period=20):
        """准备标签: 未来N日收益率"""
        df = self.factor_df.copy()
        df["label"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-forward_period) / x - 1
        )
        return df
    
    def get_factor_list(self):
        """获取所有使用的因子名"""
        factors = []
        for group, names in FACTOR_GROUPS.items():
            factors.extend(names)
        # 额外因子
        extra = ["bias_20", "bias_60", "turnover_change", "amplitude_20d", "rsi_14"]
        factors.extend(extra)
        # 只保留存在的列
        if self.factor_df is not None:
            factors = [f for f in factors if f in self.factor_df.columns]
        return factors
