"""
模拟数据生成器 - 当 AKShare 不可用时生成逼真的A股数据
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_mock_data(n_stocks=50, n_days=500, seed=42):
    """生成模拟A股行情数据，具有真实的市场特征"""
    np.random.seed(seed)
    
    dates = pd.date_range(end=datetime(2026, 6, 20), periods=n_days, freq="B")
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    
    records = []
    for code in codes:
        # 随机初始价格 (5-200)
        price = np.random.uniform(5, 200)
        # 随机波动率 (15%-60% 年化)
        annual_vol = np.random.uniform(0.15, 0.60)
        daily_vol = annual_vol / np.sqrt(252)
        # 随机漂移 (-20% to +30% 年化)
        drift = np.random.uniform(-0.20, 0.30) / 252
        
        for i, date in enumerate(dates):
            ret = np.random.normal(drift, daily_vol)
            # 加入自相关 (momentum effect)
            if i > 0:
                ret += np.random.normal(0, 0.3) * np.sign(records[-1][4] / records[-2][4] - 1) if i > 1 else 0
            
            price *= (1 + ret)
            price = max(price, 1.0)
            
            high = price * (1 + abs(np.random.normal(0, daily_vol * 0.5)))
            low = price * (1 - abs(np.random.normal(0, daily_vol * 0.5)))
            open_price = low + np.random.random() * (high - low)
            volume = np.random.lognormal(15, 1.5)
            amount = volume * price
            turnover = np.random.uniform(0.1, 10.0)
            
            records.append([date, code, open_price, high, low, price, volume, amount, turnover])
    
    df = pd.DataFrame(records, columns=["date","code","open","high","low","close","volume","amount","turnover_rate"])
    df["date"] = pd.to_datetime(df["date"])
    print(f"[Mock] 生成 {n_stocks} 只 x {n_days} 天 = {len(df)} 行模拟数据")
    return df

def generate_mock_factors(daily_data):
    """基于行情计算模拟因子（加入随机噪声模拟真实世界）"""
    df = daily_data.copy()
    grouped = df.groupby("code")
    
    # 动量
    df["ret_1m"] = grouped["close"].pct_change(20) + np.random.normal(0, 0.02, len(df))
    df["ret_3m"] = grouped["close"].pct_change(60) + np.random.normal(0, 0.03, len(df))
    df["ret_6m"] = grouped["close"].pct_change(120) + np.random.normal(0, 0.04, len(df))
    df["ret_12m"] = grouped["close"].pct_change(240) + np.random.normal(0, 0.05, len(df))
    
    # 波动率
    df["vol_20d"] = grouped["close"].pct_change().rolling(20).std().values
    df["vol_60d"] = grouped["close"].pct_change().rolling(60).std().values
    df["max_drawdown_60d"] = grouped["close"].rolling(60).apply(
        lambda x: (x / x.cummax() - 1).min() if len(x) > 0 else 0
    ).values
    
    # 流动性
    df["turnover_rate"] = df.get("turnover_rate", np.random.uniform(0.5, 5, len(df)))
    df["volume_ratio_5d"] = grouped["volume"].transform(
        lambda x: x / (x.rolling(5).mean() + 1)
    )
    df["amount_20d"] = grouped["amount"].transform(lambda x: x.rolling(20).mean())
    
    # 均线
    df["ma_20"] = grouped["close"].transform(lambda x: x.rolling(20).mean())
    df["ma_60"] = grouped["close"].transform(lambda x: x.rolling(60).mean())
    df["bias_20"] = df["close"] / (df["ma_20"] + 0.01) - 1
    df["bias_60"] = df["close"] / (df["ma_60"] + 0.01) - 1
    
    # 技术指标 - 简化RSI
    delta = grouped["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    df["rsi_14"] = 100 - 100 / (1 + gain.rolling(14).mean() / (loss.rolling(14).mean() + 0.001))
    df["rsi_14"] = df.groupby("code")["rsi_14"].transform(lambda x: x)
    
    # 振幅
    df["amplitude"] = (df["high"] - df["low"]) / (df["close"].shift(1) + 0.01)
    df["amplitude_20d"] = grouped["amplitude"].transform(lambda x: x.rolling(20).mean())
    
    # 换手率变化
    df["turnover_change"] = grouped["turnover_rate"].pct_change(20)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def generate_mock_label(df, forward=20):
    """生成未来收益率标签（含部分可预测信号）"""
    np.random.seed(42)
    # 真实未来收益
    true_ret = df.groupby("code")["close"].transform(
        lambda x: x.shift(-forward) / x - 1
    )
    # 加入基于动量和波动率的可预测成分
    signal = (0.3 * df["ret_1m"].fillna(0) - 0.2 * df["vol_20d"].fillna(0))
    signal = signal.fillna(0).clip(-0.1, 0.1)
    noise = np.random.normal(0, 0.05, len(df))
    df["label"] = true_ret.fillna(0) * 0.3 + signal + noise
    return df
