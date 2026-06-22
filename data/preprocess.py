"""
数据预处理 - 缺失值填补、中性化、标准化
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def winsorize(series, limits=(0.01, 0.99)):
    """缩尾处理"""
    q_low, q_high = series.quantile(limits[0]), series.quantile(limits[1])
    return series.clip(q_low, q_high)

def fill_na_by_group(df, col, group_col="code", method="ffill"):
    """按组填充缺失值"""
    if method == "ffill":
        df[col] = df.groupby(group_col)[col].ffill()
    elif method == "median":
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.fillna(x.median())
        )
    return df

def neutralize_factors(df, factor_cols, neutral_col="market_cap"):
    """市值中性化"""
    for col in factor_cols:
        if col in df.columns and neutral_col in df.columns:
            X = df[[neutral_col]].values
            y = df[col].values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            if mask.sum() > 10:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(X[mask], y[mask])
                residuals = y - reg.predict(X)
                df.loc[:, col + "_neutral"] = np.nan
                df.loc[mask, col + "_neutral"] = residuals
    return df

def standardize_cross_section(df, factor_cols):
    """截面标准化"""
    for col in factor_cols:
        if col in df.columns:
            df[col] = df.groupby("date")[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
    return df

