"""
模型训练模块 - LightGBM 多因子选股
"""
import os, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from config import MODEL_PARAMS, OUTPUT_DIR

class StockRankingModel:
    def __init__(self, params=None):
        self.params = params or MODEL_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def prepare_data(self, factor_df, factor_list, test_size=0.2):
        print(f"[Train] 准备数据...")
        df = factor_df.dropna(subset=["label"] + factor_list).copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=factor_list)
        for col in factor_list:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df = df[(df[col] > mean - 4*std) & (df[col] < mean + 4*std)]
        print(f"  有效样本: {len(df)}")
        dates = sorted(df["date"].unique())
        split_idx = int(len(dates) * (1 - test_size))
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        train = df[df["date"].isin(train_dates)]
        test = df[df["date"].isin(test_dates)]
        X_train = train[factor_list].values
        y_train = train["label"].values
        X_test = test[factor_list].values
        y_test = test["label"].values
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.feature_names = factor_list
        print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
        return X_train, y_train, X_test, y_test, test
    
    def train(self, factor_df, factor_list):
        X_train, y_train, X_test, y_test, test_df = self.prepare_data(factor_df, factor_list)
        
        print(f"[Train] LightGBM 训练中...")
        # 使用 LGBMRegressor 而非 LGBMRanker（连续标签）
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        y_pred = self.model.predict(X_test)
        ic = spearmanr(y_test, y_pred)[0]
        print(f"  测试集 Rank IC: {ic:.4f}")
        
        importance = pd.DataFrame({
            "factor": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        print("\n  Top 10 因子:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['factor']}: {row['importance']:.4f}")
        
        return importance
    
    def predict(self, factor_df, factor_list):
        if self.model is None:
            raise ValueError("模型未训练")
        df = factor_df.dropna(subset=factor_list).copy()
        X = self.scaler.transform(df[factor_list].values)
        df["score"] = self.model.predict(X)
        df["rank"] = df.groupby("date")["score"].rank(ascending=False, pct=True)
        return df
    
    def save(self, name="stock_model"):
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)
        print(f"[Train] 模型已保存: {path}")
    
    def load(self, name="stock_model"):
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        print(f"[Train] 模型已加载: {path}")
