# -*- coding: utf-8 -*-
import os, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR, OUTPUT_DIR, HOT_FACTORS, HOT_SECTOR_PARAMS, HOT_MODEL_PARAMS
from data.industry import INDUSTRY_MAP

CN_NAMES = {
    "bank": "银行", "securities": "证券", "insurance": "保险",
    "liquor": "白酒", "food": "食品饮料", "pharma": "医药",
    "semiconductor": "半导体", "new_energy": "新能源",
    "ai": "AI人工智能", "military": "军工",
    "power": "电力", "telecom": "通信", "consumer_elec": "消费电子", "software": "软件",
}

def build_code_to_industry():
    c2i = {}
    for ind, codes in INDUSTRY_MAP.items():
        name = CN_NAMES.get(ind, ind)
        for c in codes:
            c2i[c] = name
    return c2i

CODE_TO_INDUSTRY = build_code_to_industry()

class HotSectorEngine:
    def __init__(self, daily_data):
        self.data = daily_data.copy()
        self.data = self.data.sort_values(["code", "date"]).reset_index(drop=True)
        self.params = HOT_SECTOR_PARAMS
        self.factor_df = None

    def _assign_industry(self, df):
        df = df.copy()
        df["industry"] = df["code"].map(CODE_TO_INDUSTRY).fillna("其他")
        return df

    def compute_short_term_factors(self):
        print("[HotSector] 计算短线因子...")
        df = self.data.copy()
        df = self._assign_industry(df)
        grouped = df.groupby("code")

        df["ret_1d"] = grouped["close"].pct_change(1)
        df["ret_3d"] = grouped["close"].pct_change(3)
        df["ret_5d"] = grouped["close"].pct_change(5)

        df["vol_ratio_5d"] = grouped["volume"].transform(
            lambda x: x / x.rolling(5, min_periods=3).mean())
        df["vol_ratio_10d"] = grouped["volume"].transform(
            lambda x: x / x.rolling(10, min_periods=5).mean())

        df["high_5d"] = grouped["high"].transform(lambda x: x.rolling(5, min_periods=3).max())
        df["gap_from_5d_high"] = df["close"] / df["high_5d"] - 1

        df["ma_5"] = grouped["close"].transform(lambda x: x.rolling(5, min_periods=3).mean())
        df["ma_10"] = grouped["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
        df["bias_ma5"] = df["close"] / df["ma_5"] - 1
        df["bias_ma10"] = df["close"] / df["ma_10"] - 1

        df["turnover_change_5d"] = grouped["turnover_rate"].pct_change(5)

        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["amplitude_5d_avg"] = grouped["amplitude"].transform(
            lambda x: x.rolling(5, min_periods=3).mean())
        df["amplitude_ratio_5d"] = df["amplitude"] / df["amplitude_5d_avg"].replace(0, np.nan)

        delta = grouped["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(6, min_periods=3).mean()
        avg_loss = loss.rolling(6, min_periods=3).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df["rsi_6"] = 100 - (100 / (1 + rs))

        def count_up(close_series):
            up = (close_series.diff() > 0).astype(int)
            result = pd.Series(0, index=close_series.index)
            cnt = 0
            for i in range(len(up)):
                cnt = cnt + 1 if up.iloc[i] else 0
                result.iloc[i] = cnt
            return result
        df["consecutive_up"] = grouped["close"].transform(count_up)

        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["money_flow_pos"] = (df["close"] - df["low"]) / hl_range

        # --- 板块内相对强度 (补涨因子) ---
        # 先在个股上计算ret_5d, 后面按板块计算sector_ret_5d后会再加
        # 这里预计算, 后面prepare_data中会用到

        # --- 短期波动收敛 (蓄势待发) ---
        df["vol_5d"] = grouped["close"].pct_change().rolling(5).std().values
        df["vol_10d"] = grouped["close"].pct_change().rolling(10).std().values
        df["vol_convergence"] = df["vol_5d"] / df["vol_10d"].replace(0, np.nan)  # <1 表示波动在收敛

        # --- 距10日高点距离 (回撤空间) ---
        df["high_10d"] = grouped["high"].transform(lambda x: x.rolling(10, min_periods=5).max())
        df["gap_from_10d_high"] = df["close"] / df["high_10d"] - 1

        # --- 均量线的量价配合 ---
        df["amount_5d"] = grouped["amount"].transform(lambda x: x.rolling(5, min_periods=3).mean())

        df = df.replace([np.inf, -np.inf], np.nan)
        self.factor_df = df
        print(f"  短线因子完成: {len(df)} 行")
        return df

    def compute_sector_heat(self):
        print("[HotSector] 计算板块热度...")
        df = self.factor_df.copy()
        latest_dates = sorted(df["date"].unique())[-5:]
        recent = df[df["date"].isin(latest_dates)]

        sec = recent.groupby("industry").agg(
            sector_ret_5d=("ret_5d", "mean"),
            sector_vol_ratio=("vol_ratio_5d", "mean"),
            stock_count=("code", "nunique"),
        ).reset_index()

        min_s = self.params.get("min_stocks_per_sector", 3)
        sec = sec[sec["stock_count"] >= min_s]

        sec["sector_rank"] = sec["sector_ret_5d"].rank(ascending=False)
        sec["sector_rank_pct"] = sec["sector_rank"] / len(sec)
        sec["heat_score"] = (
            sec["sector_ret_5d"] * 0.5 +
            sec["sector_vol_ratio"] * 0.3 -
            sec["sector_rank_pct"] * 0.2
        )
        sec = sec.sort_values("heat_score", ascending=False)

        print("  Top 10 热点板块:")
        for _, row in sec.head(10).iterrows():
            print(f"    {row['industry']:<12} 5d:{row['sector_ret_5d']:>+.2%}  "
                  f"vol:{row['sector_vol_ratio']:.2f}  n:{int(row['stock_count'])}")

        self.sector_stats = sec
        return sec

    def prepare_data(self):
        df = self.factor_df.copy()
        sec = self.sector_stats[["industry","sector_ret_5d","sector_vol_ratio",
                                  "sector_rank_pct","heat_score"]]
        df = df.merge(sec, on="industry", how="left")
        for col in ["sector_ret_5d","sector_vol_ratio","sector_rank_pct"]:
            df[col] = df[col].fillna(sec[col].median() if len(sec) > 0 else 0)

        # --- 板块内相对收益率 (补涨潜力) ---
        df["sector_rel_ret_5d"] = df["ret_5d"] - df["sector_ret_5d"]
        # 负值越大 = 越落后板块 = 补涨潜力越大
        df["upside_potential"] = df["sector_ret_5d"] - df["ret_5d"]
        
        # --- RSI 在40-60区间最优 (非超买超卖, 有运动空间) ---
        df["rsi_sweet_spot"] = 1.0 - abs(df["rsi_6"] - 50) / 50

        fwd = self.params.get("forward_days", 5)
        df["label"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-fwd) / x - 1)

        self.full_df = df
        return df

class HotSectorModel:
    def __init__(self, params=None):
        self.params = params or HOT_MODEL_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def prepare_training_data(self, df, factor_list, test_size=0.15):
        print("[HotModel] 准备训练数据...")
        use_cols = ["date", "code", "label"] + factor_list
        df = df[use_cols].dropna(subset=["label"] + factor_list).copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=factor_list)

        for col in factor_list:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df = df[(df[col] > mean - 5*std) & (df[col] < mean + 5*std)]

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
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.feature_names = factor_list
        print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")
        return X_train_s, y_train, X_test_s, y_test, test

    def train(self, df, factor_list):
        X_train, y_train, X_test, y_test, test_df = self.prepare_training_data(df, factor_list)

        print("[HotModel] 训练 LightGBM 短线模型...")
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(30)]
        )

        y_pred = self.model.predict(X_test)
        ic = spearmanr(y_test, y_pred)[0]
        print(f"  测试集 Rank IC: {ic:.4f}")

        importance = pd.DataFrame({
            "factor": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\n  Top 10 因子重要度:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['factor']:<25} {row['importance']:.4f}")

        return importance

    def predict(self, df, factor_list):
        if self.model is None:
            raise ValueError("模型未训练")
        pred_df = df.dropna(subset=factor_list).copy()
        X = self.scaler.transform(pred_df[factor_list].values)
        pred_df["score"] = self.model.predict(X)
        pred_df["rank"] = pred_df.groupby("date")["score"].rank(ascending=False, pct=True)
        return pred_df

    def select_top_picks(self, df, factor_list, top_sectors=5, top_n_per_sector=6):
        pred_df = self.predict(df, factor_list)
        latest_date = pred_df["date"].max()
        latest = pred_df[pred_df["date"] == latest_date].copy()

        sector_heat = latest.groupby("industry")["sector_ret_5d"].mean().sort_values(ascending=False)
        hot_sectors = sector_heat.head(top_sectors).index.tolist()

        print(f"\n  热点板块 TOP {top_sectors}:")
        for ind in hot_sectors:
            avg_ret = sector_heat[ind]
            cnt = len(latest[latest["industry"] == ind])
            print(f"    {ind:<12} 5d涨:{avg_ret:>+.2%}  可选:{cnt}只")

        picks = []
        for industry in hot_sectors:
            sector_stocks = latest[latest["industry"] == industry]
            top_stocks = sector_stocks.nlargest(top_n_per_sector, "score")
            picks.append(top_stocks)

        result = pd.concat(picks, ignore_index=True)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)

        print(f"\n  最终选股 ({len(result)} 只):")
        for i, (_, row) in enumerate(result.iterrows()):
            print(f"    {i+1:>3}. {row['code']:<10} {row.get('industry',''):<12} "
                  f"得分:{row['score']:.4f}  5d:{row.get('ret_5d',0):>+.2%}")

        return result

    def save(self, name="hot_sector_model"):
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)
        print(f"[HotModel] 模型已保存: {path}")

    def load(self, name="hot_sector_model"):
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        print(f"[HotModel] 模型已加载: {path}")

def run_hot_sector_pipeline(fetcher, use_all_stocks=False):
    if use_all_stocks:
        try:
            codes = fetcher.get_all_stocks()
        except Exception:
            print("[HotSector] all stocks failed, fallback to HS300")
            codes = fetcher.get_stock_list()
    else:
        codes = fetcher.get_stock_list()
    print("\n" + "=" * 60)
    print("  热点板块短线选股系统")
    print("=" * 60)

    
    # 短线模型只需近期数据, 减少下载量
    import datetime
    end_dt = datetime.date.today().strftime("%Y%m%d")
    start_dt = (datetime.date.today() - datetime.timedelta(days=180)).strftime("%Y%m%d")
    print(f"[HotSector] 下载 {start_dt} ~ {end_dt} 行情...")
    daily = fetcher.get_daily_data(codes, start=start_dt, end=end_dt)
    if len(daily) == 0:
        print("[Error] 行情数据获取失败")
        return None, None, None

    engine = HotSectorEngine(daily)
    engine.compute_short_term_factors()
    engine.compute_sector_heat()
    df = engine.prepare_data()

    model = HotSectorModel()
    factor_list = [f for f in HOT_FACTORS if f in df.columns]
    print(f"\n  可用因子: {len(factor_list)} 个")
    importance = model.train(df, factor_list)
    model.save()

    picks = model.select_top_picks(df, factor_list)

    picks_cols = ["code", "industry", "score", "ret_5d", "vol_ratio_5d",
                  "rsi_6", "consecutive_up", "gap_from_5d_high"]
    available_cols = [c for c in picks_cols if c in picks.columns]
    picks_path = os.path.join(OUTPUT_DIR, "hot_sector_picks.csv")
    picks[available_cols].to_csv(picks_path, index=False, encoding="utf-8-sig")
    print(f"\n  选股结果已保存: {picks_path}")

    sector_path = os.path.join(OUTPUT_DIR, "sector_heat.csv")
    engine.sector_stats.to_csv(sector_path, index=False, encoding="utf-8-sig")
    print(f"  板块热度已保存: {sector_path}")

    return picks, engine.sector_stats, importance
