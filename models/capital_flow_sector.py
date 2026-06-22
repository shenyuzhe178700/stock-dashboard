# -*- coding: utf-8 -*-
import os, pickle
import numpy as np
import pandas as pd
from config import OUTPUT_DIR, DATA_DIR

class CapitalFlowSectorSelector:
    """资金认可度 + 上涨空间 双维度板块筛选器"""

    def __init__(self, daily_data, industry_map):
        self.data = daily_data.copy()
        self.data = self.data.sort_values(["code", "date"]).reset_index(drop=True)
        self._assign_industry(industry_map)
        self.sector_scores = None
        self.stock_scores = None

    def _assign_industry(self, code_to_industry):
        df = self.data
        df["industry"] = df["code"].map(code_to_industry).fillna("其他")
        self.data = df

    def compute(self):
        """主流程: 计算所有因子并评分"""
        print("[CapitalFlow] 计算资金认可度 + 上涨空间因子...")
        df = self.data.copy()
        grouped = df.groupby("code")

        # ===== 个股级因子 =====
        # 动量
        df["ret_1d"] = grouped["close"].pct_change(1)
        df["ret_3d"] = grouped["close"].pct_change(3)
        df["ret_5d"] = grouped["close"].pct_change(5)
        df["ret_10d"] = grouped["close"].pct_change(10)

        # 资金流
        df["vol_ratio_5d"] = grouped["volume"].transform(lambda x: x / x.rolling(5).mean())
        df["amount_ratio_5d"] = grouped["amount"].transform(lambda x: x / x.rolling(5).mean())

        # 涨跌量: 上涨日量/下跌日量 (最近10天)
        up_vol = np.where(df["close"] > df["close"].shift(1), df["volume"], 0)
        down_vol = np.where(df["close"] < df["close"].shift(1), df["volume"], 0)
        df["up_vol_10d"] = pd.Series(up_vol).rolling(10).sum()
        df["down_vol_10d"] = pd.Series(down_vol).rolling(10).sum()
        df["up_down_vol_ratio"] = df["up_vol_10d"] / df["down_vol_10d"].replace(0, 1)

        # 净资金流代理 (涨跌量差 / 总量)
        df["net_capital_flow"] = (df["up_vol_10d"] - df["down_vol_10d"]) / (
            df["up_vol_10d"] + df["down_vol_10d"]).replace(0, 1)

        # 价格位置
        df["high_10d"] = grouped["high"].transform(lambda x: x.rolling(10).max())
        df["high_20d"] = grouped["high"].transform(lambda x: x.rolling(20).max())
        df["gap_from_10d_high"] = df["close"] / df["high_10d"] - 1
        df["gap_from_20d_high"] = df["close"] / df["high_20d"] - 1

        # RSI(6) 和 RSI(14)
        delta = grouped["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain6 = gain.rolling(6).mean()
        avg_loss6 = loss.rolling(6).mean()
        rs6 = avg_gain6 / avg_loss6.replace(0, 1e-10)
        df["rsi_6"] = 100 - (100 / (1 + rs6))
        avg_gain14 = gain.rolling(14).mean()
        avg_loss14 = loss.rolling(14).mean()
        rs14 = avg_gain14 / avg_loss14.replace(0, 1e-10)
        df["rsi_14"] = 100 - (100 / (1 + rs14))

        # 振幅 / 换手
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["amplitude_5d"] = grouped["amplitude"].transform(lambda x: x.rolling(5).mean())
        df["turnover_accel"] = grouped["turnover_rate"].pct_change(5)

        # MA偏离
        df["ma_10"] = grouped["close"].transform(lambda x: x.rolling(10).mean())
        df["bias_ma10"] = df["close"] / df["ma_10"] - 1

        df = df.replace([np.inf, -np.inf], np.nan)

        # ===== 板块级聚合 =====
        print("[CapitalFlow] 聚合板块因子...")
        latest_date = df["date"].max()
        recent_5d = df[df["date"] >= df["date"].unique()[-5]]

        sec = recent_5d.groupby("industry").agg(
            # 资金认可度因子
            avg_vol_ratio=("vol_ratio_5d", "mean"),
            avg_amount_ratio=("amount_ratio_5d", "mean"),
            avg_up_down_ratio=("up_down_vol_ratio", "mean"),
            avg_net_capital_flow=("net_capital_flow", "mean"),
            avg_turnover_accel=("turnover_accel", "mean"),
            # 动量因子
            sector_ret_5d=("ret_5d", "mean"),
            sector_ret_1d=("ret_1d", "mean"),
            # 上涨空间因子
            avg_gap_from_20d_high=("gap_from_20d_high", "mean"),
            avg_gap_from_10d_high=("gap_from_10d_high", "mean"),
            avg_rsi_6=("rsi_6", "mean"),
            avg_rsi_14=("rsi_14", "mean"),
            # 振幅
            avg_amplitude=("amplitude", "mean"),
            avg_bias_ma10=("bias_ma10", "mean"),
            # 样本数
            stock_count=("code", "nunique"),
        ).reset_index()

        # 过滤股票数不足的板块
        sec = sec[sec["stock_count"] >= 3]

        # ===== 计算综合评分 =====
        sec = self._score_sectors(sec)

        self.sector_scores = sec
        self.detail_df = df
        return sec

    def _score_sectors(self, sec):
        """双维度评分: 资金认可度 + 上涨空间"""
        # 标准化各因子到 0~1
        for col in ["avg_vol_ratio", "avg_amount_ratio", "avg_up_down_ratio",
                     "avg_net_capital_flow", "avg_turnover_accel"]:
            s = sec[col].fillna(0)
            if s.std() > 0:
                sec[col + "_z"] = (s - s.min()) / (s.max() - s.min() + 1e-10)
            else:
                sec[col + "_z"] = 0.5

        # === 资金认可度得分 (55%) ===
        sec["capital_score"] = (
            sec["avg_vol_ratio_z"] * 0.20 +
            sec["avg_amount_ratio_z"] * 0.15 +
            sec["avg_up_down_ratio_z"] * 0.30 +
            sec["avg_net_capital_flow_z"] * 0.25 +
            sec["avg_turnover_accel_z"] * 0.10
        )

        # === 上涨空间得分 (45%) ===
        # RSI甜点位: 40-65最佳, 偏离扣分
        sec["rsi_sweet"] = 1.0 - abs(sec["avg_rsi_6"] - 52.5) / 50
        sec["rsi_sweet"] = sec["rsi_sweet"].clip(0, 1)

        # 距20日高点距离: 越负越有空间, 但太负意味着弱势
        sec["gap_score"] = (-sec["avg_gap_from_20d_high"] * 3).clip(-0.2, 0.8) + 0.2
        sec["gap_score"] = sec["gap_score"].clip(0, 1)

        # 动量惩罚: 涨幅过大扣分
        sec["momentum_penalty"] = np.where(
            sec["sector_ret_5d"] > 0.08,  # 5日涨超8%扣分
            np.maximum(0, 1 - (sec["sector_ret_5d"] - 0.08) * 5),
            1.0
        )
        sec["momentum_penalty"] = sec["momentum_penalty"].clip(0, 1)

        # 振幅: 适中最好
        sec["amp_score"] = 1.0 - abs(sec["avg_amplitude"] - 0.03) / 0.06
        sec["amp_score"] = sec["amp_score"].clip(0, 1)

        sec["upside_score"] = (
            sec["rsi_sweet"] * 0.35 +
            sec["gap_score"] * 0.30 +
            sec["momentum_penalty"] * 0.20 +
            sec["amp_score"] * 0.15
        )

        # === 综合 ===
        sec["total_score"] = (
            sec["capital_score"] * 0.55 +
            sec["upside_score"] * 0.45
        )

        # 过滤: 必须有一定正向动量 (非下跌趋势)
        sec = sec[sec["sector_ret_5d"] > -0.03]

        sec = sec.sort_values("total_score", ascending=False)

        # 打印
        print("\n" + "=" * 80)
        print("  板块资金认可度 + 上涨空间 综合排名")
        print("=" * 80)
        print(f"  {'排名':<4} {'板块':<12} {'5日涨幅':>8} {'资金分':>6} {'空间分':>6} {'总分':>6} {'RSI':>5} {'股票数':>5}")
        print("  " + "-" * 65)
        for i, (_, row) in enumerate(sec.head(15).iterrows()):
            print(f"  {i+1:<4} {row['industry']:<12} {row['sector_ret_5d']:>+7.2%} "
                  f"{row['capital_score']:>6.3f} {row['upside_score']:>6.3f} "
                  f"{row['total_score']:>6.3f} {row['avg_rsi_6']:>5.1f} "
                  f"{int(row['stock_count']):>5}")

        return sec

    def _select_hot_stocks(self, df, sector_scores, top_sectors=5, top_n_per=5):
        """在顶级板块中选热门股"""
        top_sec = sector_scores.head(top_sectors)
        hot_sectors = top_sec["industry"].tolist()

        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date].copy()

        # 个股评分
        latest["stock_capital"] = (
            latest["vol_ratio_5d"].fillna(1) * 0.25 +
            latest["net_capital_flow"].fillna(0) * 0.40 +
            latest["up_down_vol_ratio"].fillna(1).clip(0, 5) / 5 * 0.35
        )
        latest["stock_upside"] = (
            (1 - abs(latest["rsi_6"].fillna(50) - 52.5) / 50) * 0.40 +
            (-latest["gap_from_10d_high"].fillna(0) * 3).clip(0, 1) * 0.35 +
            np.where(latest["ret_5d"].fillna(0) > 0.10,
                     1 - (latest["ret_5d"] - 0.10) * 4, 1.0).clip(0, 1) * 0.25
        )
        latest["stock_score"] = latest["stock_capital"] * 0.55 + latest["stock_upside"] * 0.45

        # 硬过滤: RSI>85 或 5日涨幅>15% 排除 (已透支)
        latest = latest[
            (latest["rsi_6"].fillna(50) < 80) &
            (latest["ret_5d"].fillna(0) < 0.12)
        ]

        picks = []
        for industry in hot_sectors:
            sec_stocks = latest[latest["industry"] == industry].copy()
            if len(sec_stocks) == 0:
                continue
            sec_stocks = sec_stocks.nlargest(min(top_n_per, len(sec_stocks)), "stock_score")
            picks.append(sec_stocks)

        result = pd.concat(picks, ignore_index=True)
        result = result.sort_values(["industry", "stock_score"], ascending=[True, False])

        print(f"\n  [热门股筛选]  TOP {top_sectors} 板块, 每板块 {top_n_per} 只")
        print(f"  {'代码':<10} {'板块':<12} {'5日':>8} {'资金分':>6} {'空间分':>6} {'个股分':>6} {'RSI':>5} {'量比':>5}")
        print("  " + "-" * 65)
        for _, row in result.iterrows():
            print(f"  {row['code']:<10} {row['industry']:<12} {row.get('ret_5d',0):>+7.2%} "
                  f"{row['stock_capital']:>6.3f} {row['stock_upside']:>6.3f} "
                  f"{row['stock_score']:>6.3f} {row.get('rsi_6',0):>5.1f} "
                  f"{row.get('vol_ratio_5d',0):>5.2f}")

        self.stock_scores = result
        return result

    def save_results(self):
        if self.sector_scores is not None:
            self.sector_scores.to_csv(os.path.join(OUTPUT_DIR, "capital_flow_sectors.csv"),
                                       index=False, encoding="utf-8-sig")
        if self.stock_scores is not None:
            save_cols = ["code", "industry", "ret_5d", "stock_capital", "stock_upside",
                         "stock_score", "rsi_6", "vol_ratio_5d", "net_capital_flow"]
            avail = [c for c in save_cols if c in self.stock_scores.columns]
            self.stock_scores[avail].to_csv(os.path.join(OUTPUT_DIR, "capital_flow_picks.csv"),
                                             index=False, encoding="utf-8-sig")
        print(f"[CapitalFlow] 结果已保存")


def run_capital_flow_pipeline(fetcher):
    """资金认可度板块筛选 + 热门股 完整流程"""
    from models.hot_sector import CODE_TO_INDUSTRY

    print("\n" + "=" * 80)
    print("  资金认可度 + 上涨空间  板块筛选系统")
    print("=" * 80)

    codes = fetcher.get_stock_list()
    import datetime
    end_dt = datetime.date.today().strftime("%Y%m%d")
    start_dt = (datetime.date.today() - datetime.timedelta(days=180)).strftime("%Y%m%d")
    print(f"[Pipeline] 下载 {start_dt} ~ {end_dt} 行情...")
    daily = fetcher.get_daily_data(codes, start=start_dt, end=end_dt)

    selector = CapitalFlowSectorSelector(daily, CODE_TO_INDUSTRY)
    sector_scores = selector.compute()
    picks = selector._select_hot_stocks(selector.detail_df, sector_scores)
    selector.save_results()

    return selector, picks
