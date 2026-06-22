"""
回测引擎模块 - 月度调仓
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import BACKTEST_PARAMS, OUTPUT_DIR

class BacktestEngine:
    def __init__(self, predictions, daily_data, params=None):
        self.pred = predictions.copy()
        self.data = daily_data.copy()
        self.params = params or BACKTEST_PARAMS
        self.results = None
    
    def _get_monthly_dates(self, dates):
        """从日频日期中提取月末调仓日"""
        dates = pd.to_datetime(sorted(dates))
        df = pd.DataFrame({"date": dates})
        df["year_month"] = df["date"].dt.to_period("M")
        # 每月最后一个交易日
        monthly = df.groupby("year_month")["date"].max().tolist()
        return sorted(monthly)
    
    def run(self):
        print("[Backtest] 运行月度调仓回测...")
        top_n = self.params["top_n"]
        commission = self.params["commission"]
        slippage = self.params["slippage"]
        
        all_dates = sorted(self.pred["date"].unique())
        rebalance_dates = self._get_monthly_dates(all_dates)
        
        if len(rebalance_dates) < 2:
            print("  有效调仓日不足")
            return None
        
        print(f"  调仓日: {len(rebalance_dates)} 个月")
        
        portfolio_returns = []
        
        for i, date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]
            
            # 当日选股
            period = self.pred[self.pred["date"] == date].copy()
            if len(period) == 0:
                continue
            
            selected = period.nlargest(top_n, "score")
            selected_codes = selected["code"].tolist()
            
            # 计算持仓期收益
            holdings = self.data[
                (self.data["code"].isin(selected_codes)) &
                (self.data["date"] > date) &
                (self.data["date"] <= next_date)
            ].copy()
            
            if len(holdings) == 0:
                continue
            
            stock_rets = holdings.groupby("code").apply(
                lambda x: x["close"].iloc[-1] / x["close"].iloc[0] - 1
            )
            
            if len(stock_rets) > 0:
                period_ret = stock_rets.mean() - commission - slippage
                portfolio_returns.append({
                    "date": next_date,
                    "return": period_ret,
                    "n_stocks": len(stock_rets),
                })
        
        self.results = pd.DataFrame(portfolio_returns)
        if len(self.results) > 0:
            self.results["cum_return"] = (1 + self.results["return"]).cumprod() - 1
        
        print(f"  回测完成: {len(self.results)} 期")
        return self.results
    
    def compute_metrics(self):
        if self.results is None or len(self.results) == 0:
            return {}
        
        rets = self.results["return"].values
        n_periods = len(rets)
        n_years = max(n_periods / 12, 0.1)
        
        total_return = (1 + rets).prod() - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_vol = rets.std() * np.sqrt(12)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cum_series = self.results["cum_return"] + 1
        running_max = cum_series.cummax()
        drawdown = cum_series / running_max - 1
        max_dd = drawdown.min()
        win_rate = (rets > 0).mean()
        
        metrics = {
            "总收益率": f"{total_return:.2%}",
            "年化收益": f"{annual_return:.2%}",
            "年化波动": f"{annual_vol:.2%}",
            "夏普比率": f"{sharpe:.2f}",
            "最大回撤": f"{max_dd:.2%}",
            "胜率": f"{win_rate:.2%}",
            "回测期数": n_periods,
        }
        
        print("\n" + "=" * 50)
        print("  回测绩效")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        return metrics
    
    def plot(self, benchmark_df=None):
        if self.results is None or len(self.results) == 0:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        ax = axes[0]
        self.results["date"] = pd.to_datetime(self.results["date"])
        ax.plot(self.results["date"], self.results["cum_return"] + 1, 
                label="Strategy", linewidth=2, color="#1f77b4")
        
        if benchmark_df is not None and len(benchmark_df) > 0:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
            bench = benchmark_df.merge(
                self.results[["date"]], on="date", how="inner"
            )
            if len(bench) > 0:
                bench["cum"] = (1 + bench["close"].pct_change().fillna(0)).cumprod()
                ax.plot(bench["date"], bench["cum"], 
                        label="HS300", linewidth=1.5, color="#ff7f0e", alpha=0.7)
        
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Cumulative Return", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        
        ax2 = axes[1]
        self.results["month"] = pd.to_datetime(self.results["date"]).dt.to_period("M")
        monthly = self.results.groupby("month")["return"].sum()
        colors = ["#d62728" if r < 0 else "#2ca02c" for r in monthly]
        monthly.plot(kind="bar", ax=ax2, color=colors, width=0.8)
        ax2.set_title("Monthly Returns", fontsize=14, fontweight="bold")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "backtest_result.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Backtest] 图表已保存: {path}")
        return path
