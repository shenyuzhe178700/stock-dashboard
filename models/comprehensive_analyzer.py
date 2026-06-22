# -*- coding: utf-8 -*-
import os, sys, warnings
sys.path.insert(0, r"D:\quant_a_stock")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import OUTPUT_DIR

class ComprehensiveStockAnalyzer:

    def __init__(self, stock_code, sector="电力", lookback_years=2):
        self.code = stock_code
        self.sector = sector
        self.lookback = lookback_years
        self.df = None
        self.results = {}

    def fetch_all_data(self):
        from data.fetcher import DataFetcher, _fmt_date
        f = DataFetcher()
        plain = self.code.split(".")[-1] if "." in self.code else self.code
        bs_code = f"sh.{plain}" if plain.startswith(("6","5","9")) else f"sz.{plain}"

        end_dt = datetime.today().strftime("%Y%m%d")
        start_dt = (datetime.today() - timedelta(days=self.lookback*365)).strftime("%Y%m%d")

        # Stock data
        df = f._fetch_one_stock(bs_code, _fmt_date(start_dt), _fmt_date(end_dt))
        if df is None: raise ValueError(f"Cannot fetch {bs_code}")
        self.df = df.sort_values("date").reset_index(drop=True)

        # Market index
        mkt = f._fetch_one_stock("sh.000001", _fmt_date(start_dt), _fmt_date(end_dt))
        if mkt is not None:
            mkt["date"] = pd.to_datetime(mkt["date"])
            self.df["date"] = pd.to_datetime(self.df["date"])
            mkt = mkt.rename(columns={c: f"mkt_{c}" for c in mkt.columns if c != "date"})
            self.df = self.df.merge(mkt[["date","mkt_close","mkt_volume"]], on="date", how="left")

        # Sector index
        from data.sector_index import SECTOR_INDEX_MAP
        sec_code = SECTOR_INDEX_MAP.get(self.sector)
        if sec_code:
            sec = f._fetch_one_stock(sec_code, _fmt_date(start_dt), _fmt_date(end_dt))
            if sec is not None:
                sec["date"] = pd.to_datetime(sec["date"])
                sec = sec.rename(columns={c: f"sec_{c}" for c in sec.columns if c != "date"})
                self.df = self.df.merge(sec[["date","sec_close","sec_volume"]], on="date", how="left")

        print(f"Data: {len(self.df)} trading days")
        print(f"Period: {self.df['date'].min().date()} ~ {self.df['date'].max().date()}")

    def compute_all_metrics(self):
        df = self.df.copy()
        close = df["close"].values
        n = len(close)

        # ===== 1. Multi-Timeframe Returns =====
        for h in [1,3,5,10,20,60]:
            df[f"ret_{h}d"] = df["close"].pct_change(h)
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        # ===== 2. Volatility Regime =====
        for w in [5,10,20,60]:
            df[f"vol_{w}d"] = df["log_ret"].rolling(w).std() * np.sqrt(252)
        df["vol_regime"] = np.where(df["vol_20d"] > df["vol_60d"], "HIGH_VOL", "LOW_VOL")

        # ===== 3. Technical Indicators =====
        # Bollinger Bands (20,2)
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Keltner Channel (20, 2*ATR)
        df["atr_20"] = df["high"].rolling(20).max() - df["low"].rolling(20).min()
        df["kc_mid"] = df["bb_mid"]
        df["kc_upper"] = df["kc_mid"] + 2*df["atr_20"]/20
        df["kc_lower"] = df["kc_mid"] - 2*df["atr_20"]/20

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        df["rsi_14"] = 100 - 100/(1 + gain/loss.replace(0,1e-10))

        # ===== 4. Volume Analysis =====
        df["vol_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
        df["vol_ratio_20d"] = df["volume"] / df["volume"].rolling(20).mean()
        df["amount_ma20"] = df["amount"].rolling(20).mean()

        # OBV
        obv_dir = np.where(close[1:] > close[:-1], 1, np.where(close[1:] < close[:-1], -1, 0))
        obv = np.zeros(n)
        obv[1:] = np.cumsum(obv_dir * df["volume"].values[1:])
        df["obv"] = obv
        df["obv_ma20"] = df["obv"].rolling(20).mean()
        df["obv_divergence"] = df["obv"] / df["obv_ma20"] - 1

        # ===== 5. Money Flow =====
        typical = (df["high"] + df["low"] + df["close"]) / 3
        raw_mf = typical * df["volume"]
        pos_mf = np.where(typical > typical.shift(1), raw_mf, 0)
        neg_mf = np.where(typical < typical.shift(1), raw_mf, 0)
        mf_ratio_14 = pd.Series(pos_mf).rolling(14).sum() / pd.Series(neg_mf).rolling(14).sum().replace(0,1)
        df["mfi_14"] = 100 - 100/(1 + mf_ratio_14)

        # Close position in daily range
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        df["close_pos_ma5"] = df["close_position"].rolling(5).mean()

        # ===== 6. Sector Relative =====
        if "sec_close" in df.columns:
            df["sector_ret_20d"] = df["sec_close"].pct_change(20)
            df["relative_strength"] = df["ret_20d"] - df["sector_ret_20d"]
            df["sector_corr_20d"] = df["close"].pct_change().rolling(20).corr(df["sec_close"].pct_change())
        if "mkt_close" in df.columns:
            df["mkt_ret_20d"] = df["mkt_close"].pct_change(20)
            df["beta_60d"] = df["close"].pct_change().rolling(60).cov(df["mkt_close"].pct_change()) / df["mkt_close"].pct_change().rolling(60).var()

        # ===== 7. Key Levels =====
        recent = df.tail(60)
        self.results["resistance"] = recent["high"].max()
        self.results["support"] = recent["low"].min()
        self.results["current"] = close[-1]
        self.results["ma20"] = df["bb_mid"].iloc[-1]
        self.results["ma60"] = df["close"].rolling(60).mean().iloc[-1]

        # ===== 8. Drawdown Analysis =====
        cummax = np.maximum.accumulate(close[-120:])
        dd = (close[-120:] / cummax - 1) * 100
        self.results["max_dd_120d"] = dd.min()
        self.results["current_dd"] = dd[-1]

        self.df = df

    def monte_carlo_simulation(self, n_sim=5000, horizon=20):
        close = self.df["close"].dropna().values
        log_rets = np.log(close[1:] / close[:-1])

        # Estimate parameters from recent 60 days
        recent_rets = log_rets[-120:]
        mu = recent_rets.mean() * 252
        sigma = recent_rets.std() * np.sqrt(252)

        # GARCH-like: separate vol regimes
        high_vol_mask = self.df["vol_regime"].iloc[-120:].values == "HIGH_VOL"
        if high_vol_mask.sum() > 5:
            rets_120 = recent_rets
            sigma_high = rets_120[high_vol_mask].std() * np.sqrt(252)
            sigma_low = rets_120[~high_vol_mask].std() * np.sqrt(252)
            current_is_high = self.df["vol_regime"].iloc[-1] == "HIGH_VOL"
            sigma = sigma_high if current_is_high else sigma_low

        S0 = close[-1]
        dt = 1/252
        np.random.seed(42)
        paths = np.zeros((n_sim, horizon))
        paths[:, 0] = S0

        for t in range(1, horizon):
            Z = np.random.normal(0, 1, n_sim)
            paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)

        final_prices = paths[:, -1]
        returns = (final_prices / S0 - 1) * 100

        self.results["mc_mean_return"] = returns.mean()
        self.results["mc_median_return"] = np.median(returns)
        self.results["mc_prob_up"] = (returns > 0).mean() * 100
        self.results["mc_prob_5pct"] = (returns > 5).mean() * 100
        self.results["mc_prob_10pct"] = (returns > 10).mean() * 100
        self.results["mc_5pct_var"] = np.percentile(returns, 5)
        self.results["mc_95pct_var"] = np.percentile(returns, 95)
        self.results["mc_sigma"] = sigma
        self.results["mc_mu"] = mu
        self.results["mc_paths"] = paths

    def ml_prediction(self, horizon=20):
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb

        df = self.df.dropna().copy()
        df["label"] = df["close"].shift(-horizon) / df["close"] - 1

        features = [c for c in df.columns if c not in
            ["date","code","open","high","low","close","volume","amount","turnover_rate",
             "label","mkt_close","mkt_volume","sec_close","sec_volume",
             "vol_regime","log_ret"] and not c.startswith("us_")]
        features = [f for f in features if df[f].notna().sum() > 100 and df[f].std() > 0]

        df_valid = df.dropna(subset=["label"] + features)
        if len(df_valid) < 60:
            print(f"ML: insufficient data ({len(df_valid)}), skipping")
            return

        split = int(len(df_valid) * 0.8)
        train, test = df_valid.iloc[:split], df_valid.iloc[split:]

        X_tr = StandardScaler().fit_transform(train[features].values)
        X_te = StandardScaler().fit_transform(test[features].values)

        model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.03,
            num_leaves=15, random_state=42, verbosity=-1)
        model.fit(X_tr, train["label"].values)

        pred = model.predict(X_te)
        from scipy.stats import spearmanr
        ic = spearmanr(test["label"].values, pred)[0]

        # Latest prediction
        latest_X = StandardScaler().fit_transform(df_valid[features].iloc[-1:].values)
        latest_pred = model.predict(latest_X)[0]

        self.results["ml_pred_20d"] = latest_pred
        self.results["ml_ic"] = ic
        self.results["ml_features"] = len(features)

        # Feature importance
        imp = pd.DataFrame({"f": features, "imp": model.feature_importances_}).nlargest(10, "imp")
        self.results["ml_top_features"] = imp

    def generate_report(self):
        r = self.results
        df = self.df
        latest = df.iloc[-1]
        prev_m = df.iloc[-20] if len(df) >= 21 else df.iloc[0]

        print("\n" + "=" * 75)
        print(f"  {self.code} 月度上涨空间量化分析报告")
        print(f"  板块: {self.sector}  |  分析日期: {datetime.today().strftime('%Y-%m-%d')}")
        print("=" * 75)

        # Basic
        print(f"\n{'='*40}")
        print("  一、基础数据")
        print(f"{'='*40}")
        print(f"  最新收盘: {r['current']:.2f}")
        print(f"  20日涨跌: {latest.get('ret_20d',0):+.2%}")
        vol_5 = latest.get("ret_5d",0)
        print(f"  5日涨跌:  {vol_5:+.2%}")
        ma20, ma60 = latest.get("bb_mid",r["current"]), r["ma60"]
        print(f"  vs MA20:  {latest.get('bias_ma20',r['current']/ma20-1):+.2%}" if "bias_ma20" in df.columns else f"  vs MA20:  {r['current']/ma20-1:+.2%}")
        print(f"  vs MA60:  {r['current']/ma60-1:+.2%}")
        print(f"  支撑/阻力: {r['support']:.2f} / {r['resistance']:.2f}")

        # Technical
        print(f"\n{'='*40}")
        print("  二、技术面分析")
        print(f"{'='*40}")
        bb = latest.get("bb_position", 0.5)
        bb_status = "接近上轨(压力)" if bb > 0.8 else ("接近下轨(支撑)" if bb < 0.2 else "中轨附近")
        print(f"  Bollinger位置: {bb:.2f} ({bb_status})")
        print(f"  RSI(14): {latest.get('rsi_14',50):.1f} " +
              ("(超买)" if latest.get("rsi_14",50)>70 else "(超卖)" if latest.get("rsi_14",50)<30 else "(中性)"))
        macd_h = latest.get("macd_hist", 0)
        print(f"  MACD柱: {macd_h:.3f} " + ("(多头)" if macd_h>0 else "(空头)"))
        print(f"  波动率(年化): {latest.get('vol_20d',0)*100:.1f}%")
        print(f"  波动率状态: {latest.get('vol_regime','N/A')}")

        # Volume
        print(f"\n{'='*40}")
        print("  三、资金面分析")
        print(f"{'='*40}")
        print(f"  量比(5日): {latest.get('vol_ratio_5d',1):.2f}")
        print(f"  量比(20日): {latest.get('vol_ratio_20d',1):.2f}")
        mfi = latest.get("mfi_14", 50)
        mfi_s = "资金流入" if mfi>60 else ("资金流出" if mfi<40 else "中性")
        print(f"  MFI(14): {mfi:.1f} ({mfi_s})")
        obv_div = latest.get("obv_divergence", 0)
        obv_s = "OBV强于均线(资金积累)" if obv_div > 0.05 else ("OBV弱于均线(资金流出)" if obv_div < -0.05 else "OBV中性")
        print(f"  OBV偏离: {obv_div:+.2%} ({obv_s})")
        cp = latest.get("close_pos_ma5", 0.5)
        print(f"  收盘位置(5日均): {cp:.2f} " + ("(强势收盘)" if cp>0.6 else "(弱势收盘)" if cp<0.4 else "(中性)"))

        # Sector
        print(f"\n{'='*40}")
        print("  四、板块与市场")
        print(f"{'='*40}")
        if "sector_ret_20d" in df.columns:
            sr = latest.get("sector_ret_20d", 0)
            rs = latest.get("relative_strength", 0)
            corr = latest.get("sector_corr_20d", 0)
            print(f"  板块({self.sector})20日: {sr:+.2%}")
            print(f"  个股vs板块超额: {rs:+.2%} " + ("(领涨)" if rs>0.03 else "(跑输)" if rs<-0.03 else "(同步)"))
            print(f"  个股-板块相关: {corr:.2f}")
        if "mkt_ret_20d" in df.columns:
            mr = latest.get("mkt_ret_20d", 0)
            print(f"  上证指数20日: {mr:+.2%}")
            beta = latest.get("beta_60d", 1)
            print(f"  Beta(60日): {beta:.2f} " + ("(高弹性)" if beta>1.2 else "(防御型)" if beta<0.8 else "(中性)"))

        # Current DD
        print(f"\n{'='*40}")
        print("  五、风险评估")
        print(f"{'='*40}")
        print(f"  60日最大回撤: {r.get('max_dd_120d',0):.1f}%")
        print(f"  当前距高点: {r.get('current_dd',0):.1f}%")

        # ML
        print(f"\n{'='*40}")
        print("  六、机器学习预测")
        print(f"{'='*40}")
        if "ml_pred_20d" in r:
            ml_p = r["ml_pred_20d"] * 100
            print(f"  20日预测: {ml_p:+.2f}%")
            print(f"  Rank IC: {r.get('ml_ic',0):.4f}")
            print(f"  使用特征: {r.get('ml_features',0)} 个")
            if "ml_top_features" in r:
                print(f"  Top 5 特征: {', '.join(r['ml_top_features']['f'].head(5).tolist())}")
        else:
            print("  数据不足, 跳过ML预测")

        # Monte Carlo
        print(f"\n{'='*40}")
        print("  七、蒙特卡洛模拟 (5000次)")
        print(f"{'='*40}")
        if "mc_mean_return" in r:
            print(f"  预期20日收益率: {r['mc_mean_return']:+.2f}%")
            print(f"  中位数收益率:   {r['mc_median_return']:+.2f}%")
            print(f"  上涨概率:       {r['mc_prob_up']:.0f}%")
            print(f"  >+5%概率:       {r['mc_prob_5pct']:.0f}%")
            print(f"  >+10%概率:      {r['mc_prob_10pct']:.0f}%")
            print(f"  95% VaR:        {r['mc_5pct_var']:+.2f}%")
            print(f"  95% 上涨空间:   {r['mc_95pct_var']:+.2f}%")
            print(f"  年化波动率:     {r['mc_sigma']*100:.1f}%")
            print(f"  年化预期收益:   {r['mc_mu']*100:.1f}%")

        # Ensemble Conclusion
        print(f"\n{'='*75}")
        print("  综合结论")
        print(f"{'='*75}")

        signals = []
        # ML signal
        ml_s = r.get("ml_pred_20d", 0)
        signals.append(("ML模型", ml_s*100, 5 if ml_s>0.03 else (3 if ml_s>0 else 1)))

        # MC signal
        mc_s = r.get("mc_prob_up", 50)
        signals.append(("蒙特卡洛", mc_s-50, 5 if mc_s>70 else (3 if mc_s>50 else 1)))

        # Technical signal
        rsi_v = latest.get("rsi_14", 50)
        bb_v = latest.get("bb_position", 0.5)
        tech_score = 3
        if 40 < rsi_v < 65 and 0.3 < bb_v < 0.7:
            tech_score = 4
        elif rsi_v < 30 or bb_v < 0.1:
            tech_score = 4  # oversold bounce
        elif rsi_v > 75:
            tech_score = 2
        signals.append(("技术面", 0, tech_score))

        # Fund flow
        mfi_v = latest.get("mfi_14", 50)
        fund_score = 3
        if mfi_v > 55:
            fund_score = 4
        elif mfi_v < 35:
            fund_score = 2
        signals.append(("资金面", 0, fund_score))

        total_score = sum(s[2] for s in signals)
        max_score = len(signals) * 5
        stars = int(total_score / max_score * 5 + 0.5)
        stars = max(1, min(5, stars))
        star_str = "[" + "*" * stars + "o" * (5 - stars) + "]"

        verdict = ""
        if total_score >= 15:
            verdict = "看多 - 多维度信号一致向好, 一个月内上涨空间较大"
        elif total_score >= 11:
            verdict = "偏多 - 多数信号积极, 存在结构性上涨机会"
        elif total_score >= 7:
            verdict = "中性 - 信号分歧, 建议观望或轻仓"
        else:
            verdict = "谨慎 - 多维度信号偏弱, 短期上涨空间有限"

        print(f"\n  {'维度':<12} {'评分':>8}")
        print(f"  {'-'*22}")
        for name, val, score in signals:
            bar = "[" + "#" * score + "-" * (5-score) + "]"
            extra = f" ({val:+.1f}%)" if val != 0 else ""
            print(f"  {name:<12} {bar} {score}/5{extra}")
        print(f"\n  综合评分: {total_score}/{max_score}  评级: {star_str}")
        print(f"  结论: {verdict}")

        # Price targets
        if "mc_95pct_var" in r:
            tgt_up = r["current"] * (1 + r["mc_95pct_var"]/100)
            tgt_median = r["current"] * (1 + r["mc_median_return"]/100)
            tgt_down = r["current"] * (1 + r["mc_5pct_var"]/100)
            print(f"\n  价格预测区间 (20交易日):")
            print(f"    乐观(95%): {tgt_up:.2f}")
            print(f"    中位:      {tgt_median:.2f}")
            print(f"    保守(5%):  {tgt_down:.2f}")

        print(f"\n{'='*75}")
        print("  风险提示: 量化模型基于历史统计, 不构成投资建议。")
        print("  实际走势受政策、资金、情绪等多因素影响, 请综合判断。")
        print(f"{'='*75}\n")
