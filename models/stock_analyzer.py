# -*- coding: utf-8 -*-
import os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, DATA_DIR
from data.sector_index import SECTOR_INDEX_MAP, FUTURES_PROXY, STOCK_FUTURES_MAP
from data.us_reference import US_TECH_MAP, US_SECTOR_ETF
from data.fetcher import _fmt_date


class SingleStockAnalyzer:
    """个股短线走势量化分析模型
    5维因子: 大盘 / 资金成分 / 主力资金 / 板块 / 期货联动
    """

    def __init__(self, fetcher, stock_code, sector_name=None, futures_symbol=None):
        self.fetcher = fetcher
        self.stock_code = stock_code
        self.sector_name = sector_name
        self.futures_symbol = futures_symbol or STOCK_FUTURES_MAP.get(stock_code.split(".")[-1])

        # 自动推断板块
        if not self.sector_name:
            from models.hot_sector import CODE_TO_INDUSTRY
            plain = stock_code.split(".")[-1] if "." in stock_code else stock_code
            self.sector_name = CODE_TO_INDUSTRY.get(plain, None)

        self.sector_index_code = SECTOR_INDEX_MAP.get(self.sector_name) if self.sector_name else None
        self.market_index_code = "sh.000001"  # 上证指数

        self.stock_data = None
        self.market_data = None
        self.sector_data = None
        self.factor_df = None
        self.model = None
        self.scaler = StandardScaler()

    def _bs_code(self, plain_code):
        if "." in plain_code:
            return plain_code
        pre = "sh" if plain_code.startswith(("6", "5", "9")) else "sz"
        return f"{pre}.{plain_code}"

    def fetch_data(self, lookback_days=120):
        """获取个股、大盘、板块数据"""
        import datetime
        end_dt = datetime.date.today().strftime("%Y%m%d")
        start_dt = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y%m%d")

        bs_code = self._bs_code(self.stock_code)
        print(f"[Analyzer] 获取 {bs_code} 数据 ({start_dt}~{end_dt})...")

        # 个股
        self.stock_data = self.fetcher._fetch_one_stock(
            bs_code, _fmt_date(start_dt), _fmt_date(end_dt))
        if self.stock_data is None or len(self.stock_data) < 20:
            raise ValueError(f"股票 {bs_code} 数据不足")

        # 大盘
        print(f"[Analyzer] 获取大盘指数 {self.market_index_code}...")
        rs = self.fetcher._fetch_one_stock(
            self.market_index_code, _fmt_date(start_dt), _fmt_date(end_dt))
        if rs is not None:
            self.market_data = rs.rename(columns={
                "open": "m_open", "high": "m_high", "low": "m_low",
                "close": "m_close", "volume": "m_volume", "amount": "m_amount"
            })

        # 板块
        if self.sector_index_code:
            print(f"[Analyzer] 获取板块指数 {self.sector_name}({self.sector_index_code})...")
            sr = self.fetcher._fetch_one_stock(
                self.sector_index_code, _fmt_date(start_dt), _fmt_date(end_dt))
            if sr is not None:
                self.sector_data = sr.rename(columns={
                    "open": "s_open", "high": "s_high", "low": "s_low",
                    "close": "s_close", "volume": "s_volume", "amount": "s_amount"
                })

                # --- 美股参考数据 ---
        self.us_data = {}
        plain_code = self.stock_code.split(".")[-1] if "." in self.stock_code else self.stock_code
        us_refs = US_TECH_MAP.get(plain_code, [])
        if us_refs:
            print(f"[Analyzer] 获取美股参考数据...")
            try:
                import yfinance as yf
                for ticker, name, note in us_refs[:2]:
                    try:
                        import time as _time
                        _time.sleep(1)  # 避免限流
                        us_df = yf.download(ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=True, timeout=10)
                        if len(us_df) > 5:
                            us_df = us_df.reset_index()
                            us_df["date"] = pd.to_datetime(us_df["Date"]).dt.tz_localize(None)
                            us_df = us_df.rename(columns={
                                "Open": f"us_{ticker}_open", "High": f"us_{ticker}_high",
                                "Low": f"us_{ticker}_low", "Close": f"us_{ticker}_close",
                                "Volume": f"us_{ticker}_volume"
                            })
                            self.us_data[ticker] = us_df[[c for c in us_df.columns if c.startswith("us_") or c == "date"]]
                            print(f"    {ticker}({name}): {len(us_df)}天")
                    except Exception as e:
                        print(f"    {ticker}: 获取失败 ({e})")
            except ImportError:
                print("    yfinance 未安装, 跳过美股数据")

        # 美股ETF参考
        self.us_etf_data = None
        etf_ticker = US_SECTOR_ETF.get(self.sector_name) if self.sector_name else None
        if etf_ticker:
            try:
                import yfinance as yf
                etf_df = yf.download(etf_ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=True, timeout=10)
                if len(etf_df) > 5:
                    etf_df = etf_df.reset_index()
                    etf_df["date"] = pd.to_datetime(etf_df["Date"]).dt.tz_localize(None)
                    self.us_etf_data = etf_df.rename(columns={"Close": "us_etf_close", "Volume": "us_etf_volume"})
                    print(f"    ETF {etf_ticker}: {len(etf_df)}天")
            except:
                pass

        print(f"  个股: {len(self.stock_data)}天, 大盘: {len(self.market_data) if self.market_data is not None else 0}天, "
              f"板块: {len(self.sector_data) if self.sector_data is not None else 0}天, "
              f"美股: {len(self.us_data)}只")
        return True

    def _merge_data(self):
        """合并个股、大盘、板块数据"""
        df = self.stock_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if self.market_data is not None:
            self.market_data["date"] = pd.to_datetime(self.market_data["date"])
            df = df.merge(self.market_data, on="date", how="left")

        if self.sector_data is not None:
            self.sector_data["date"] = pd.to_datetime(self.sector_data["date"])
            df = df.merge(self.sector_data, on="date", how="left")

        # 美股参考数据
        for ticker, us_df in self.us_data.items():
            us_df["date"] = pd.to_datetime(us_df["date"])
            df = df.merge(us_df, on="date", how="left")

        # 美股ETF
        if self.us_etf_data is not None:
            self.us_etf_data["date"] = pd.to_datetime(self.us_etf_data["date"])
            df = df.merge(self.us_etf_data, on="date", how="left")

        return df

    def compute_factors(self):
        """计算5维因子体系"""
        print("[Analyzer] 计算5维因子...")
        df = self._merge_data()

        # === 1. 大盘因子 (4个) ===
        if "m_close" in df.columns:
            df["market_ret_1d"] = df["m_close"].pct_change(1)
            df["market_ret_3d"] = df["m_close"].pct_change(3)
            df["market_ret_5d"] = df["m_close"].pct_change(5)
            df["market_vol_5d"] = df["m_close"].pct_change().rolling(5).std()
            df["market_vol_ratio"] = df["m_volume"] / df["m_volume"].rolling(5).mean()
            df["market_amp"] = (df["m_high"] - df["m_low"]) / df["m_close"].shift(1)
        else:
            for c in ["market_ret_1d","market_ret_3d","market_ret_5d",
                       "market_vol_5d","market_vol_ratio","market_amp"]:
                df[c] = np.nan

        # === 2. 资金成分因子 (5个) ===
        # OBV (On-Balance Volume)
        close_diff = df["close"].diff()
        direction = np.where(close_diff > 0, 1, np.where(close_diff < 0, -1, 0))
        df["obv"] = (direction * df["volume"]).cumsum()
        df["obv_ratio_5d"] = df["obv"].pct_change(5)

        # MFI (Money Flow Index) 14日
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]
        pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
        pos_sum = pd.Series(pos_flow).rolling(14).sum()
        neg_sum = pd.Series(neg_flow).rolling(14).sum()
        money_ratio = pos_sum / neg_sum.replace(0, 1e-10)
        df["mfi_14"] = 100 - (100 / (1 + money_ratio))

        # VWAP 偏离 (当日收盘相对成交量加权均价)
        df["vwap"] = (df["amount"] / df["volume"]).replace([np.inf, -np.inf], np.nan)
        df["vwap_deviation"] = df["close"] / df["vwap"] - 1

        # 大单成交占比代理: 每笔成交额 = amount/volume, 变化率表示成交结构变化
        df["avg_trade_size"] = df["amount"] / df["volume"].replace(0, np.nan)
        df["amount_concentration"] = df["avg_trade_size"].pct_change(5)

        # 净流入代理: 上涨日量-下跌日量 的滚动累计
        up_vol = np.where(df["close"] > df["close"].shift(1), df["volume"], 0)
        down_vol = np.where(df["close"] < df["close"].shift(1), df["volume"], 0)
        df["net_flow_proxy"] = pd.Series(up_vol - down_vol).rolling(10).sum() / df["volume"].rolling(10).mean().replace(0, 1)

        # === 3. 主力资金因子 (4个) ===
        # 尾盘放量 (最后30分钟占全天比代理: 用振幅和收盘位置推断)
        # 收盘在日内高位+放量 = 主力吸筹信号
        df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
        df["big_order_proxy"] = df["close_position"] * df["volume"] / df["volume"].rolling(5).mean()

        # 异常放量: 当日量 / 20日均量
        df["block_volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        # 上涨日 vs 下跌日 量能对比
        df["up_vol_avg"] = pd.Series(up_vol).rolling(20).mean()
        df["down_vol_avg"] = pd.Series(down_vol).rolling(20).mean()
        df["up_down_vol_ratio"] = df["up_vol_avg"] / df["down_vol_avg"].replace(0, 1)

        # 主力流向: 位置*量*方向
        df["institutional_flow"] = df["close_position"] * df["volume"] * np.sign(df["close"].diff().fillna(0))

        # === 4. 板块因子 (4个) ===
        if "s_close" in df.columns:
            df["sector_ret_1d"] = df["s_close"].pct_change(1)
            df["sector_ret_5d"] = df["s_close"].pct_change(5)
            df["sector_vol_ratio"] = df["s_volume"] / df["s_volume"].rolling(5).mean()

            # 个股相对板块超额
            df["stock_vs_sector_5d"] = df["close"].pct_change(5) - df["sector_ret_5d"]
            df["stock_vs_sector_1d"] = df["close"].pct_change(1) - df["sector_ret_1d"]

            # 个股与板块相关性
            df["stock_sector_corr"] = df["close"].pct_change().rolling(20).corr(
                df["s_close"].pct_change())
        else:
            for c in ["sector_ret_1d","sector_ret_5d","sector_vol_ratio",
                       "stock_vs_sector_5d","stock_vs_sector_1d","stock_sector_corr"]:
                df[c] = np.nan

        # === 5. 期货联动因子 (3个) ===
        # 使用板块指数作为期货代理
        if self.futures_symbol and "s_close" in df.columns:
            df["futures_ret_1d"] = df["s_close"].pct_change(1)
            df["futures_ret_5d"] = df["s_close"].pct_change(5)

            # 期货/现货背离: 股票涨幅 - 板块涨幅
            df["stock_futures_divergence"] = (
                df["close"].pct_change(5) - df["s_close"].pct_change(5))

            # 背离度(标准化): 过去5日个股与板块走势差异的极值
            df["divergence_extreme"] = df["stock_futures_divergence"].rolling(20).apply(
                lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10) if len(x) > 5 else 0
            )
        else:
            df["futures_ret_1d"] = 0
            df["futures_ret_5d"] = 0
            df["stock_futures_divergence"] = 0
            df["divergence_extreme"] = 0

        # === 6. 美股参考因子 ===
        us_close_cols = [c for c in df.columns if c.startswith("us_") and c.endswith("_close") and "etf" not in c]
        for i, col in enumerate(us_close_cols):
            ticker = col.replace("us_", "").replace("_close", "")
            df[f"us_{ticker}_ret_1d"] = df[col].pct_change(1)
            df[f"us_{ticker}_ret_5d"] = df[col].pct_change(5)
            # 中美联动: A股 vs 美股 5日相关性
            df[f"us_{ticker}_corr"] = df["close"].pct_change().rolling(20).corr(df[col].pct_change())

        # 美股ETF
        if "us_etf_close" in df.columns:
            df["us_etf_ret_1d"] = df["us_etf_close"].pct_change(1)
            df["us_etf_ret_5d"] = df["us_etf_close"].pct_change(5)
            df["us_etf_corr"] = df["close"].pct_change().rolling(20).corr(
                df["us_etf_close"].pct_change())

        # 美股领先指标: 美股前一日涨幅 (时差优势)
        for i, col in enumerate(us_close_cols):
            ticker = col.replace("us_", "").replace("_close", "")
            ret_col = f"us_{ticker}_ret_1d"
            if ret_col in df.columns:
                df[f"us_{ticker}_lead"] = df[ret_col].shift(1)  # T-1日美股 -> T日A股
        if "us_etf_ret_1d" in df.columns:
            df["us_etf_lead"] = df["us_etf_ret_1d"].shift(1)

        # === 6. 个股自身技术因子 (补充) ===
        df["ret_1d"] = df["close"].pct_change(1)
        df["ret_5d"] = df["close"].pct_change(5)
        df["vol_ratio_5d"] = df["volume"] / df["volume"].rolling(5).mean()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["bias_ma5"] = df["close"] / df["ma_5"] - 1
        df["bias_ma20"] = df["close"] / df["ma_20"] - 1
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)

        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(6).mean()
        loss = (-delta).clip(lower=0).rolling(6).mean()
        rs = gain / loss.replace(0, 1e-10)
        df["rsi_6"] = 100 - (100 / (1 + rs))

        # 连涨天数
        up = (df["close"].diff() > 0).astype(int)
        cnt = 0
        consec = []
        for v in up.values:
            cnt = cnt + 1 if v else 0
            consec.append(cnt)
        df["consecutive_up"] = consec

        # 标签: 5日 forward return
        df["label_5d"] = df["close"].shift(-5) / df["close"] - 1

        df = df.replace([np.inf, -np.inf], np.nan)
        self.factor_df = df
        print(f"  因子计算完成: {len(df)} 行, {len(df.columns)} 列")
        return df

    def get_factor_list(self):
        """获取因子列表"""
        factors = [
            # 大盘
            "market_ret_1d", "market_ret_3d", "market_ret_5d",
            "market_vol_5d", "market_vol_ratio", "market_amp",
            # 资金
            "obv_ratio_5d", "mfi_14", "vwap_deviation",
            "amount_concentration", "net_flow_proxy",
            # 主力
            "big_order_proxy", "block_volume_ratio",
            "up_down_vol_ratio", "institutional_flow",
            # 板块
            "sector_ret_1d", "sector_ret_5d", "sector_vol_ratio",
            "stock_vs_sector_5d", "stock_vs_sector_1d", "stock_sector_corr",
            # 期货
            "futures_ret_1d", "futures_ret_5d",
            "stock_futures_divergence", "divergence_extreme",
            # 技术
            "ret_1d", "ret_5d", "vol_ratio_5d",
            "bias_ma5", "bias_ma20", "amplitude", "rsi_6", "consecutive_up",
        ]
        return [f for f in factors if f in self.factor_df.columns]

    def train_model(self, lookback_days=120):
        """训练个股专属预测模型"""
        if self.factor_df is None:
            self.compute_factors()

        df = self.factor_df.copy()
        factor_list = self.get_factor_list()
        # Remove factors that are all NaN (missing data dimensions)
        valid_factors = []
        for f in factor_list:
            if df[f].notna().sum() > 10:
                valid_factors.append(f)
        skipped = [f for f in factor_list if f not in valid_factors]
        if skipped:
            print(f"[Analyzer] 跳过缺失因子({len(skipped)}个): {skipped[:5]}...")
        factor_list = valid_factors
        df = self.factor_df.dropna(subset=["label_5d"] + factor_list).copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=factor_list)

        if len(df) < 40:
            print(f"[Analyzer] 数据不足({len(df)}行), 跳过训练")
            return None

        # 时间序列切分
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        X_train = train[factor_list].values
        y_train = train["label_5d"].values
        X_test = test[factor_list].values
        y_test = test["label_5d"].values

        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        print(f"[Analyzer] 训练个股模型: 训练{len(X_train)}, 测试{len(X_test)}, 因子{len(factor_list)}")

        self.model = lgb.LGBMRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.03,
            num_leaves=15, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, n_jobs=-1, verbosity=-1
        )
        self.model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
        )

        y_pred = self.model.predict(X_test_s)
        ic = spearmanr(y_test, y_pred)[0]
        print(f"  个股 Rank IC: {ic:.4f}")

        # 因子重要度
        importance = pd.DataFrame({
            "factor": factor_list,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        print("\n  Top 10 因子:")
        for _, row in importance.head(10).iterrows():
            print(f"    {row['factor']:<30} {row['importance']:.4f}")

        self.feature_names = factor_list
        self.importance = importance
        return importance

    def predict_latest(self):
        """最新交易日预测"""
        if self.model is None:
            raise ValueError("请先训练模型")

        factor_list = self.feature_names
        df = self.factor_df.dropna(subset=factor_list).copy()
        if len(df) == 0:
            raise ValueError("无有效数据")

        latest = df.iloc[-1:]
        X = self.scaler.transform(latest[factor_list].values)
        score = self.model.predict(X)[0]

        return score, latest

    def generate_report(self):
        """生成完整分析报告"""
        print("\n" + "=" * 70)
        stock_name = self.stock_code
        if "." in stock_name:
            stock_name = stock_name.split(".")[-1]
        print(f"  {stock_name} 短线走势量化分析报告")
        print("=" * 70)

        df = self.factor_df
        if df is None or len(df) < 5:
            print("  数据不足, 无法生成报告")
            return

        latest = df.iloc[-1]
        prev = df.iloc[-6] if len(df) >= 6 else df.iloc[0]

        # 基础信息
        print(f"\n  [基本信息]")
        print(f"  最新日期: {latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else latest['date']}")
        print(f"  收盘价: {latest['close']:.2f}")
        print(f"  5日涨跌: {latest.get('ret_5d', 0):+.2%}")

        # 大盘环境
        print(f"\n  [大盘环境]")
        mr5 = latest.get("market_ret_5d", 0)
        if pd.notna(mr5):
            env = "强势" if mr5 > 0.02 else ("弱势" if mr5 < -0.02 else "震荡")
            print(f"  上证指数5日: {mr5:+.2%}  ({env})")
        mr1 = latest.get("market_ret_1d", 0)
        mvol = latest.get("market_vol_ratio", 1)
        if pd.notna(mr1):
            print(f"  大盘1日: {mr1:+.2%}  量比: {mvol:.2f}")

        # 资金面
        print(f"\n  [资金成分]")
        obv = latest.get("obv_ratio_5d", 0)
        mfi = latest.get("mfi_14", 50)
        if pd.notna(obv):
            obv_signal = "资金流入" if obv > 0 else "资金流出"
            print(f"  OBV 5日: {obv:+.2%} ({obv_signal})")
            print(f"  MFI(14): {mfi:.1f} " +
                  ("(超买)" if mfi > 80 else "(超卖)" if mfi < 20 else "(正常)"))
        vwap = latest.get("vwap_deviation", 0)
        if pd.notna(vwap):
            print(f"  VWAP偏离: {vwap:+.2%} " + ("(收盘偏强)" if vwap > 0 else "(收盘偏弱)"))
            print(f"  成交浓度: {latest.get('amount_concentration', 0):+.2%}")

        # 主力资金
        print(f"\n  [主力资金]")
        big = latest.get("big_order_proxy", 0)
        block = latest.get("block_volume_ratio", 1)
        if pd.notna(big):
            print(f"  主力代理信号: {big:.2f} " + ("(主力活跃)" if big > 1.5 else "(正常)" if big > 0.5 else "(主力休眠)"))
            print(f"  异常放量比: {block:.2f} " + ("(异常放量!)" if block > 2 else "(放量)" if block > 1.3 else ""))
        ud_ratio = latest.get("up_down_vol_ratio", 1)
        if pd.notna(ud_ratio):
            print(f"  涨跌量比: {ud_ratio:.2f} " + ("(上涨放量>下跌)" if ud_ratio > 1 else "(下跌放量>上涨)"))

        # 板块
        print(f"\n  [板块: {self.sector_name or '未知'}]")
        sr5 = latest.get("sector_ret_5d", 0)
        if self.sector_name is None or self.sector_name == "未知":
            print(f"\n  [板块: 暂无分类]")
            print(f"  请在 data/industry.py 中添加该股票的行业分类")
        sr5 = latest.get("sector_ret_5d", 0)
        if pd.notna(sr5) and self.sector_name is not None:
            print(f"  板块5日: {sr5:+.2%}")
            vs_sec = latest.get("stock_vs_sector_5d", 0)
            status = "领涨板块" if vs_sec > 0.02 else ("跑输板块" if vs_sec < -0.02 else "同步板块")
            print(f"  个股vs板块: {vs_sec:+.2%}  ({status})")
        corr = latest.get("stock_sector_corr", 0)
        if pd.notna(corr):
            print(f"  个股-板块相关性: {corr:.2f}")

        # 期货联动
        if self.futures_symbol:
            print(f"\n  [期货联动: {self.futures_symbol.upper()}]")
            print(f"  代理板块5日: {latest.get('futures_ret_5d', 0):+.2%}")
            div = latest.get("stock_futures_divergence", 0)
            if pd.notna(div):
                div_signal = "个股强于期货(补涨动力)" if div > 0.03 else ("个股弱于期货(补跌风险)" if div < -0.03 else "联动正常")
                print(f"  个股-期货背离: {div:+.2%} ({div_signal})")
            print(f"  背离度: {latest.get('divergence_extreme', 0):.2f} " +
                  ("(极端背离!)" if abs(latest.get('divergence_extreme', 0)) > 2 else ""))

        # 技术面
        print(f"\n  [技术面]")
        print(f"  RSI(6): {latest.get('rsi_6', 50):.1f} " +
              ("(超买)" if latest.get('rsi_6', 50) > 80 else "(超卖)" if latest.get('rsi_6', 50) < 20 else ""))
        print(f"  连涨: {int(latest.get('consecutive_up', 0))}天")
        print(f"  MA5偏离: {latest.get('bias_ma5', 0):+.2%}")
        print(f"  MA20偏离: {latest.get('bias_ma20', 0):+.2%}")
        print(f"  振幅: {latest.get('amplitude', 0):.2%}")

        # 综合评分
        print(f"\n  [综合评分]")
        try:
            score, _ = self.predict_latest()
            score_pct = score * 100
        except (ValueError, AttributeError):
            score = 0
            score_pct = 0
        stars = min(5, max(1, int((score_pct + 0.5) / 1.0 + 3)))
        star_str = "\u2605" * stars + "\u2606" * (5 - stars)
        signal = "看多" if score > 0.005 else ("看空" if score < -0.005 else "中性")
        print(f"  预测5日收益: {score:+.2%}")
        print(f"  评级: {star_str} ({signal})")

        # 关键信号
        print(f"\n  [关键信号]")
        signals = self._generate_signals(latest)
        for s in signals:
            print(f"  {s}")

        print("\n" + "=" * 70)
        print("  风险提示: 模型基于历史数据, 仅供参考, 不构成投资建议")
        print("=" * 70)

        return True

    def _generate_signals(self, row):
        """生成交易信号列表"""
        signals = []

        # 大盘环境
        mr5 = row.get("market_ret_5d", 0)
        if pd.notna(mr5):
            if mr5 > 0.03:
                signals.append("[大盘] 强势市场, 短线环境良好")
            elif mr5 < -0.03:
                signals.append("[大盘] 弱势市场, 注意系统性风险")

        # 资金
        mfi = row.get("mfi_14", 50)
        if pd.notna(mfi):
            if mfi < 25:
                signals.append("[资金] MFI超卖, 短期反弹概率增大")
            elif mfi > 80:
                signals.append("[资金] MFI超买, 短期回调风险")

        # 主力
        block = row.get("block_volume_ratio", 1)
        if pd.notna(block):
            if block > 2:
                signals.append("[主力] 异常放量, 关注主力动向")
            elif block > 1.5:
                signals.append("[主力] 温和放量, 资金关注度提升")

        # 板块
        vs_sec = row.get("stock_vs_sector_5d", 0)
        if pd.notna(vs_sec):
            if vs_sec < -0.05:
                signals.append("[板块] 大幅跑输板块, 存在补涨机会")
            elif vs_sec > 0.05:
                signals.append("[板块] 大幅跑赢板块, 注意获利回吐")

        # 期货
        div = row.get("stock_futures_divergence", 0)
        if self.futures_symbol and pd.notna(div):
            if abs(div) > 0.05:
                signals.append(f"[期货] 个股与{self.futures_symbol.upper()}背离{div:+.1%}, 关注回归")

        # RSI
        rsi = row.get("rsi_6", 50)
        if pd.notna(rsi):
            if rsi < 30:
                signals.append("[技术] RSI超卖, 短期技术反弹需求")
            elif rsi > 85:
                signals.append("[技术] RSI超买, 短期技术回调需求")

        if not signals:
            signals.append("[综合] 暂无明显信号, 建议观望")

        return signals

    def save(self, name=None):
        if name is None:
            plain = self.stock_code.split(".")[-1] if "." in self.stock_code else self.stock_code
            name = f"analyzer_{plain}"
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "stock_code": self.stock_code,
                "sector_name": self.sector_name,
                "futures_symbol": self.futures_symbol,
            }, f)
        print(f"[Analyzer] 模型已保存: {path}")

    def load(self, name=None):
        if name is None:
            plain = self.stock_code.split(".")[-1] if "." in self.stock_code else self.stock_code
            name = f"analyzer_{plain}"
        path = os.path.join(OUTPUT_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        print(f"[Analyzer] 模型已加载: {path}")


def run_single_stock_analysis(fetcher, stock_code, sector_name=None, futures_symbol=None):
    """完整的个股短线分析流程"""
    analyzer = SingleStockAnalyzer(fetcher, stock_code, sector_name, futures_symbol)

    # 1. 获取数据
    analyzer.fetch_data(lookback_days=120)

    # 2. 计算因子
    analyzer.compute_factors()

    # 3. 训练模型
    importance = analyzer.train_model()

    # 4. 生成报告
    analyzer.generate_report()

    # 5. 保存模型
    analyzer.save()

    return analyzer, importance
