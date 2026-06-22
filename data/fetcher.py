"""
数据采集模块 v3 - baostock 串行下载 (session-based, 不支持真正并发)
"""
import os, time, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import baostock as bs
from config import DATA_DIR, START_DATE, END_DATE, STOCK_POOL

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

def _fmt_date(yyyymmdd):
    s = str(yyyymmdd)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def _bs_code_to_plain(code):
    return code.split(".")[1] if "." in code else code

class DataFetcher:
    def __init__(self, max_workers=1):
        self.max_workers = max_workers
        os.makedirs(DATA_DIR, exist_ok=True)
        self._login()
    
    def _login(self):
        lg = bs.login()
        if lg.error_code != "0":
            print(f"[Data] baostock 登录失败: {lg.error_msg}")
        else:
            print("[Data] baostock 登录成功")
    
    def _cache_path(self, key):
        return os.path.join(DATA_DIR, f"{key}.pkl")
    
    def _load_cache(self, key):
        path = self._cache_path(key)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    
    def _save_cache(self, key, data):
        path = self._cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def _fetch_one_stock(self, bs_code, start_fmt, end_fmt):
        try:
            plain_code = _bs_code_to_plain(bs_code)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_fmt, end_date=end_fmt,
                frequency="d", adjustflag="2"
            )
            if rs.error_code != "0":
                return None
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=rs.fields)
            df["code"] = plain_code
            df.rename(columns={
                "date": "date", "open": "open", "close": "close",
                "high": "high", "low": "low", "volume": "volume",
                "amount": "amount", "turn": "turnover_rate",
            }, inplace=True)
            for col in ["open","high","low","close","volume","amount","turnover_rate"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            cols = ["date","code","open","high","low","close","volume","amount","turnover_rate"]
            return df[[c for c in cols if c in df.columns]]
        except:
            return None
    
    def get_stock_list(self):
        cache = self._load_cache("stock_list_bs")
        if cache is not None:
            print(f"[Data] 股票列表 (缓存): {len(cache)} 只")
            return cache
        
        print("[Data] 获取沪深300成分股 (baostock)...")
        try:
            rs = bs.query_hs300_stocks()
            codes = []
            while rs.next():
                codes.append(rs.get_row_data()[1])
            if not codes:
                raise ValueError("空列表")
            print(f"  共 {len(codes)} 只")
        except Exception as e:
            print(f"  沪深300获取失败: {e}, 降级为测试股池")
            codes = ["sh.600519","sh.600036","sh.601318","sh.600276","sh.600900",
                     "sz.000858","sz.000333","sz.002415","sz.300750","sz.000001"]
        self._save_cache("stock_list_bs", codes)
        return codes
    
    
    def get_all_stocks(self):
        """获取全A股列表"""
        cache = self._load_cache("stock_list_all")
        if cache is not None:
            print(f"[Data] 全A股列表 (缓存): {len(cache)} 只")
            return cache

        print("[Data] 获取全A股股票列表...")
        try:
            rs = bs.query_stock_basic()
            codes = []
            while rs.next():
                row = rs.get_row_data()
                # row: [code, code_name, ipoDate, outDate, type, status]
                # type=1: stock, type=2: index, type=3: other
                if len(row) > 4 and row[4] == "1":
                    code = row[0]
                    if len(code) == 9 and "." in code:
                        codes.append(code)
            if not codes:
                raise ValueError("empty")
            print(f"  共 {len(codes)} 只")
        except Exception as e:
            print(f"  全A股获取失败: {e}, 降级为沪深300")
            codes = self.get_stock_list()

        self._save_cache("stock_list_all", codes)
        return codes

    def get_industry_data(self):
        """获取行业分类"""
        cache = self._load_cache("industry_bs")
        if cache is not None:
            print(f"[Data] 行业分类 (缓存): {len(cache)} 条")
            return cache

        print("[Data] 获取行业分类...")
        try:
            rs = bs.query_stock_industry()
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows, columns=rs.fields)
                self._save_cache("industry_bs", df)
                print(f"  共 {len(df)} 条")
                return df
        except Exception as e:
            print(f"  行业获取失败: {e}")
        return None

    def get_daily_data(self, codes, start=None, end=None, use_cache=True):
        start = start or START_DATE
        end = end or END_DATE
        start_fmt = _fmt_date(start)
        end_fmt = _fmt_date(end)
        cache_key = f"daily_bs_{len(codes)}_{start}_{end}"
        
        if use_cache:
            cache = self._load_cache(cache_key)
            if cache is not None:
                print(f"[Data] 行情数据 (缓存): {cache['date'].nunique()} 天")
                return cache
        
        print(f"[Data] 串行下载 {len(codes)} 只股票行情 ({start_fmt} ~ {end_fmt})...")
        results = []
        failed = 0
        t0 = time.time()
        
        for code in tqdm(codes, desc="下载行情"):
            result = self._fetch_one_stock(code, start_fmt, end_fmt)
            if result is not None and len(result) > 0:
                results.append(result)
            else:
                failed += 1
        
        elapsed = time.time() - t0
        
        if not results:
            raise RuntimeError(f"所有 {len(codes)} 只下载失败，请检查网络")
        
        result = pd.concat(results, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values(["code", "date"]).reset_index(drop=True)
        
        print(f"  成功: {len(result['code'].unique())} 只, 失败: {failed}, "
              f"数据量: {len(result)} 行, 耗时: {elapsed:.0f}s")
        print(f"  日期: {result['date'].min().date()} ~ {result['date'].max().date()}")
        
        self._save_cache(cache_key, result)
        return result
    
    def get_index_data(self, symbol="000300", start=None, end=None):
        start = start or START_DATE
        end = end or END_DATE
        start_fmt = _fmt_date(start)
        end_fmt = _fmt_date(end)
        cache = self._load_cache(f"index_bs_{symbol}")
        if cache is not None:
            print(f"[Data] 指数 {symbol} (缓存)")
            return cache
        
        print(f"[Data] 获取指数 {symbol}...")
        try:
            bs_code = f"sh.{symbol}"
            rs = bs.query_history_k_data_plus(
                bs_code, "date,open,high,low,close,volume,amount",
                start_date=start_fmt, end_date=end_fmt,
                frequency="d", adjustflag="2"
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=rs.fields)
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open","high","low","close","volume","amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            self._save_cache(f"index_bs_{symbol}", df)
            print(f"  共 {len(df)} 条")
            return df
        except Exception as e:
            print(f"  指数获取失败: {e}")
            return pd.DataFrame()
