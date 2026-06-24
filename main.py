"""
A股多因子量化选股系统 - 主入口 (v4 - 支持实时数据)
用法: python main.py [fetch|train|backtest|predict|dashboard|hot|all|realtime|realtime_hot|realtime_capital]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
from config import OUTPUT_DIR


def cmd_fetch():
    """1. 数据采集 + 因子计算 (baostock历史数据)"""
    from data.fetcher import DataFetcher
    from data.factors import FactorEngine
    from data.preprocess import winsorize, standardize_cross_section
    
    fetcher = DataFetcher()
    codes = fetcher.get_stock_list()
    daily = fetcher.get_daily_data(codes)
    if len(daily) == 0:
        print("[Error] 行情数据获取失败，请检查网络")
        return None
    
    engine = FactorEngine(daily)
    factor_df = engine.compute_all_factors()
    factor_df = engine.prepare_labels(forward_period=20)
    
    factor_cols = engine.get_factor_list()
    for col in factor_cols:
        factor_df[col] = factor_df.groupby("date")[col].transform(winsorize)
    factor_df = standardize_cross_section(factor_df, factor_cols)
    
    print(f"[Done] 数据准备完成, 保存至 {OUTPUT_DIR}")
    factor_df.to_parquet(os.path.join(OUTPUT_DIR, "factor_data.parquet"), index=False)
    return factor_df


def cmd_realtime():
    """1b. 实时数据采集 (akshare - 无需token)"""
    from data.akshare_fetcher import AKSharesetcher
    import pandas as pd
    
    fetcher = AKSharesetcher(force_refresh=True)
    pkg = fetcher.get_today_data_package()
    
    # 保存实时数据
    spot = pkg.get("spot", pd.DataFrame())
    sectors = pkg.get("sectors", pd.DataFrame())
    fund_flow = pkg.get("fund_flow", pd.DataFrame())
    
    if not spot.empty:
        spot.to_csv(os.path.join(OUTPUT_DIR, "realtime_spot.csv"), index=False, encoding="utf-8-sig")
        print(f"[Done] 实时行情保存: {len(spot)} 只")
    if not sectors.empty:
        sectors.to_csv(os.path.join(OUTPUT_DIR, "realtime_sectors.csv"), index=False, encoding="utf-8-sig")
        print(f"[Done] 板块行情保存: {len(sectors)} 个")
    if not fund_flow.empty:
        fund_flow.to_csv(os.path.join(OUTPUT_DIR, "realtime_fund_flow.csv"), index=False, encoding="utf-8-sig")
        print(f"[Done] 资金流向保存: {len(fund_flow)} 条")
    
    return pkg


def cmd_realtime_hot():
    """实时热点板块选股 (akshare 实时数据)"""
    from data.akshare_fetcher import AKSharesetcher, quick_sector_hot, quick_realtime_top
    import pandas as pd
    
    fetcher = AKSharesetcher(force_refresh=True)
    pkg = fetcher.get_today_data_package()
    
    # 板块热度综合排名
    hot_sectors = quick_sector_hot(fetcher)
    if not hot_sectors.empty:
        hot_sectors.to_csv(os.path.join(OUTPUT_DIR, "realtime_hot_sectors.csv"), index=False, encoding="utf-8-sig")
        print("\n=== 热点板块 Top10 (综合评分) ===")
        cols = [c for c in ["sector_name", "pct_chg", "main_net_inflow", "hot_score"] if c in hot_sectors.columns]
        print(hot_sectors.head(10)[cols].to_string())
    
    # 实时涨幅Top
    top_stocks = quick_realtime_top(fetcher, n=30)
    if not top_stocks.empty:
        top_stocks.to_csv(os.path.join(OUTPUT_DIR, "realtime_top_stocks.csv"), index=False, encoding="utf-8-sig")
    
    return hot_sectors


def cmd_realtime_capital():
    """实时资金认可度分析 (akshare)"""
    from data.akshare_fetcher import AKSharesetcher
    import pandas as pd
    
    fetcher = AKSharesetcher(force_refresh=True)
    
    spot = fetcher.get_all_stocks_spot()
    fund_flow = fetcher.get_sector_fund_flow()
    sectors = fetcher.get_sector_performance()
    
    # 资金认可度 = 主力净流入排名 * 0.5 + 涨幅排名 * 0.3 + 上涨家数比例 * 0.2
    if not fund_flow.empty:
        ff = fund_flow.copy()
        for col in ["main_net_inflow", "pct_chg"]:
            if col in ff.columns:
                ff[f"{col}_rank"] = ff[col].rank(ascending=False, pct=True)
        
        score_cols = [c for c in ["main_net_inflow_rank", "pct_chg_rank"] if c in ff.columns]
        if score_cols:
            ff["capital_score"] = ff[score_cols].mean(axis=1)
            ff = ff.sort_values("capital_score", ascending=False)
        
        ff.to_csv(os.path.join(OUTPUT_DIR, "realtime_capital_flow.csv"), index=False, encoding="utf-8-sig")
        print("\n=== 资金认可度 Top10 板块 ===")
        print_cols = [c for c in ["sector_name", "pct_chg", "main_net_inflow", "capital_score"] if c in ff.columns]
        print(ff.head(10)[print_cols].to_string())
    
    # 板块内选股
    if not sectors.empty and not spot.empty:
        top_sectors = sectors.nlargest(5, "pct_chg")["sector_name"].tolist() if "pct_chg" in sectors.columns else []
        print(f"\n热点板块: {top_sectors}")
    
    return fund_flow


def cmd_train():
    """2. 模型训练"""
    import pandas as pd
    from models.train import StockRankingModel
    
    path = os.path.join(OUTPUT_DIR, "factor_data.parquet")
    if not os.path.exists(path):
        print("[Error] 请先运行 fetch 获取数据")
        return
    
    factor_df = pd.read_parquet(path)
    factor_list = [c for c in factor_df.columns 
                   if c not in ["date", "code", "label", "open", "high", "low", "close", "volume", "amount"]]
    
    model = StockRankingModel()
    model.train(factor_df, factor_list)
    model.save()


def cmd_backtest():
    """3. 回测"""
    import pandas as pd
    from data.fetcher import DataFetcher
    from models.train import StockRankingModel
    from models.backtest import BacktestEngine
    
    path = os.path.join(OUTPUT_DIR, "factor_data.parquet")
    if not os.path.exists(path):
        print("[Error] 请先运行 fetch")
        return
    
    factor_df = pd.read_parquet(path)
    factor_list = [c for c in factor_df.columns 
                   if c not in ["date", "code", "label", "open", "high", "low", "close", "volume", "amount"]]
    
    model = StockRankingModel()
    try:
        model.load()
    except:
        print("[Error] 请先运行 train")
        return
    
    scored = model.predict(factor_df, factor_list)
    daily_data = factor_df[["date", "code", "close"]].drop_duplicates()
    
    engine = BacktestEngine(scored, daily_data)
    engine.run()
    engine.compute_metrics()
    
    fetcher = DataFetcher()
    benchmark = fetcher.get_index_data("000300")
    engine.plot(benchmark)


def cmd_predict():
    """4. 生成选股信号"""
    import pandas as pd
    from models.train import StockRankingModel
    from models.predict import StockSelector
    
    path = os.path.join(OUTPUT_DIR, "factor_data.parquet")
    if not os.path.exists(path):
        print("[Error] 请先运行 fetch")
        return
    
    factor_df = pd.read_parquet(path)
    
    model = StockRankingModel()
    try:
        model.load()
    except:
        print("[Error] 请先运行 train")
        return
    
    selector = StockSelector(model, None)
    selected, report = selector.generate_report(factor_df)
    return selected


def cmd_dashboard():
    """5. 启动Gradio看板"""
    import gradio as gr
    from dashboard.app import create_dashboard
    app = create_dashboard()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, prevent_thread_lock=True, show_error=True)
    print("\nDashboard: http://localhost:7860")
    print("Ctrl+C to stop\n")
    import time
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nDashboard stopped")


def cmd_hot():
    """热点板块选股 (baostock历史数据)"""
    from data.fetcher import DataFetcher
    from models.hot_sector import run_hot_sector_pipeline
    fetcher = DataFetcher()
    picks, sector_stats, importance = run_hot_sector_pipeline(fetcher, use_all_stocks=False)
    if picks is not None:
        print("\n" + "=" * 60)
        print("  热点板块短线选股 - 完成!")
        print("=" * 60)
    return picks


def cmd_hot_all():
    """全A股热点板块选股"""
    from data.fetcher import DataFetcher
    from models.hot_sector import run_hot_sector_pipeline
    fetcher = DataFetcher()
    picks, sector_stats, importance = run_hot_sector_pipeline(fetcher, use_all_stocks=True)
    return picks


def cmd_analyze():
    """个股短线量化分析"""
    import sys
    from data.fetcher import DataFetcher
    from models.stock_analyzer import run_single_stock_analysis
    
    stock = sys.argv[2] if len(sys.argv) > 2 else "600362"
    sector = sys.argv[3] if len(sys.argv) > 3 else None
    futures = sys.argv[4] if len(sys.argv) > 4 else None
    
    print(f"\n分析目标: {stock}")
    fetcher = DataFetcher()
    analyzer, importance = run_single_stock_analysis(
        fetcher, stock, sector_name=sector, futures_symbol=futures
    )
    return analyzer


def cmd_capital():
    """资金认可度板块筛选"""
    from data.fetcher import DataFetcher
    from models.capital_flow_sector import run_capital_flow_pipeline
    fetcher = DataFetcher()
    selector, picks = run_capital_flow_pipeline(fetcher)
    print("\n" + "=" * 80)
    print("  资金认可度板块筛选 - 完成!")
    print("=" * 80)
    return selector


def cmd_all():
    """完整流程 (历史数据)"""
    cmd_fetch()
    cmd_train()
    cmd_backtest()
    cmd_predict()


if __name__ == "__main__":
    cmds = {
        "fetch": cmd_fetch,
        "realtime": cmd_realtime,
        "realtime_hot": cmd_realtime_hot,
        "realtime_capital": cmd_realtime_capital,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "dashboard": cmd_dashboard,
        "hot": cmd_hot,
        "hot_all": cmd_hot_all,
        "capital": cmd_capital,
        "analyze": cmd_analyze,
        "all": cmd_all,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in cmds:
        cmds[sys.argv[1]]()
    else:
        print("用法: python main.py [命令]")
        print("  历史数据模式 (baostock):")
        print("    fetch     - 采集历史数据 + 计算因子")
        print("    train     - 训练模型")
        print("    backtest  - 回测评估")
        print("    predict   - 生成选股信号")
        print("    hot       - 热点板块选股(HS300)")
        print("    hot_all   - 全A股热点板块选股")
        print("    capital   - 资金认可度板块筛选")
        print("    analyze   - 个股短线分析")
        print("    dashboard - 启动看板")
        print("    all       - 完整流程")
        print("  实时数据模式 (akshare - 免费, 无需token):")
        print("    realtime         - 采集实时全A股行情")
        print("    realtime_hot     - 实时热点板块选股")
        print("    realtime_capital - 实时资金认可度分析")
