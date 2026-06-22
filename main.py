"""
A股多因子量化选股系统 - 主入口
用法: python main.py [fetch|train|backtest|predict|dashboard|hot|all]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import gradio as gr
from config import OUTPUT_DIR

def cmd_fetch():
    """1. 数据采集 + 因子计算"""
    from data.fetcher import DataFetcher
    from data.factors import FactorEngine
    from data.preprocess import winsorize, standardize_cross_section
    
    fetcher = DataFetcher()
    
    # 获取股票列表
    codes = fetcher.get_stock_list()
    
    # 获取行情数据
    daily = fetcher.get_daily_data(codes)
    if len(daily) == 0:
        print("[Error] 行情数据获取失败，请检查网络")
        return None
    
    # 计算因子
    engine = FactorEngine(daily)
    factor_df = engine.compute_all_factors()
    
    # 准备标签
    factor_df = engine.prepare_labels(forward_period=20)
    
    # 预处理
    factor_cols = engine.get_factor_list()
    for col in factor_cols:
        factor_df[col] = factor_df.groupby("date")[col].transform(winsorize)
    
    factor_df = standardize_cross_section(factor_df, factor_cols)
    
    print(f"[Done] 数据准备完成, 保存至 {OUTPUT_DIR}")
    factor_df.to_parquet(os.path.join(OUTPUT_DIR, "factor_data.parquet"), index=False)
    
    return factor_df

def cmd_train():
    """2. 模型训练"""
    import pandas as pd
    from data.factors import FactorEngine
    from models.train import StockRankingModel
    
    path = os.path.join(OUTPUT_DIR, "factor_data.parquet")
    if not os.path.exists(path):
        print("[Error] 请先运行 fetch 获取数据")
        return
    
    factor_df = pd.read_parquet(path)
    factor_list = [c for c in factor_df.columns 
                   if c not in ["date", "code", "label", "open", "high", "low", "close", "volume", "amount"]]
    
    model = StockRankingModel()
    importance = model.train(factor_df, factor_list)
    model.save()
    
    return model

def cmd_backtest():
    """3. 回测"""
    import pandas as pd
    from data.fetcher import DataFetcher
    from models.train import StockRankingModel
    from models.backtest import BacktestEngine
    
    path = os.path.join(OUTPUT_DIR, "factor_data.parquet")
    if not os.path.exists(path):
        print("[Error] 请先运行 fetch 获取数据")
        return
    
    factor_df = pd.read_parquet(path)
    factor_list = [c for c in factor_df.columns 
                   if c not in ["date", "code", "label", "open", "high", "low", "close", "volume", "amount"]]
    
    # 加载模型
    model = StockRankingModel()
    try:
        model.load()
    except:
        print("[Error] 请先运行 train 训练模型")
        return
    
    # 预测
    scored = model.predict(factor_df, factor_list)
    
    # 回测 (需要原始行情数据)
    daily_data = factor_df[["date", "code", "close"]].drop_duplicates()
    
    engine = BacktestEngine(scored, daily_data)
    engine.run()
    engine.compute_metrics()
    
    # 获取基准
    fetcher = DataFetcher()
    benchmark = fetcher.get_index_data("000300")
    engine.plot(benchmark)
    
    return engine

def cmd_predict():
    """4. 生成选股信号"""
    import pandas as pd
    from models.train import StockRankingModel
    from models.predict import StockSelector
    from data.factors import FactorEngine
    
    path = os.path.join(OUTPUT_DIR, "factor_data.parquet")
    if not os.path.exists(path):
        print("[Error] 请先运行 fetch 获取数据")
        return
    
    factor_df = pd.read_parquet(path)
    
    model = StockRankingModel()
    try:
        model.load()
    except:
        print("[Error] 请先运行 train 训练模型")
        return
    
    selector = StockSelector(model, None)
    selected, report = selector.generate_report(factor_df)
    return selected

def cmd_dashboard():
    """5. 启动看板"""
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
    from data.fetcher import DataFetcher
    from models.hot_sector import run_hot_sector_pipeline
    fetcher = DataFetcher()
    picks, sector_stats, importance = run_hot_sector_pipeline(fetcher, use_all_stocks=False)
    if picks is not None:
        print()
        print("=" * 60)
        print("  热点板块短线选股 - 完成!")
        print("=" * 60)
    return picks


def cmd_hot_all():
    from data.fetcher import DataFetcher
    from models.hot_sector import run_hot_sector_pipeline
    fetcher = DataFetcher()
    picks, sector_stats, importance = run_hot_sector_pipeline(fetcher, use_all_stocks=True)
    if picks is not None:
        print()
        print("=" * 60)
        print("  全A股热点板块短线选股 - 完成!")
        print("=" * 60)
    return picks


def cmd_analyze():
    import sys
    from data.fetcher import DataFetcher
    from models.stock_analyzer import run_single_stock_analysis

    # 默认: 江西铜业 600362, 有色金属板块, 沪铜联动
    stock = sys.argv[2] if len(sys.argv) > 2 else "600362"
    sector = sys.argv[3] if len(sys.argv) > 3 else None
    futures = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"\n分析目标: {stock}")
    print(f"板块: {sector or '(自动推断)'}")
    print(f"期货联动: {futures or '(自动推断)'}")

    fetcher = DataFetcher()
    analyzer, importance = run_single_stock_analysis(
        fetcher, stock, sector_name=sector, futures_symbol=futures
    )
    return analyzer


def cmd_capital():
    from data.fetcher import DataFetcher
    from models.capital_flow_sector import run_capital_flow_pipeline
    fetcher = DataFetcher()
    selector, picks = run_capital_flow_pipeline(fetcher)
    print()
    print("=" * 80)
    print("  资金认可度板块筛选 - 完成!")
    print("=" * 80)
    return selector

def cmd_all():
    """完整流程"""
    cmd_fetch()
    cmd_train()
    cmd_backtest()
    cmd_predict()

if __name__ == "__main__":
    cmds = {
        "capital": cmd_capital,
        "analyze": cmd_analyze,
        "hot": cmd_hot,
        "hot_all": cmd_hot_all,
        "fetch": cmd_fetch,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "predict": cmd_predict,
        "dashboard": cmd_dashboard,
        "all": cmd_all,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in cmds:
        cmds[sys.argv[1]]()
    else:
        print("用法: python main.py [analyze|fetch|train|backtest|predict|dashboard|hot|all]")
        print("  fetch     - 采集数据 + 计算因子")
        print("  train     - 训练模型")
        print("  backtest  - 回测评估")
        print("  predict   - 生成选股信号")
        print("  dashboard - 启动看板")
        print("  capital   - 资金认可度板块筛选+热门股")
        print("  analyze   - 个股短线量化分析(默认江西铜业)")
        print("  hot       - 热点板块短线选股(HS300)")
        print("  hot_all   - 全A股热点板块选股")
        print("  all       - 执行完整流程")
