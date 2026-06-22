"""
选股预测模块 - 生成每日/每月选股信号
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import OUTPUT_DIR

class StockSelector:
    def __init__(self, model, factor_engine):
        self.model = model
        self.engine = factor_engine
    
    def select(self, factor_df, top_n=30):
        """选出 top N 股票"""
        factor_list = self.model.feature_names
        scored = self.model.predict(factor_df, factor_list)
        
        # 最新截面
        latest_date = scored["date"].max()
        latest = scored[scored["date"] == latest_date].copy()
        selected = latest.nlargest(top_n, "score")
        
        return selected[["code", "score", "rank"]].reset_index(drop=True)
    
    def generate_report(self, factor_df, top_n=30):
        """生成选股报告"""
        selected = self.select(factor_df, top_n)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"  A股量化选股信号 - {datetime.now().strftime('%Y-%m-%d')}")
        report_lines.append("=" * 60)
        report_lines.append(f"\n  选股数量: {len(selected)}")
        report_lines.append(f"\n  {'排名':<6} {'代码':<10} {'得分':<10}")
        report_lines.append("  " + "-" * 30)
        
        for i, (_, row) in enumerate(selected.iterrows()):
            report_lines.append(
                f"  {i+1:<6} {row['code']:<10} {row['score']:.4f}"
            )
        
        report_lines.append("\n" + "=" * 60)
        report_lines.append("  风险提示: 量化选股仅供参考，不构成投资建议")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # 保存
        path = os.path.join(OUTPUT_DIR, "stock_picks.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        
        # 保存CSV
        csv_path = os.path.join(OUTPUT_DIR, "stock_picks.csv")
        selected.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        print(report)
        return selected, report
