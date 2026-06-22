# -*- coding: utf-8 -*-
"""
A股量化选股系统 - 每日自动更新流水线
用法: python daily_pipeline.py [--skip-fetch] [--skip-deploy]
"""
import sys, os, argparse, subprocess, time
from datetime import datetime

ROOT = r"D:\quant_a_stock"
PYTHON = r"C:\Users\20686\Documents\Codex\2026-06-21\ni\dl_env\Scripts\python.exe"
sys.path.insert(0, ROOT)

def run_step(name, args):
    """运行一个子步骤"""
    print(f"\n{'='*60}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([PYTHON] + args, cwd=ROOT, capture_output=False, timeout=600)
    elapsed = time.time() - t0
    status = "✅ 成功" if result.returncode == 0 else f"❌ 失败 (code={result.returncode})"
    print(f"  {status} | 耗时: {elapsed:.0f}秒")
    return result.returncode == 0

def run_pipeline(skip_fetch=False, skip_deploy=False):
    """运行完整流水线"""
    print(f"\n{'#'*60}")
    print(f"  A股量化选股系统 - 每日自动流水线")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    all_ok = True
    
    # Step 1: 采集数据
    if not skip_fetch:
        if not run_step("数据采集", ["main.py", "fetch"]):
            all_ok = False
    
    # Step 2: 热点板块选股
    if not run_step("热点板块选股", ["main.py", "hot"]):
        all_ok = False
    
    # Step 3: 资金认可度分析
    if not run_step("资金认可度分析", ["main.py", "capital"]):
        all_ok = False
    
    # Step 4: 月度选股
    if not run_step("月度多因子选股", ["main.py", "all"]):
        all_ok = False
    
    # Step 5: 生成HTML看板
    print(f"\n{'='*60}")
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] 生成HTML看板")
    print(f"{'='*60}")
    result = subprocess.run([PYTHON, "build_html.py"], cwd=ROOT, capture_output=True, timeout=120)
    print(result.stdout.decode("utf-8", errors="ignore"))
    if result.returncode != 0:
        print(result.stderr.decode("utf-8", errors="ignore"))
        all_ok = False
    
    # Step 6: 部署到 Gitee Pages
    if not skip_deploy and all_ok:
        if not run_step("部署到 Gitee Pages", ["deploy_gitee.py"]):
            all_ok = False
    
    print(f"\n{'#'*60}")
    print(f"  流水线完成: {'✅ 全部成功' if all_ok else '❌ 部分失败'}")
    print(f"  结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    return all_ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true", help="跳过数据采集")
    parser.add_argument("--skip-deploy", action="store_true", help="跳过Gitee部署")
    args = parser.parse_args()
    run_pipeline(skip_fetch=args.skip_fetch, skip_deploy=args.skip_deploy)