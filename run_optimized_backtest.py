#!/usr/bin/env python3
"""
Run optimized backtest with automatic two-phase optimization
This ensures we always get good parameters that use ~100% of account safely
"""

import subprocess
import sys
import json
from pathlib import Path

def main():
    """Run two-phase optimization then show results"""
    
    print("🚀 RUNNING OPTIMIZED DCA BACKTEST")
    print("=" * 80)
    print("This will automatically:")
    print("1. Run two-phase optimization to find best parameters")
    print("2. Validate position sizes stay under 100%")
    print("3. Show final results with ~100% account usage")
    print("=" * 80)
    print()
    
    # Check if we have data
    data_path = "data/SOLUSDT_1m.csv"
    if not Path(data_path).exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please ensure you have the SOL data file.")
        return
    
    # Run two-phase optimization
    print("📊 Starting two-phase optimization...")
    print("This will find parameters that maximize APY while using ~100% of account")
    print()
    
    cmd = [
        sys.executable,
        "two_phase_optimizer.py",
        "--phase1_trials", "100",
        "--phase2_trials", "50"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Optimization failed!")
        print(result.stderr)
        return
    
    print(result.stdout)
    
    # Show the results
    print("\n" + "=" * 80)
    print("📈 SHOWING OPTIMIZATION RESULTS")
    print("=" * 80)
    
    cmd = [sys.executable, "show_optimization_results.py"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Failed to show results!")
        print(result.stderr)
        return
    
    print(result.stdout)

if __name__ == "__main__":
    main()
