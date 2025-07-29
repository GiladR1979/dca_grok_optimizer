#!/usr/bin/env python3
"""
GPU vs CPU Verification Script
Compares GPU and CPU simulation results to verify correctness
"""

import pandas as pd
import numpy as np
import time
from dca_backtest import *
import torch


def verify_gpu_cpu_consistency(data_path: str, test_days: int = 150):
    """Compare GPU vs CPU results for the same parameters"""

    print("🔍 GPU vs CPU Verification Test")
    print("=" * 50)

    # Load data
    print(f"Loading data from {data_path}...")
    full_data = DataProcessor.load_data(data_path)
    print(f"Loaded {len(full_data)} total data points")

    # Sample only the specified number of days for testing
    minutes_per_day = 1440  # 1-minute data
    sample_size = test_days * minutes_per_day

    if len(full_data) > sample_size:
        # Take the last N days for more recent market conditions
        data = full_data.tail(sample_size).copy()
        print(f"Using last {test_days} days ({len(data)} data points) for testing")
    else:
        data = full_data
        actual_days = len(data) / minutes_per_day
        print(f"Dataset smaller than {test_days} days - using all {actual_days:.1f} days")

    # Initialize backtester
    backtester = Backtester(data, 10000)

    # Test parameters (using simple, fixed parameters)
    test_params = StrategyParams(
        base_percent=1.0,
        initial_deviation=3.0,
        step_multiplier=1.5,
        volume_multiplier=1.2,
        max_safeties=3,  # Reduced for easier debugging
        trailing_deviation=2.0,
        tp_level1=2.0,
        tp_percent1=50.0,
        tp_percent2=30.0,
        tp_percent3=20.0,
        rsi_entry_threshold=40.0,
        rsi_safety_threshold=30.0,
        rsi_exit_threshold=70.0,
        fees=0.075
    )

    print(f"\nTest Parameters:")
    print(f"  TP1: {test_params.tp_level1}%")
    print(f"  Trailing: {test_params.trailing_deviation}%")
    print(f"  Max Safeties: {test_params.max_safeties}")

    # Run CPU simulation
    print(f"\n🖥️  Running CPU simulation...")
    cpu_start = time.time()
    cpu_apy, cpu_dd, cpu_balance_history, cpu_trades = backtester.simulate_strategy(test_params)
    cpu_time = time.time() - cpu_start

    cpu_final_balance = cpu_balance_history[-1][1] if cpu_balance_history else 10000

    print(f"CPU Results:")
    print(f"  APY: {cpu_apy:.2f}%")
    print(f"  Max DD: {cpu_dd:.2f}%")
    print(f"  Final Balance: ${cpu_final_balance:,.2f}")
    print(f"  Total Trades: {len(cpu_trades)}")
    print(f"  Time: {cpu_time:.1f}s")

    # Run GPU simulation
    if GPU_AVAILABLE:
        print(f"\n🚀 Running GPU simulation...")
        try:
            gpu_simulator = GPUBatchSimulator(data, backtester.indicators, 10000)
            gpu_start = time.time()
            gpu_results = gpu_simulator.simulate_batch([test_params])
            gpu_time = time.time() - gpu_start

            if gpu_results:
                gpu_apy, gpu_dd = gpu_results[0]

                print(f"GPU Results:")
                print(f"  APY: {gpu_apy:.2f}%")
                print(f"  Max DD: {gpu_dd:.2f}%")
                print(f"  Time: {gpu_time:.1f}s")

                # Compare results
                print(f"\n📊 Comparison:")
                apy_diff = abs(cpu_apy - gpu_apy)
                dd_diff = abs(cpu_dd - gpu_dd)

                print(f"  APY Difference: {apy_diff:.2f}%")
                print(f"  DD Difference: {dd_diff:.2f}%")

                # Tolerance check
                if apy_diff < 0.1 and dd_diff < 0.1:
                    print(f"  ✅ Results are consistent!")
                    return True
                else:
                    print(f"  ❌ Results are inconsistent!")
                    print(f"     This indicates GPU simulation has bugs")
                    return False
            else:
                print(f"  ❌ GPU simulation returned no results")
                return False

        except Exception as e:
            print(f"  ❌ GPU simulation failed: {e}")
            return False
    else:
        print(f"  ⚠️  No GPU available for comparison")
        return False


def debug_single_step():
    """Debug a single simulation step"""
    print("\n🐛 Single Step Debug")
    print("-" * 30)

    # Create minimal test data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    prices = np.random.normal(100, 5, 100)  # Random walk around $100

    test_data = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'vol': np.random.normal(1000, 100, 100)
    }, index=dates)

    print(f"Test data: {len(test_data)} points, price range: ${prices.min():.2f}-${prices.max():.2f}")

    # Simple test
    backtester = Backtester(test_data, 10000)

    test_params = StrategyParams(
        base_percent=5.0,  # Larger orders for easier detection
        initial_deviation=2.0,
        max_safeties=2,
        tp_level1=1.0,  # Easy to hit
        trailing_deviation=1.0,
        fees=0.0  # No fees for simpler math
    )

    apy, dd, balance_history, trades = backtester.simulate_strategy(test_params)

    print(f"Simple test results:")
    print(f"  APY: {apy:.2f}%")
    print(f"  Trades: {len(trades)}")
    print(f"  Final: ${balance_history[-1][1]:,.2f}")

    if len(trades) > 0:
        print(f"  First trade: {trades[0].action} {trades[0].amount_coin:.4f} at ${trades[0].price:.2f}")
        print(f"  Last trade: {trades[-1].action} {trades[-1].amount_coin:.4f} at ${trades[-1].price:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--test_days', type=int, default=150,
                        help='Number of days to test (default: 150)')
    args = parser.parse_args()

    # Run verification
    consistent = verify_gpu_cpu_consistency(args.data_path, args.test_days)

    # Run simple debug
    debug_single_step()

    if not consistent:
        print(f"\n⚠️  RECOMMENDATION: Use CPU-only mode until GPU bugs are fixed")
        print(f"   Add --no_gpu flag to your multi-coin script")
    else:
        print(f"\n✅ GPU simulation appears to be working correctly")


if __name__ == "__main__":
    main()
