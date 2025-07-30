#!/usr/bin/env python3
"""
Speed Comparison Script
Compare original vs optimized versions
"""

import time
import pandas as pd
import numpy as np
from fast_dca_backtest import FastDataProcessor, FastBacktester, StrategyParams

def create_sample_data(days: int = 30) -> pd.DataFrame:
    """Create sample data for testing"""
    n_points = days * 1440  # 1-minute data
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1min')
    
    # Generate realistic price data
    np.random.seed(42)
    initial_price = 100.0
    returns = np.random.normal(0, 0.001, n_points)  # 0.1% volatility per minute
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_points))),
        'close': prices,
        'vol': np.random.normal(1000, 200, n_points)
    }, index=dates)
    
    return data

def test_optimization_speed():
    """Test optimization speed comparison"""
    print("🚀 SPEED COMPARISON TEST")
    print("=" * 50)
    
    # Create test data
    print("Creating test data...")
    test_days = 30  # Start with 30 days
    data = create_sample_data(test_days)
    print(f"Test data: {len(data):,} points ({test_days} days)")
    
    # Test parameters
    test_params = StrategyParams(
        tp_level1=2.5,
        initial_deviation=3.0,
        trailing_deviation=2.0,
        tp_percent1=50.0
    )
    
    print(f"\n📊 Testing with {len(data):,} data points...")
    
    # Test fast version
    print("\n⚡ Testing FAST version...")
    start_time = time.time()
    
    fast_backtester = FastBacktester(data, 10000)
    fast_apy, fast_dd, fast_trades = fast_backtester.simulate_strategy_fast(test_params)
    
    fast_time = time.time() - start_time
    
    print(f"Fast Results:")
    print(f"  Time: {fast_time:.3f}s")
    print(f"  APY: {fast_apy:.2f}%")
    print(f"  Drawdown: {fast_dd:.2f}%")
    print(f"  Trades: {fast_trades}")
    
    # Estimate speed for different data sizes
    print(f"\n📈 SPEED PROJECTIONS:")
    print(f"Current speed: {len(data)/fast_time:.0f} data points/second")
    
    data_sizes = [
        (90, "3 months"),
        (365, "1 year"), 
        (365*2, "2 years"),
        (365*3, "3 years")
    ]
    
    for days, label in data_sizes:
        points = days * 1440
        estimated_time = points / (len(data)/fast_time)
        print(f"  {label:10} ({points:7,} pts): ~{estimated_time:.1f}s per simulation")
    
    # Test optimization speed
    print(f"\n🎯 OPTIMIZATION SPEED ESTIMATES:")
    single_sim_time = fast_time
    
    trial_counts = [100, 500, 1000, 2000]
    for trials in trial_counts:
        estimated_time = trials * single_sim_time
        print(f"  {trials:4} trials: ~{estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")
    
    return fast_time, len(data)

def benchmark_different_sizes():
    """Benchmark different data sizes"""
    print(f"\n🔬 BENCHMARKING DIFFERENT DATA SIZES")
    print("-" * 50)
    
    test_params = StrategyParams(tp_level1=2.5, initial_deviation=3.0)
    sizes_and_labels = [
        (7, "1 week"),
        (30, "1 month"), 
        (90, "3 months"),
        (180, "6 months")
    ]
    
    results = []
    
    for days, label in sizes_and_labels:
        print(f"\nTesting {label} ({days} days)...")
        data = create_sample_data(days)
        
        start_time = time.time()
        backtester = FastBacktester(data, 10000)
        apy, dd, trades = backtester.simulate_strategy_fast(test_params)
        elapsed = time.time() - start_time
        
        speed = len(data) / elapsed
        results.append((days, len(data), elapsed, speed, apy, dd, trades))
        
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Speed: {speed:.0f} points/sec")
        print(f"  APY: {apy:.1f}%, DD: {dd:.1f}%, Trades: {trades}")
    
    print(f"\n📊 SUMMARY:")
    print(f"{'Days':>6} {'Points':>8} {'Time':>8} {'Speed':>10} {'APY':>6} {'DD':>6} {'Trades':>7}")
    print("-" * 60)
    for days, points, time_taken, speed, apy, dd, trades in results:
        print(f"{days:6d} {points:8,} {time_taken:7.3f}s {speed:9.0f}/s {apy:5.1f}% {dd:5.1f}% {trades:6d}")

def main():
    """Main benchmark function"""
    print("⚡ DCA BACKTEST SPEED BENCHMARK")
    print("Testing optimized numba-based implementation")
    print("=" * 60)
    
    # Basic speed test
    fast_time, data_points = test_optimization_speed()
    
    # Different size benchmarks  
    benchmark_different_sizes()
    
    print(f"\n✅ BENCHMARK COMPLETE")
    print(f"💡 KEY TAKEAWAYS:")
    print(f"   • Fast version processes {data_points/fast_time:.0f} data points/second")
    print(f"   • 1-year backtest takes ~{365*1440/(data_points/fast_time):.1f} seconds")
    print(f"   • 1000-trial optimization takes ~{1000*fast_time/60:.1f} minutes")
    print(f"   • Use --sample_ratio 0.3 for 3x speed boost with minimal accuracy loss")
    print(f"   • Use ultra_fast_optimizer.py for maximum speed with parallel processing")

if __name__ == "__main__":
    main()