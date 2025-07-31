#!/usr/bin/env python3
"""
Optimize DCA Strategy with Volume Filters Enabled
Based on drawdown analysis, test volume filters to improve APY
"""

import numpy as np
from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams
from ultra_fast_optimizer import UltraFastOptimizer
import json
from datetime import datetime

def test_volume_filter_combinations():
    """Test different volume filter combinations based on analysis"""
    
    print("🔬 TESTING VOLUME FILTER COMBINATIONS")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    backtester = FastBacktester(data, 10000)
    
    # Base parameters from best optimization
    base_params = {
        'base_percent': 1.0,
        'initial_deviation': 1.0,
        'trailing_deviation': 4.5,
        'tp_level1': 8.0,
        'tp_percent1': 50,
        'tp_percent2': 30,
        'tp_percent3': 20,
        'rsi_entry_threshold': 55.0,
        'rsi_safety_threshold': 55.0,
        'sma_trend_filter': False,
        'ema_trend_filter': False,
        'atr_volatility_filter': False,
        'higher_highs_filter': False
    }
    
    # Test different volume filter configurations
    test_configs = [
        {
            'name': 'No Filters (Baseline)',
            'volume_confirmation': False,
            'volume_sma_period': 20
        },
        {
            'name': 'Volume SMA(10) Filter',
            'volume_confirmation': True,
            'volume_sma_period': 10
        },
        {
            'name': 'Volume SMA(20) Filter',
            'volume_confirmation': True,
            'volume_sma_period': 20
        },
        {
            'name': 'Volume SMA(30) Filter',
            'volume_confirmation': True,
            'volume_sma_period': 30
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting: {config['name']}")
        
        # Create parameters (exclude 'name' field)
        config_params = {k: v for k, v in config.items() if k != 'name'}
        params = StrategyParams(**{**base_params, **config_params})
        
        # Run simulation
        apy, max_drawdown, num_trades, _ = backtester.simulate_strategy_fast(params)
        
        result = {
            'name': config['name'],
            'apy': apy,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'volume_confirmation': config['volume_confirmation'],
            'volume_sma_period': config['volume_sma_period'],
            'risk_adjusted_return': apy / max_drawdown if max_drawdown > 0 else 0
        }
        
        results.append(result)
        
        print(f"  APY: {apy:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Trades: {num_trades}")
        print(f"  Risk-Adjusted Return: {result['risk_adjusted_return']:.3f}")
    
    return results

def optimize_with_best_volume_filter():
    """Run full optimization with the best volume filter enabled"""
    
    print("\n🚀 OPTIMIZING WITH VOLUME FILTER")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    
    # Create optimizer
    optimizer = UltraFastOptimizer(data, initial_balance=10000)
    
    # Define optimization ranges with volume filter enabled
    param_ranges = {
        'base_percent': [0.5, 1.0, 1.5, 2.0],
        'initial_deviation': [0.5, 1.0, 1.5, 2.0],
        'trailing_deviation': [2.0, 3.0, 4.0, 5.0, 6.0],
        'tp_level1': [3.0, 5.0, 8.0, 10.0, 12.0],
        'tp_percent1': [40, 50, 60],
        'tp_percent2': [25, 30, 35],
        'tp_percent3': [15, 20, 25],
        'rsi_entry_threshold': [45.0, 50.0, 55.0, 60.0],
        'rsi_safety_threshold': [50.0, 55.0, 60.0],
        
        # Enable volume filter with best period from analysis
        'volume_confirmation': [True],  # Force enable
        'volume_sma_period': [10, 20, 30],  # Test the top 3 periods
        
        # Keep other filters disabled for now
        'sma_trend_filter': [False],
        'ema_trend_filter': [False],
        'atr_volatility_filter': [False],
        'higher_highs_filter': [False]
    }
    
    print("Running optimization with volume filters...")
    best_params, best_apy, best_drawdown = optimizer.optimize(
        param_ranges=param_ranges,
        n_trials=100,
        apy_weight=0.8,
        drawdown_weight=0.2
    )
    
    print(f"\n✅ OPTIMIZATION COMPLETE")
    print(f"Best APY: {best_apy:.2f}%")
    print(f"Best Drawdown: {best_drawdown:.2f}%")
    print(f"Risk-Adjusted Return: {best_apy/best_drawdown:.3f}")
    
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params, best_apy, best_drawdown

def test_trend_filters_addition():
    """Test adding trend filters to the volume-optimized strategy"""
    
    print("\n🔍 TESTING TREND FILTERS ADDITION")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    backtester = FastBacktester(data, 10000)
    
    # Use volume-optimized base parameters (will be updated after optimization)
    base_params = {
        'base_percent': 1.0,
        'initial_deviation': 1.0,
        'trailing_deviation': 4.5,
        'tp_level1': 8.0,
        'tp_percent1': 50,
        'tp_percent2': 30,
        'tp_percent3': 20,
        'rsi_entry_threshold': 55.0,
        'rsi_safety_threshold': 55.0,
        'volume_confirmation': True,
        'volume_sma_period': 20,  # Will update based on optimization
        'atr_volatility_filter': False,
        'higher_highs_filter': False
    }
    
    # Test trend filter combinations
    trend_configs = [
        {
            'name': 'Volume Only (Baseline)',
            'sma_trend_filter': False,
            'ema_trend_filter': False
        },
        {
            'name': 'Volume + SMA(200)',
            'sma_trend_filter': True,
            'sma_trend_period': 200,
            'ema_trend_filter': False
        },
        {
            'name': 'Volume + SMA(100)',
            'sma_trend_filter': True,
            'sma_trend_period': 100,
            'ema_trend_filter': False
        },
        {
            'name': 'Volume + EMA(50)',
            'sma_trend_filter': False,
            'ema_trend_filter': True,
            'ema_trend_period': 50
        },
        {
            'name': 'Volume + SMA(200) + EMA(50)',
            'sma_trend_filter': True,
            'sma_trend_period': 200,
            'ema_trend_filter': True,
            'ema_trend_period': 50
        }
    ]
    
    trend_results = []
    
    for config in trend_configs:
        print(f"\nTesting: {config['name']}")
        
        # Create parameters (exclude 'name' field)
        config_params = {k: v for k, v in config.items() if k != 'name'}
        params = StrategyParams(**{**base_params, **config_params})
        
        # Run simulation
        apy, max_drawdown, num_trades, _ = backtester.simulate_strategy_fast(params)
        
        result = {
            'name': config['name'],
            'apy': apy,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'risk_adjusted_return': apy / max_drawdown if max_drawdown > 0 else 0
        }
        
        trend_results.append(result)
        
        print(f"  APY: {apy:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Trades: {num_trades}")
        print(f"  Risk-Adjusted Return: {result['risk_adjusted_return']:.3f}")
    
    return trend_results

def main():
    """Main optimization workflow"""
    
    print("🎯 ITERATIVE OPTIMIZATION BASED ON DRAWDOWN ANALYSIS")
    print("=" * 70)
    
    # Step 1: Test volume filter combinations
    volume_results = test_volume_filter_combinations()
    
    # Find best volume configuration
    best_volume = max(volume_results, key=lambda x: x['apy'])
    print(f"\n🏆 BEST VOLUME CONFIGURATION:")
    print(f"  {best_volume['name']}")
    print(f"  APY: {best_volume['apy']:.2f}%")
    print(f"  Max Drawdown: {best_volume['max_drawdown']:.2f}%")
    print(f"  Risk-Adjusted Return: {best_volume['risk_adjusted_return']:.3f}")
    
    # Step 2: Full optimization with best volume filter
    if best_volume['volume_confirmation']:
        print(f"\nVolume filter improves performance! Running full optimization...")
        best_params, best_apy, best_drawdown = optimize_with_best_volume_filter()
    else:
        print(f"\nVolume filter doesn't improve performance. Keeping baseline.")
        best_params = None
        best_apy = best_volume['apy']
        best_drawdown = best_volume['max_drawdown']
    
    # Step 3: Test adding trend filters
    trend_results = test_trend_filters_addition()
    
    # Find best overall configuration
    best_trend = max(trend_results, key=lambda x: x['apy'])
    print(f"\n🏆 BEST OVERALL CONFIGURATION:")
    print(f"  {best_trend['name']}")
    print(f"  APY: {best_trend['apy']:.2f}%")
    print(f"  Max Drawdown: {best_trend['max_drawdown']:.2f}%")
    print(f"  Risk-Adjusted Return: {best_trend['risk_adjusted_return']:.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_results = {
        'timestamp': timestamp,
        'volume_filter_tests': volume_results,
        'trend_filter_tests': trend_results,
        'best_volume_config': best_volume,
        'best_trend_config': best_trend,
        'optimized_parameters': best_params,
        'final_performance': {
            'apy': best_trend['apy'],
            'max_drawdown': best_trend['max_drawdown'],
            'risk_adjusted_return': best_trend['risk_adjusted_return']
        }
    }
    
    results_path = f"results/volume_optimization_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {results_path}")
    
    # Performance comparison
    baseline_apy = volume_results[0]['apy']  # No filters
    final_apy = best_trend['apy']
    improvement = final_apy - baseline_apy
    
    print(f"\n📈 PERFORMANCE IMPROVEMENT:")
    print(f"  Baseline APY (No Filters): {baseline_apy:.2f}%")
    print(f"  Final APY (With Filters): {final_apy:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    print(f"  Relative Improvement: {(improvement/baseline_apy)*100:+.1f}%")
    
    return final_results

if __name__ == "__main__":
    main()
