#!/usr/bin/env python3
"""
Aggressive Optimization - Push APY Higher
Test wider parameter ranges and more aggressive settings
"""

import numpy as np
from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams
from ultra_fast_optimizer import UltraFastOptimizer, UltraFastGridSearch
import json
from datetime import datetime

def run_aggressive_optimization():
    """Run optimization with wider parameter ranges to push APY higher"""
    
    print("🚀 AGGRESSIVE OPTIMIZATION FOR MAXIMUM APY")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    
    # Create optimizer and grid search
    optimizer = UltraFastOptimizer(data, initial_balance=10000)
    grid_search = UltraFastGridSearch(optimizer)
    
    print("Running aggressive optimization...")
    
    # Run optimization with more combinations
    results = grid_search.run_parallel_optimization(
        n_combinations=3000,  # More combinations for better exploration
        sample_ratio=0.5,     # Use more data for accuracy
        n_processes=None      # Use all available cores
    )
    
    best_apy = results['apy']
    best_drawdown = results['max_drawdown']
    best_params = results['parameters']
    
    print(f"\n✅ AGGRESSIVE OPTIMIZATION COMPLETE")
    print(f"Best APY: {best_apy:.2f}%")
    print(f"Best Drawdown: {best_drawdown:.2f}%")
    print(f"Risk-Adjusted Return: {best_apy/best_drawdown:.3f}")
    
    print(f"\nBest Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params, best_apy, best_drawdown

def test_extreme_configurations():
    """Test some extreme configurations manually"""
    
    print("\n🔥 TESTING EXTREME CONFIGURATIONS")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    backtester = FastBacktester(data, 10000)
    
    # Test extreme configurations
    extreme_configs = [
        {
            'name': 'Ultra Aggressive',
            'base_percent': 3.0,
            'initial_deviation': 0.3,
            'trailing_deviation': 1.5,
            'tp_level1': 2.0,
            'tp_percent1': 70,
            'tp_percent2': 20,
            'tp_percent3': 10,
            'rsi_entry_threshold': 70.0,
            'rsi_safety_threshold': 75.0
        },
        {
            'name': 'High Frequency',
            'base_percent': 2.0,
            'initial_deviation': 0.5,
            'trailing_deviation': 2.0,
            'tp_level1': 3.0,
            'tp_percent1': 60,
            'tp_percent2': 25,
            'tp_percent3': 15,
            'rsi_entry_threshold': 65.0,
            'rsi_safety_threshold': 70.0
        },
        {
            'name': 'Quick Profits',
            'base_percent': 1.5,
            'initial_deviation': 0.8,
            'trailing_deviation': 3.0,
            'tp_level1': 1.5,
            'tp_percent1': 80,
            'tp_percent2': 15,
            'tp_percent3': 5,
            'rsi_entry_threshold': 60.0,
            'rsi_safety_threshold': 65.0
        },
        {
            'name': 'Deep Value',
            'base_percent': 2.5,
            'initial_deviation': 1.0,
            'trailing_deviation': 5.0,
            'tp_level1': 8.0,
            'tp_percent1': 40,
            'tp_percent2': 35,
            'tp_percent3': 25,
            'rsi_entry_threshold': 45.0,
            'rsi_safety_threshold': 50.0
        }
    ]
    
    results = []
    
    for config in extreme_configs:
        print(f"\nTesting: {config['name']}")
        
        # Create parameters (exclude 'name' field)
        config_params = {k: v for k, v in config.items() if k != 'name'}
        
        # Add default values for missing parameters
        default_params = {
            'sma_trend_filter': False,
            'ema_trend_filter': False,
            'atr_volatility_filter': False,
            'higher_highs_filter': False,
            'volume_confirmation': False
        }
        
        params = StrategyParams(**{**default_params, **config_params})
        
        # Run simulation
        apy, max_drawdown, num_trades, _ = backtester.simulate_strategy_fast(params)
        
        result = {
            'name': config['name'],
            'apy': apy,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'risk_adjusted_return': apy / max_drawdown if max_drawdown > 0 else 0,
            'parameters': config_params
        }
        
        results.append(result)
        
        print(f"  APY: {apy:.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Total Trades: {num_trades}")
        print(f"  Risk-Adjusted Return: {result['risk_adjusted_return']:.3f}")
    
    return results

def run_final_optimization_round():
    """Final optimization round with best insights"""
    
    print("\n🎯 FINAL OPTIMIZATION ROUND")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    
    # Create optimizer and grid search
    optimizer = UltraFastOptimizer(data, initial_balance=10000)
    grid_search = UltraFastGridSearch(optimizer)
    
    print("Running final optimization...")
    
    # Run focused optimization
    results = grid_search.run_parallel_optimization(
        n_combinations=2000,  # Focused search
        sample_ratio=0.7,     # Use more data for final accuracy
        n_processes=None      # Use all available cores
    )
    
    best_apy = results['apy']
    best_drawdown = results['max_drawdown']
    best_params = results['parameters']
    
    print(f"\n✅ FINAL OPTIMIZATION COMPLETE")
    print(f"Best APY: {best_apy:.2f}%")
    print(f"Best Drawdown: {best_drawdown:.2f}%")
    print(f"Risk-Adjusted Return: {best_apy/best_drawdown:.3f}")
    
    return best_params, best_apy, best_drawdown

def main():
    """Main aggressive optimization workflow"""
    
    print("⚡ AGGRESSIVE OPTIMIZATION FOR MAXIMUM APY")
    print("=" * 70)
    
    # Step 1: Aggressive optimization
    aggressive_params, aggressive_apy, aggressive_drawdown = run_aggressive_optimization()
    
    # Step 2: Test extreme configurations
    extreme_results = test_extreme_configurations()
    
    # Find best extreme configuration
    best_extreme = max(extreme_results, key=lambda x: x['apy'])
    print(f"\n🏆 BEST EXTREME CONFIGURATION:")
    print(f"  {best_extreme['name']}")
    print(f"  APY: {best_extreme['apy']:.2f}%")
    print(f"  Max Drawdown: {best_extreme['max_drawdown']:.2f}%")
    print(f"  Risk-Adjusted Return: {best_extreme['risk_adjusted_return']:.3f}")
    
    # Step 3: Final optimization round
    final_params, final_apy, final_drawdown = run_final_optimization_round()
    
    # Compare all results
    all_results = [
        {
            'name': 'Aggressive Optimization',
            'apy': aggressive_apy,
            'max_drawdown': aggressive_drawdown,
            'parameters': aggressive_params
        },
        {
            'name': f'Best Extreme ({best_extreme["name"]})',
            'apy': best_extreme['apy'],
            'max_drawdown': best_extreme['max_drawdown'],
            'parameters': best_extreme['parameters']
        },
        {
            'name': 'Final Optimization',
            'apy': final_apy,
            'max_drawdown': final_drawdown,
            'parameters': final_params
        }
    ]
    
    # Find overall best
    overall_best = max(all_results, key=lambda x: x['apy'])
    
    print(f"\n🏆 OVERALL BEST RESULT:")
    print(f"  Method: {overall_best['name']}")
    print(f"  APY: {overall_best['apy']:.2f}%")
    print(f"  Max Drawdown: {overall_best['max_drawdown']:.2f}%")
    print(f"  Risk-Adjusted Return: {overall_best['apy']/overall_best['max_drawdown']:.3f}")
    
    print(f"\n  Best Parameters:")
    for key, value in overall_best['parameters'].items():
        print(f"    {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_results = {
        'timestamp': timestamp,
        'aggressive_optimization': {
            'apy': aggressive_apy,
            'max_drawdown': aggressive_drawdown,
            'parameters': aggressive_params
        },
        'extreme_configurations': extreme_results,
        'final_optimization': {
            'apy': final_apy,
            'max_drawdown': final_drawdown,
            'parameters': final_params
        },
        'overall_best': overall_best
    }
    
    results_path = f"results/aggressive_optimization_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {results_path}")
    
    return final_results

if __name__ == "__main__":
    main()
