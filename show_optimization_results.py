#!/usr/bin/env python3
"""
Display optimization results and run final backtest
"""

import json
import glob
import os
from datetime import datetime
from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams

def show_latest_results():
    """Show the latest optimization results"""
    
    # Find the latest results file
    result_files = glob.glob('results/SOL_two_phase_*.json')
    if not result_files:
        print("No optimization results found!")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    
    # Load results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*80)
    print("📊 TWO-PHASE OPTIMIZATION RESULTS")
    print("="*80)
    print(f"File: {latest_file}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Coin: {results['coin']}")
    print(f"Phase 1 Trials: {results['phase1_trials']}")
    print(f"Phase 2 Trials: {results['phase2_trials']}")
    
    best = results['best_configuration']
    params = best['parameters']
    
    print("\n🏆 BEST CONFIGURATION:")
    print("-"*80)
    print(f"APY: {best['apy']:.2f}%")
    print(f"Max Drawdown: {best['max_drawdown']:.2f}%")
    print(f"Total Trades: {best['num_trades']}")
    print(f"Total Position Size: {best['total_position_size']:.1f}% ✅")
    print(f"Est. Avg DD Duration: {best['avg_drawdown_duration_estimate']:.1f} days")
    print(f"Risk-Adjusted Return: {best['apy']/best['max_drawdown']:.3f}")
    
    print("\n📋 PARAMETERS:")
    print("-"*80)
    print(f"Base Order: {params['base_percent']}%")
    print(f"Volume Multiplier: {params['volume_multiplier']}")
    print(f"Max Safety Orders: {params['max_safeties']}")
    print(f"Initial Deviation: {params['initial_deviation']}%")
    print(f"Step Multiplier: {params['step_multiplier']}")
    print(f"Take Profit: {params['tp_level1']}%")
    print(f"Trailing Stop: {params['trailing_deviation']}%")
    print(f"RSI Entry: < {params['rsi_entry_threshold']}")
    print(f"RSI Safety: < {params['rsi_safety_threshold']}")
    
    print("\n📈 TOP 10 CONFIGURATIONS FROM PHASE 1:")
    print("-"*80)
    print(f"{'Rank':<6} {'APY':<10} {'Max DD':<10} {'Trades':<10} {'Position':<10}")
    print("-"*80)
    for config in results['top_10_configurations']:
        print(f"{config['rank']:<6} {config['apy']:<10.2f} {config['max_dd']:<10.2f} "
              f"{config['num_trades']:<10} {config['total_position']:<10.1f}")
    
    # Run final backtest with the best parameters
    print("\n" + "="*80)
    print("🚀 RUNNING FINAL BACKTEST WITH BEST PARAMETERS")
    print("="*80)
    
    # Create StrategyParams object
    best_params = StrategyParams(
        base_percent=params['base_percent'],
        volume_multiplier=params['volume_multiplier'],
        max_safeties=params['max_safeties'],
        initial_deviation=params['initial_deviation'],
        step_multiplier=params['step_multiplier'],
        tp_level1=params['tp_level1'],
        tp_percent1=params['tp_percent1'],
        trailing_deviation=params['trailing_deviation'],
        rsi_entry_threshold=params['rsi_entry_threshold'],
        rsi_safety_threshold=params['rsi_safety_threshold'],
        rsi_exit_threshold=params['rsi_exit_threshold']
    )
    
    # Load data and run backtest
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    backtester = FastBacktester(data, 10000)
    
    apy, max_dd, num_trades, trades_df = backtester.simulate_strategy_fast(best_params)
    
    print(f"\nFinal Validation:")
    print(f"  APY: {apy:.2f}%")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Total Trades: {num_trades}")
    
    # Save final backtest results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results = {
        'coin': 'SOL',
        'timestamp': timestamp,
        'source': 'two_phase_optimization',
        'apy': apy,
        'max_drawdown': max_dd,
        'num_trades': num_trades,
        'parameters': params
    }
    
    final_path = f"results/SOL_final_optimized_{timestamp}.json"
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Final results saved to: {final_path}")
    
    # Create a summary for 3commas
    print("\n" + "="*80)
    print("📝 3COMMAS BOT CONFIGURATION")
    print("="*80)
    print("Copy these settings to your 3commas DCA bot:")
    print(f"  Base Order Size: {params['base_percent']}% of account")
    print(f"  Safety Order Size: {params['base_percent']}% of account")
    print(f"  Safety Order Volume Scale: {params['volume_multiplier']}")
    print(f"  Safety Order Step Scale: {params['step_multiplier']}")
    print(f"  Max Safety Orders: {params['max_safeties']}")
    print(f"  Price Deviation to Open Safety: {params['initial_deviation']}%")
    print(f"  Take Profit: {params['tp_level1']}%")
    print(f"  Trailing Take Profit: Enabled")
    print(f"  Trailing Deviation: {params['trailing_deviation']}%")
    print("\nStart Conditions:")
    print(f"  - RSI-14 (1H) < {params['rsi_entry_threshold']}")
    print("\nDeal Start Condition:")
    print("  - Open new deal ASAP after previous deal closed")

if __name__ == "__main__":
    show_latest_results()
