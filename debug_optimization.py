#!/usr/bin/env python3
"""Debug script to investigate optimization vs final backtest discrepancy"""

import pandas as pd
import numpy as np
from strategy_config import StrategyParams, OptimizationConfig
from fast_dca_backtest import FastBacktester, FastOptimizer, Visualizer

# Load data
print("Loading data...")
data = pd.read_csv('data/SOLUSDT_1m.csv')
data['ts'] = pd.to_datetime(data['ts'])
data.set_index('ts', inplace=True)
print(f"Loaded {len(data)} data points")

# Initialize backtester
backtester = FastBacktester(data, 10000)

# Run optimization with just 5 trials
print("\nRunning optimization...")
optimizer = FastOptimizer(backtester)
best_params = optimizer.optimize_fast(5)

print(f"\nOptimization complete!")
print(f"Best APY: {optimizer.best_apy:.2f}%")
print(f"Best Drawdown: {optimizer.best_drawdown:.2f}%")
print(f"Best num trades: {optimizer.best_num_trades}")

# Print the best parameters
print("\nBest parameters:")
for attr in ['base_percent', 'initial_deviation', 'trailing_deviation', 
             'tp_level1', 'tp_percent1', 'tp_percent2', 'tp_percent3',
             'rsi_entry_threshold', 'rsi_safety_threshold']:
    print(f"  {attr}: {getattr(best_params, attr)}")

# Now run the same parameters through the fast simulation
print("\nRunning fast simulation with best params...")
apy, max_dd, num_trades, balance_history = backtester.simulate_strategy_fast(best_params)
print(f"Fast simulation APY: {apy:.2f}%")
print(f"Fast simulation Drawdown: {max_dd:.2f}%")
print(f"Fast simulation Trades: {num_trades}")

# Try the visualization simulation
print("\nRunning visualization simulation...")
try:
    viz_apy, viz_dd, viz_balance_history, viz_trades = Visualizer.simulate_with_trades(backtester, best_params)
    print(f"Visualization APY: {viz_apy:.2f}%")
    print(f"Visualization Drawdown: {viz_dd:.2f}%")
    print(f"Visualization Trades: {len(viz_trades)}")
except Exception as e:
    print(f"Visualization simulation failed: {e}")
    
# Check if it's an issue with the DCAStrategy import
print("\nChecking DCAStrategy availability...")
try:
    from dca_backtest import DCAStrategy
    print("DCAStrategy imported successfully")
    
    # Try running with DCAStrategy directly
    strategy = DCAStrategy(best_params, 10000)
    print(f"DCAStrategy initialized with balance: {strategy.balance}")
except ImportError as e:
    print(f"Could not import DCAStrategy: {e}")
