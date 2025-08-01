#!/usr/bin/env python3
"""
Two-Phase DCA Optimization with Position Size Validation
Phase 1: Find high APY strategies
Phase 2: Optimize for short drawdown duration among top performers
"""

import optuna
import numpy as np
from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams, OptimizationConfig
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_max_position_size(base_percent, volume_multiplier, max_safeties):
    """Calculate maximum position size including all safety orders"""
    total = base_percent
    for i in range(max_safeties):
        total += base_percent * (volume_multiplier ** i)
    return total

def validate_position_size(params):
    """Check if total position size is within account limits"""
    total = calculate_max_position_size(
        params.base_percent,
        params.volume_multiplier,
        params.max_safeties
    )
    return total <= 100.0, total

class TwoPhaseOptimizer:
    def __init__(self, data_path, coin, initial_balance=10000):
        """Initialize optimizer with data"""
        print(f"Loading data from {data_path}...")
        self.data = FastDataProcessor.load_data(data_path)
        self.backtester = FastBacktester(self.data, initial_balance)
        self.coin = coin
        self.initial_balance = initial_balance
        self.phase1_results = []
        
    def objective_phase1(self, trial):
        """Phase 1: Maximize APY with position size constraints"""
        
        # Get suggested parameters
        config = OptimizationConfig()
        params = config.suggest_params(trial)
        
        # CRITICAL: Validate position size BEFORE running backtest
        is_valid, total_size = validate_position_size(params)
        
        if not is_valid:
            # Return terrible score for invalid configurations
            print(f"  ❌ Invalid config: Total position {total_size:.1f}% > 100%")
            return -1000.0  # Heavily penalize invalid configs
        
        # Run backtest
        apy, max_dd, num_trades, _ = self.backtester.simulate_strategy_fast(params)
        
        # Store results for phase 2
        self.phase1_results.append({
            'trial': trial.number,
            'params': params,
            'apy': apy,
            'max_dd': max_dd,
            'num_trades': num_trades,
            'total_position': total_size
        })
        
        # Objective: Maximize APY while keeping drawdown reasonable
        if max_dd > 90:
            return apy * 0.5  # Penalize extreme drawdowns
        
        return apy
    
    def objective_phase2(self, trial, top_params):
        """Phase 2: Minimize drawdown duration among top performers"""
        
        # Select one of the top configurations from phase 1
        idx = trial.suggest_int('config_idx', 0, len(top_params) - 1)
        params = top_params[idx]
        
        # Run detailed backtest to get drawdown duration
        apy, max_dd, num_trades, trades_df = self.backtester.simulate_strategy_fast(params)
        
        # Calculate average drawdown duration
        if trades_df is not None and len(trades_df) > 0:
            # Simple proxy: more frequent trades = shorter drawdown periods
            avg_dd_duration = 365 / max(num_trades, 1)  # Days between trades
        else:
            avg_dd_duration = 365  # Worst case
        
        # Objective: Minimize drawdown duration while maintaining high APY
        score = -avg_dd_duration + (apy / 10)  # Balance between short DD and high APY
        
        return score
    
    def run_optimization(self, n_trials_phase1=100, n_trials_phase2=50):
        """Run two-phase optimization"""
        
        print("\n" + "="*80)
        print("🚀 TWO-PHASE DCA OPTIMIZATION")
        print("="*80)
        
        # PHASE 1: Find high APY strategies
        print("\n📊 PHASE 1: Finding High APY Strategies (with valid position sizes)")
        print("-"*80)
        
        study1 = optuna.create_study(direction='maximize')
        study1.optimize(self.objective_phase1, n_trials=n_trials_phase1)
        
        # Get top 20% of configurations
        sorted_results = sorted(self.phase1_results, key=lambda x: x['apy'], reverse=True)
        top_count = max(10, int(len(sorted_results) * 0.2))
        top_results = sorted_results[:top_count]
        
        print(f"\n✅ Phase 1 Complete: Found {len(top_results)} top configurations")
        print(f"Best APY: {top_results[0]['apy']:.2f}% (Total Position: {top_results[0]['total_position']:.1f}%)")
        
        # PHASE 2: Optimize for short drawdown duration
        print("\n📊 PHASE 2: Optimizing for Short Drawdown Duration")
        print("-"*80)
        
        top_params = [r['params'] for r in top_results]
        
        study2 = optuna.create_study(direction='maximize')
        study2.optimize(lambda trial: self.objective_phase2(trial, top_params), n_trials=n_trials_phase2)
        
        # Get best configuration
        best_idx = study2.best_params['config_idx']
        best_result = top_results[best_idx]
        best_params = best_result['params']
        
        # Run final detailed backtest
        print("\n📊 FINAL VALIDATION")
        print("-"*80)
        
        apy, max_dd, num_trades, trades_df = self.backtester.simulate_strategy_fast(best_params)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'coin': self.coin,
            'timestamp': timestamp,
            'optimization_type': 'two_phase',
            'phase1_trials': n_trials_phase1,
            'phase2_trials': n_trials_phase2,
            'best_configuration': {
                'apy': apy,
                'max_drawdown': max_dd,
                'num_trades': num_trades,
                'total_position_size': best_result['total_position'],
                'avg_drawdown_duration_estimate': 365 / max(num_trades, 1),
                'parameters': {
                    'base_percent': best_params.base_percent,
                    'volume_multiplier': best_params.volume_multiplier,
                    'max_safeties': best_params.max_safeties,
                    'initial_deviation': best_params.initial_deviation,
                    'step_multiplier': best_params.step_multiplier,
                    'tp_level1': best_params.tp_level1,
                    'tp_percent1': best_params.tp_percent1,
                    'trailing_deviation': best_params.trailing_deviation,
                    'rsi_entry_threshold': best_params.rsi_entry_threshold,
                    'rsi_safety_threshold': best_params.rsi_safety_threshold,
                    'rsi_exit_threshold': best_params.rsi_exit_threshold
                }
            },
            'top_10_configurations': [
                {
                    'rank': i+1,
                    'apy': r['apy'],
                    'max_dd': r['max_dd'],
                    'total_position': r['total_position'],
                    'num_trades': r['num_trades']
                }
                for i, r in enumerate(top_results[:10])
            ]
        }
        
        # Save to file
        results_path = f"results/{self.coin}_two_phase_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\n📊 Best Configuration Found:")
        print(f"  APY: {apy:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Total Trades: {num_trades}")
        print(f"  Total Position Size: {best_result['total_position']:.1f}% ✅")
        print(f"  Est. Avg DD Duration: {365 / max(num_trades, 1):.1f} days")
        print(f"\n💾 Results saved to: {results_path}")
        
        # Print parameters
        print(f"\n🔧 Optimal Parameters:")
        print(f"  Base Order: {best_params.base_percent}%")
        print(f"  Volume Multiplier: {best_params.volume_multiplier}")
        print(f"  Max Safety Orders: {best_params.max_safeties}")
        print(f"  Initial Deviation: {best_params.initial_deviation}%")
        print(f"  Step Multiplier: {best_params.step_multiplier}")
        print(f"  Take Profit: {best_params.tp_level1}%")
        print(f"  Trailing Stop: {best_params.trailing_deviation}%")
        print(f"  RSI Entry: < {best_params.rsi_entry_threshold}")
        print(f"  RSI Safety: < {best_params.rsi_safety_threshold}")
        
        return best_params, results

def main():
    """Run two-phase optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-Phase DCA Optimization')
    parser.add_argument('--data_path', default='data/SOLUSDT_1m.csv', help='Path to data file')
    parser.add_argument('--coin', default='SOL', help='Coin symbol')
    parser.add_argument('--phase1_trials', type=int, default=100, help='Number of trials for phase 1')
    parser.add_argument('--phase2_trials', type=int, default=50, help='Number of trials for phase 2')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = TwoPhaseOptimizer(args.data_path, args.coin)
    best_params, results = optimizer.run_optimization(
        n_trials_phase1=args.phase1_trials,
        n_trials_phase2=args.phase2_trials
    )
    
    # Run final backtest with visualization
    print("\n📈 Running final backtest with visualization...")
    import subprocess
    subprocess.run([
        'python3', 'fast_dca_backtest.py',
        '--data_path', args.data_path,
        '--coin', args.coin,
        '--custom_params',
        f"base_percent={best_params.base_percent}",
        f"initial_deviation={best_params.initial_deviation}",
        f"tp_level1={best_params.tp_level1}",
        f"rsi_entry_threshold={best_params.rsi_entry_threshold}",
        f"rsi_safety_threshold={best_params.rsi_safety_threshold}"
    ])

if __name__ == "__main__":
    main()
