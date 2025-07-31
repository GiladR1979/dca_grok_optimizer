#!/usr/bin/env python3
"""
Test the Ultra Aggressive configuration that achieved 21.16% APY
"""

from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams
import json
from datetime import datetime

def test_ultra_aggressive():
    """Test the ultra aggressive configuration"""
    
    print("🚀 TESTING ULTRA AGGRESSIVE CONFIGURATION")
    print("=" * 60)
    
    # Load data
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    backtester = FastBacktester(data, 10000)
    
    # Ultra Aggressive parameters that achieved 21.16% APY
    params = StrategyParams(
        base_percent=3.0,
        initial_deviation=0.3,
        trailing_deviation=1.5,
        tp_level1=2.0,
        tp_percent1=70,
        tp_percent2=20,
        tp_percent3=10,
        rsi_entry_threshold=70.0,
        rsi_safety_threshold=75.0,
        
        # Keep filters disabled
        sma_trend_filter=False,
        ema_trend_filter=False,
        atr_volatility_filter=False,
        higher_highs_filter=False,
        volume_confirmation=False
    )
    
    print("Parameters:")
    print(f"  Base Percent: {params.base_percent}%")
    print(f"  Initial Deviation: {params.initial_deviation}%")
    print(f"  Trailing Deviation: {params.trailing_deviation}%")
    print(f"  Take Profit Level 1: {params.tp_level1}%")
    print(f"  TP Percentages: {params.tp_percent1}% / {params.tp_percent2}% / {params.tp_percent3}%")
    print(f"  RSI Entry Threshold: {params.rsi_entry_threshold}")
    print(f"  RSI Safety Threshold: {params.rsi_safety_threshold}")
    
    print("\nRunning simulation...")
    
    # Run simulation
    apy, max_drawdown, num_trades, trades_df = backtester.simulate_strategy_fast(params)
    
    print(f"\n✅ SIMULATION COMPLETE")
    print(f"APY: {apy:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {num_trades}")
    print(f"Risk-Adjusted Return: {apy/max_drawdown:.3f}")
    
    # Calculate additional metrics
    win_rate = None
    avg_profit = None
    profitable_trades = None
    
    if trades_df is not None and len(trades_df) > 0:
        try:
            profitable_trades = len(trades_df[trades_df['profit_pct'] > 0])
            win_rate = (profitable_trades / len(trades_df)) * 100
            avg_profit = trades_df['profit_pct'].mean()
            
            print(f"\nTrade Statistics:")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Average Profit per Trade: {avg_profit:.2f}%")
            print(f"  Profitable Trades: {profitable_trades}/{len(trades_df)}")
        except (TypeError, KeyError):
            print(f"\nTrade Statistics:")
            print(f"  Total Trades: {num_trades}")
            print(f"  (Detailed trade analysis not available)")
    else:
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {num_trades}")
        print(f"  (No trade details available)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'configuration': 'Ultra Aggressive',
        'apy': apy,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'risk_adjusted_return': apy / max_drawdown if max_drawdown > 0 else 0,
        'parameters': {
            'base_percent': params.base_percent,
            'initial_deviation': params.initial_deviation,
            'trailing_deviation': params.trailing_deviation,
            'tp_level1': params.tp_level1,
            'tp_percent1': params.tp_percent1,
            'tp_percent2': params.tp_percent2,
            'tp_percent3': params.tp_percent3,
            'rsi_entry_threshold': params.rsi_entry_threshold,
            'rsi_safety_threshold': params.rsi_safety_threshold
        }
    }
    
    if win_rate is not None:
        results['trade_statistics'] = {
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit,
            'profitable_trades': profitable_trades,
            'total_trades': len(trades_df)
        }
    
    results_path = f"results/ultra_aggressive_test_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    test_ultra_aggressive()
