
#!/usr/bin/env python3
"""Wrapper to run fast_dca_backtest with ultra-aggressive parameters"""

import sys
sys.path.insert(0, '.')

from fast_dca_backtest import *
from ultra_aggressive_config import get_ultra_aggressive_params

# Override the main function to use our parameters
original_main = main

def custom_main():
    """Modified main that uses ultra-aggressive parameters"""
    parser = argparse.ArgumentParser(description='Fast DCA Strategy Backtester')
    parser.add_argument('--data_path', required=True, help='Path to CSV data file')
    parser.add_argument('--coin', required=True, help='Coin symbol')
    parser.add_argument('--initial_balance', type=float, default=10000)
    parser.add_argument('--output_dir', default='./results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        data = FastDataProcessor.load_data(args.data_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize fast backtester
    backtester = FastBacktester(data, args.initial_balance)
    
    # Use ultra-aggressive parameters
    strategy_params = get_ultra_aggressive_params()
    print("Using ultra-aggressive parameters (matching test_ultra_aggressive.py)")
    
    # Run simulation with trades
    print("Running full simulation with trades for visualization...")
    try:
        apy, max_drawdown, balance_history_for_save, trades = Visualizer.simulate_with_trades(backtester, strategy_params)
        print(f"Full simulation completed: APY={apy:.2f}%, Max DD={max_drawdown:.2f}%")
    except Exception as e:
        print(f"❌ Error in final simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate additional metrics
    num_trades = len(trades)
    avg_drawdown_duration = 0
    
    final_balance = args.initial_balance * (1 + apy/100)
    
    # Results
    results = {
        'coin': args.coin,
        'initial_balance': args.initial_balance,
        'final_balance': final_balance,
        'apy': round(apy, 2),
        'max_drawdown': round(max_drawdown, 2),
        'avg_drawdown_duration_hours': round(avg_drawdown_duration/60, 1),
        'total_trades': num_trades,
        'parameters': {
            'base_percent': strategy_params.base_percent,
            'tp_level1': strategy_params.tp_level1,
            'initial_deviation': strategy_params.initial_deviation,
            'trailing_deviation': strategy_params.trailing_deviation,
            'tp_percent1': strategy_params.tp_percent1,
            'rsi_entry_threshold': strategy_params.rsi_entry_threshold,
            'rsi_safety_threshold': strategy_params.rsi_safety_threshold,
            'fees': strategy_params.fees,
        },
        'data_period': {
            'start': data.index[0].isoformat(),
            'end': data.index[-1].isoformat(),
            'total_days': (data.index[-1] - data.index[0]).days
        }
    }
    
    print(f"\n⚡ FAST BACKTEST RESULTS FOR {args.coin}")
    print("=" * 60)
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"APY: {apy:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {num_trades}")
    print("=" * 60)
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{args.coin}_ultra_aggressive_test_{timestamp}"
    
    # Save results JSON
    results_path = output_dir / f"{base_filename}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

# Replace main with our custom version
main = custom_main

if __name__ == "__main__":
    main()
