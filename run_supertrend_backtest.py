#!/usr/bin/env python3
"""
Simple script to backtest a coin with the new Supertrend-based DCA strategy
"""

import argparse
from pathlib import Path
import json  # Added to fix NameError for json.dump
from fast_dca_backtest import FastDataProcessor, FastBacktester, FastOptimizer, Visualizer
from strategy_config import StrategyParams, OptimizationConfig

def main():
    parser = argparse.ArgumentParser(description='Backtest coin with Supertrend DCA strategy')
    parser.add_argument('--coin', required=True, help='Coin symbol (e.g., BTCUSDT)')
    parser.add_argument('--data_path', required=True, help='Path to CSV data file')
    parser.add_argument('--optimize', action='store_true', help='Run optimization (recommended)')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--initial_balance', type=float, default=10000, help='Starting balance in USDT')
    parser.add_argument('--max-apy-only', action='store_true', help='Maximize APY only during optimization (ignore drawdown penalties)')

    args = parser.parse_args()

    print(f"🚀 SUPERTREND DCA BACKTEST FOR {args.coin}")
    print("=" * 60)

    # Load data
    print("📊 Loading data...")
    try:
        data = FastDataProcessor.load_data(args.data_path)
        print(f"✅ Loaded {len(data)} data points")
        print(f"📅 Period: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Initialize backtester
    print("\n🔧 Initializing backtester...")
    backtester = FastBacktester(data, args.initial_balance)
    print("✅ Backtester ready")

    if args.optimize:
        print(f"\n🎯 OPTIMIZING SUPERTREND PARAMETERS ({args.trials} trials)...")
        print("This will find the best Supertrend timeframe, TP levels, and other parameters")

        # Create optimizer with Supertrend-focused configuration
        optimizer = FastOptimizer(backtester, OptimizationConfig(), max_apy_only=args.max_apy_only)

        # Run optimization
        best_params = optimizer.optimize_fast(args.trials)

        print(f"\n✅ OPTIMIZATION COMPLETE!")
        print(f"Best APY: {optimizer.best_apy:.2f}%")
        print(f"Best Max Drawdown: {optimizer.best_drawdown:.2f}%")

        strategy_params = best_params
    else:
        print("\n📋 Using default Supertrend parameters...")
        # Default Supertrend-optimized parameters
        strategy_params = StrategyParams(
            # Position sizing (keep as is)
            base_percent=1.33,
            volume_multiplier=1.5,
            max_safeties=8,

            # Supertrend settings
            use_supertrend_filter=True,
            supertrend_timeframe='1h',
            supertrend_period=10,
            supertrend_multiplier=3.0,
            require_bullish_supertrend=True,

            # TP settings
            tp_level1=3.0,
            tp_percent1=100.0,  # Sell entire position
            trailing_deviation=2.0,

            # Entry/safety conditions (less important with Supertrend)
            initial_deviation=2.5,
            rsi_entry_threshold=50.0,
            rsi_safety_threshold=40.0,
        )

    # Run final simulation with trade tracking
    print(f"\n🔄 RUNNING FINAL SIMULATION...")
    try:
        apy, max_drawdown, balance_history, trades = Visualizer.simulate_with_trades(backtester, strategy_params)

        print(f"\n🏆 BACKTEST RESULTS FOR {args.coin}")
        print("=" * 60)
        print(f"💰 Initial Balance: ${args.initial_balance:,.2f}")
        print(f"💰 Final Balance: ${args.initial_balance * (1 + apy/100):,.2f}")
        print(f"📈 APY: {apy:.2f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🔄 Total Trades: {len(trades)}")
        print("=" * 60)

        # Show Supertrend settings used
        print(f"\n⚙️  SUPERTREND SETTINGS USED:")
        print(f"Timeframe: {strategy_params.supertrend_timeframe}")
        print(f"Period: {strategy_params.supertrend_period}")
        print(f"Multiplier: {strategy_params.supertrend_multiplier}")
        print(f"TP Level: {strategy_params.tp_level1}%")
        print(f"Trailing: {strategy_params.trailing_deviation}%")

        # Save results
        output_dir = Path('./results')
        output_dir.mkdir(exist_ok=True)

        # Save chart
        chart_path = output_dir / f"{args.coin}_supertrend_chart.png"
        Visualizer.plot_results(balance_history, trades, args.coin, str(chart_path))
        print(f"\n📊 Chart saved: {chart_path}")

        # Save trades log
        trades_path = output_dir / f"{args.coin}_supertrend_trades.csv"
        Visualizer.save_trades_log(trades, str(trades_path))
        print(f"📋 Trades log saved: {trades_path}")

        # Save JSON results (added)
        results = {
            'coin': args.coin,
            'initial_balance': args.initial_balance,
            'final_balance': args.initial_balance * (1 + apy/100),
            'apy': apy,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'parameters': {
                'base_percent': strategy_params.base_percent,
                'volume_multiplier': strategy_params.volume_multiplier,
                'step_multiplier': strategy_params.step_multiplier,
                'max_safeties': strategy_params.max_safeties,
                'use_supertrend_filter': strategy_params.use_supertrend_filter,
                'supertrend_timeframe': strategy_params.supertrend_timeframe,
                'supertrend_period': strategy_params.supertrend_period,
                'supertrend_multiplier': strategy_params.supertrend_multiplier,
                'require_bullish_supertrend': strategy_params.require_bullish_supertrend,
                'tp_level1': strategy_params.tp_level1,
                'tp_percent1': strategy_params.tp_percent1,
                'trailing_deviation': strategy_params.trailing_deviation,
                'initial_deviation': strategy_params.initial_deviation,
                'rsi_entry_threshold': strategy_params.rsi_entry_threshold,
                'rsi_safety_threshold': strategy_params.rsi_safety_threshold,
            },
            'data_period': {
                'start': str(data.index[0]),
                'end': str(data.index[-1]),
                'total_days': (data.index[-1] - data.index[0]).days
            }
        }
        json_path = output_dir / f"{args.coin}_supertrend_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"📄 Results JSON saved: {json_path}")

        print(f"\n✅ All results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error in simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
