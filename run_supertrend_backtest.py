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
        # Default Supertrend-optimized parameters (updated for long/short)
        strategy_params = StrategyParams(
            # Position sizing (keep as is, symmetric defaults)
            base_percent_long=1.33,
            base_percent_short=1.33,
            volume_multiplier_long=1.5,
            volume_multiplier_short=1.5,
            max_safeties_long=8,
            max_safeties_short=8,

            # Supertrend settings (shared)
            use_supertrend_filter=True,
            supertrend_timeframe='1h',
            supertrend_period=10,
            supertrend_multiplier=3.0,
            require_bullish_supertrend=True,

            # TP settings (symmetric defaults)
            tp_level1_long=3.0,
            tp_level1_short=3.0,
            tp_percent1_long=100.0,  # Sell entire position
            tp_percent1_short=100.0,
            trailing_deviation_long=2.0,
            trailing_deviation_short=2.0,

            # Entry/safety conditions (less important with Supertrend, symmetric)
            initial_deviation_long=2.5,
            initial_deviation_short=2.5,
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
        print(f"TP Level: {strategy_params.tp_level1_long}%")  # Showing long as example; could expand
        print(f"Trailing: {strategy_params.trailing_deviation_long}%")

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

        # Save JSON results (updated for long/short params)
        results = {
            'coin': args.coin,
            'initial_balance': args.initial_balance,
            'final_balance': args.initial_balance * (1 + apy/100),
            'apy': apy,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'parameters': {
                'base_percent_long': strategy_params.base_percent_long,
                'base_percent_short': strategy_params.base_percent_short,
                'volume_multiplier_long': strategy_params.volume_multiplier_long,
                'volume_multiplier_short': strategy_params.volume_multiplier_short,
                'step_multiplier_long': strategy_params.step_multiplier_long,
                'step_multiplier_short': strategy_params.step_multiplier_short,
                'max_safeties_long': strategy_params.max_safeties_long,
                'max_safeties_short': strategy_params.max_safeties_short,
                'use_supertrend_filter': strategy_params.use_supertrend_filter,
                'supertrend_timeframe': strategy_params.supertrend_timeframe,
                'supertrend_period': strategy_params.supertrend_period,
                'supertrend_multiplier': strategy_params.supertrend_multiplier,
                'require_bullish_supertrend': strategy_params.require_bullish_supertrend,
                'tp_level1_long': strategy_params.tp_level1_long,
                'tp_level1_short': strategy_params.tp_level1_short,
                'tp_percent1_long': strategy_params.tp_percent1_long,
                'tp_percent1_short': strategy_params.tp_percent1_short,
                'trailing_deviation_long': strategy_params.trailing_deviation_long,
                'trailing_deviation_short': strategy_params.trailing_deviation_short,
                'initial_deviation_long': strategy_params.initial_deviation_long,
                'initial_deviation_short': strategy_params.initial_deviation_short,
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

