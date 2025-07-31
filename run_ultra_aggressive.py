#!/usr/bin/env python3
"""
Simple script to run the Ultra Aggressive DCA strategy
Usage: python run_ultra_aggressive.py [options]
"""

import argparse
import sys
from pathlib import Path
from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams
import json
from datetime import datetime

def get_ultra_aggressive_params():
    """Get the proven ultra aggressive parameters"""
    return StrategyParams(
        base_percent=3.0,
        initial_deviation=0.3,
        trailing_deviation=1.5,
        tp_level1=2.0,
        tp_percent1=70,
        tp_percent2=20,
        tp_percent3=10,
        rsi_entry_threshold=70.0,
        rsi_safety_threshold=75.0,
        
        # Disable all filters for maximum trading frequency
        sma_trend_filter=False,
        ema_trend_filter=False,
        atr_volatility_filter=False,
        higher_highs_filter=False,
        volume_confirmation=False
    )

def run_ultra_aggressive(data_path: str, coin: str, initial_balance: float = 10000):
    """Run the ultra aggressive strategy"""
    
    print("🚀 RUNNING ULTRA AGGRESSIVE DCA STRATEGY")
    print("=" * 60)
    print(f"Coin: {coin}")
    print(f"Data: {data_path}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print()
    
    # Load data
    try:
        data = FastDataProcessor.load_data(data_path)
        print(f"✅ Loaded {len(data):,} data points")
        print(f"   Period: {data.index[0]} to {data.index[-1]}")
        print(f"   Duration: {(data.index[-1] - data.index[0]).days} days")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Initialize backtester
    backtester = FastBacktester(data, initial_balance)
    
    # Get ultra aggressive parameters
    params = get_ultra_aggressive_params()
    
    print("\n📊 ULTRA AGGRESSIVE PARAMETERS:")
    print(f"   Base Percent: {params.base_percent}%")
    print(f"   Initial Deviation: {params.initial_deviation}%")
    print(f"   Trailing Deviation: {params.trailing_deviation}%")
    print(f"   Take Profit Level 1: {params.tp_level1}%")
    print(f"   TP Distribution: {params.tp_percent1}%/{params.tp_percent2}%/{params.tp_percent3}%")
    print(f"   RSI Entry: {params.rsi_entry_threshold}")
    print(f"   RSI Safety: {params.rsi_safety_threshold}")
    print(f"   All Filters: Disabled")
    
    print("\n🔄 Running simulation...")
    
    # Run simulation
    try:
        apy, max_drawdown, num_trades, balance_history = backtester.simulate_strategy_fast(params)
        
        # Calculate additional metrics
        final_balance = initial_balance * (1 + apy/100)
        risk_adjusted_return = apy / max_drawdown if max_drawdown > 0 else 0
        
        print("\n✅ SIMULATION COMPLETE!")
        print("=" * 60)
        print(f"📈 APY: {apy:.2f}%")
        print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        print(f"🔄 Total Trades: {num_trades:,}")
        print(f"💰 Final Balance: ${final_balance:,.2f}")
        print(f"⚖️  Risk-Adjusted Return: {risk_adjusted_return:.3f}")
        print("=" * 60)
        
        # Performance assessment
        if apy > 20:
            print("🎯 EXCELLENT: APY > 20%")
        elif apy > 15:
            print("✅ GOOD: APY > 15%")
        elif apy > 10:
            print("👍 DECENT: APY > 10%")
        elif apy > 0:
            print("⚠️  LOW: APY > 0% but below expectations")
        else:
            print("❌ PROBLEM: APY = 0% - Check parameters!")
        
        if max_drawdown < 25:
            print("🛡️  LOW RISK: Drawdown < 25%")
        elif max_drawdown < 35:
            print("⚖️  MODERATE RISK: Drawdown < 35%")
        else:
            print("⚠️  HIGH RISK: Drawdown > 35%")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'timestamp': timestamp,
            'coin': coin,
            'strategy': 'Ultra Aggressive',
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'apy': apy,
            'max_drawdown': max_drawdown,
            'total_trades': num_trades,
            'risk_adjusted_return': risk_adjusted_return,
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
            },
            'data_info': {
                'file': data_path,
                'start_date': data.index[0].isoformat(),
                'end_date': data.index[-1].isoformat(),
                'total_days': (data.index[-1] - data.index[0]).days,
                'data_points': len(data)
            }
        }
        
        results_file = results_dir / f"{coin}_ultra_aggressive_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run Ultra Aggressive DCA Strategy')
    parser.add_argument('--coin', default='SOLUSDT', help='Coin symbol (default: SOLUSDT)')
    parser.add_argument('--data_path', default='data/SOLUSDT_1m.csv', help='Path to CSV data file')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--list_data', action='store_true', help='List available data files')
    
    args = parser.parse_args()
    
    # List available data files
    if args.list_data:
        data_dir = Path('data')
        if data_dir.exists():
            print("📁 Available data files:")
            for file in sorted(data_dir.glob('*.csv')):
                print(f"   {file}")
        else:
            print("❌ Data directory not found")
        return
    
    # Check if data file exists
    if not Path(args.data_path).exists():
        print(f"❌ Data file not found: {args.data_path}")
        print("💡 Use --list_data to see available files")
        return
    
    # Run the strategy
    results = run_ultra_aggressive(args.data_path, args.coin, args.balance)
    
    if results:
        print("\n🎯 QUICK TIPS:")
        print("• For higher APY: Increase base_percent or decrease initial_deviation")
        print("• For lower risk: Enable trend filters or increase initial_deviation")
        print("• For other coins: Use --data_path and --coin parameters")
        print("• For optimization: Run 'python aggressive_optimization.py'")
        print("\n📖 See ULTRA_AGGRESSIVE_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()
