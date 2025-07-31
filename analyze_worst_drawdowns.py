#!/usr/bin/env python3
"""
Analyze Worst Drawdowns and Test 3commas Filters
Find the longest/deepest drawdowns and test which 3commas indicators could prevent them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fast_dca_backtest import FastBacktester, FastDataProcessor
from strategy_config import StrategyParams
import json

def analyze_drawdowns(balance_history, timestamps, prices):
    """Analyze all drawdown periods and find the worst ones"""
    
    drawdowns = []
    current_drawdown = None
    peak_value = balance_history[0]
    peak_idx = 0
    
    for i, balance in enumerate(balance_history):
        if balance > peak_value:
            # New peak - end any ongoing drawdown
            if current_drawdown is not None:
                current_drawdown['end_idx'] = i - 1
                current_drawdown['end_date'] = timestamps[i-1]
                current_drawdown['duration_hours'] = (timestamps[i-1] - current_drawdown['start_date']).total_seconds() / 3600
                current_drawdown['recovery_price'] = prices[i]
                drawdowns.append(current_drawdown)
                current_drawdown = None
            
            peak_value = balance
            peak_idx = i
        else:
            # Potential drawdown
            drawdown_pct = (peak_value - balance) / peak_value * 100
            
            if drawdown_pct > 1.0:  # Only track drawdowns > 1%
                if current_drawdown is None:
                    # Start new drawdown
                    current_drawdown = {
                        'start_idx': peak_idx,
                        'start_date': timestamps[peak_idx],
                        'peak_value': peak_value,
                        'peak_price': prices[peak_idx],
                        'max_drawdown_pct': drawdown_pct,
                        'max_drawdown_idx': i,
                        'max_drawdown_date': timestamps[i],
                        'min_value': balance,
                        'min_price': prices[i]
                    }
                else:
                    # Update existing drawdown
                    if drawdown_pct > current_drawdown['max_drawdown_pct']:
                        current_drawdown['max_drawdown_pct'] = drawdown_pct
                        current_drawdown['max_drawdown_idx'] = i
                        current_drawdown['max_drawdown_date'] = timestamps[i]
                        current_drawdown['min_value'] = balance
                        current_drawdown['min_price'] = prices[i]
    
    # Handle any ongoing drawdown at the end
    if current_drawdown is not None:
        current_drawdown['end_idx'] = len(balance_history) - 1
        current_drawdown['end_date'] = timestamps[-1]
        current_drawdown['duration_hours'] = (timestamps[-1] - current_drawdown['start_date']).total_seconds() / 3600
        current_drawdown['recovery_price'] = prices[-1]
        drawdowns.append(current_drawdown)
    
    return drawdowns

def get_indicators_at_time(backtester, idx):
    """Get all indicator values at a specific time index"""
    indicators = {}
    
    # Basic indicators
    indicators['rsi_1h'] = backtester.indicators['rsi_1h'][idx]
    indicators['sma_fast'] = backtester.indicators['sma_fast_1h'][idx]
    indicators['sma_slow'] = backtester.indicators['sma_slow_1h'][idx]
    
    # Trend indicators
    indicators['sma_50'] = backtester.indicators['sma_50'][idx]
    indicators['sma_100'] = backtester.indicators['sma_100'][idx]
    indicators['sma_200'] = backtester.indicators['sma_200'][idx]
    indicators['ema_21'] = backtester.indicators['ema_21'][idx]
    indicators['ema_50'] = backtester.indicators['ema_50'][idx]
    indicators['ema_100'] = backtester.indicators['ema_100'][idx]
    
    # Volatility indicators
    indicators['atr_14'] = backtester.indicators['atr_14'][idx]
    indicators['atr_21'] = backtester.indicators['atr_21'][idx]
    indicators['atr_28'] = backtester.indicators['atr_28'][idx]
    
    # Volume indicators
    indicators['volume'] = backtester.indicators['volume'][idx]
    indicators['vol_sma_10'] = backtester.indicators['vol_sma_10'][idx]
    indicators['vol_sma_20'] = backtester.indicators['vol_sma_20'][idx]
    indicators['vol_sma_30'] = backtester.indicators['vol_sma_30'][idx]
    
    return indicators

def test_3commas_filters(price, indicators, lookback_prices=None):
    """Test which 3commas filters would have prevented entry"""
    filters_triggered = []
    
    # 1. SMA Trend Filters
    if price <= indicators['sma_50']:
        filters_triggered.append("SMA_50_Filter: Price below SMA(50)")
    if price <= indicators['sma_100']:
        filters_triggered.append("SMA_100_Filter: Price below SMA(100)")
    if price <= indicators['sma_200']:
        filters_triggered.append("SMA_200_Filter: Price below SMA(200)")
    
    # 2. EMA Trend Filters
    if price <= indicators['ema_21']:
        filters_triggered.append("EMA_21_Filter: Price below EMA(21)")
    if price <= indicators['ema_50']:
        filters_triggered.append("EMA_50_Filter: Price below EMA(50)")
    if price <= indicators['ema_100']:
        filters_triggered.append("EMA_100_Filter: Price below EMA(100)")
    
    # 3. Volume Filters
    if indicators['volume'] < indicators['vol_sma_10'] * 0.8:
        filters_triggered.append("Volume_10_Filter: Volume below 80% of SMA(10)")
    if indicators['volume'] < indicators['vol_sma_20'] * 0.8:
        filters_triggered.append("Volume_20_Filter: Volume below 80% of SMA(20)")
    if indicators['volume'] < indicators['vol_sma_30'] * 0.8:
        filters_triggered.append("Volume_30_Filter: Volume below 80% of SMA(30)")
    
    # 4. ATR Volatility Filters (if we have recent price data)
    if lookback_prices is not None and len(lookback_prices) > 1:
        recent_change = abs(price - lookback_prices[-2])
        
        if recent_change > indicators['atr_14'] * 1.5:
            filters_triggered.append("ATR_14_1.5x_Filter: Recent volatility too high")
        if recent_change > indicators['atr_14'] * 2.0:
            filters_triggered.append("ATR_14_2.0x_Filter: Recent volatility too high")
        if recent_change > indicators['atr_21'] * 1.5:
            filters_triggered.append("ATR_21_1.5x_Filter: Recent volatility too high")
        if recent_change > indicators['atr_28'] * 2.0:
            filters_triggered.append("ATR_28_2.0x_Filter: Recent volatility too high")
    
    # 5. Higher Highs Filter (if we have lookback data)
    if lookback_prices is not None and len(lookback_prices) >= 20:
        recent_high_10 = max(lookback_prices[-10:])
        recent_high_20 = max(lookback_prices[-20:])
        
        if price < recent_high_10 * 0.95:
            filters_triggered.append("Higher_Highs_10_Filter: Price below 95% of 10-period high")
        if price < recent_high_20 * 0.95:
            filters_triggered.append("Higher_Highs_20_Filter: Price below 95% of 20-period high")
    
    return filters_triggered

def run_simulation_with_best_params():
    """Run simulation with current best parameters and analyze drawdowns"""
    
    # Load data
    print("Loading SOL/USDT data...")
    data = FastDataProcessor.load_data('data/SOLUSDT_1m.csv')
    print(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
    
    # Initialize backtester
    backtester = FastBacktester(data, 10000)
    
    # Use the best parameters from recent optimization
    best_params = StrategyParams(
        base_percent=1.0,
        initial_deviation=1.0,
        trailing_deviation=4.5,
        tp_level1=8.0,
        tp_percent1=50,
        tp_percent2=30,
        tp_percent3=20,
        rsi_entry_threshold=55.0,
        rsi_safety_threshold=55.0,
        # All 3commas filters disabled (as per optimization)
        sma_trend_filter=False,
        ema_trend_filter=False,
        atr_volatility_filter=False,
        higher_highs_filter=False,
        volume_confirmation=False
    )
    
    print("\nRunning simulation with best parameters...")
    apy, max_drawdown, num_trades, balance_history = backtester.simulate_strategy_fast(best_params)
    
    print(f"Simulation Results:")
    print(f"  APY: {apy:.2f}%")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Total Trades: {num_trades}")
    
    # Extract balance and timestamp arrays
    timestamps = [item[0] for item in balance_history]
    balances = [item[1] for item in balance_history]
    
    # Convert timestamps to datetime objects if needed
    timestamps = [pd.to_datetime(ts) for ts in timestamps]
    
    return backtester, balances, timestamps, best_params

def analyze_worst_periods(backtester, balances, timestamps):
    """Analyze the worst drawdown periods"""
    
    print("\nAnalyzing drawdown periods...")
    
    # Analyze all drawdowns
    drawdowns = analyze_drawdowns(balances, timestamps, backtester.prices)
    
    if not drawdowns:
        print("No significant drawdowns found!")
        return []
    
    # Sort by different criteria
    by_duration = sorted(drawdowns, key=lambda x: x.get('duration_hours', 0), reverse=True)
    by_magnitude = sorted(drawdowns, key=lambda x: x['max_drawdown_pct'], reverse=True)
    
    print(f"\nFound {len(drawdowns)} significant drawdown periods")
    
    print("\n=== TOP 5 LONGEST DRAWDOWNS ===")
    for i, dd in enumerate(by_duration[:5]):
        duration_days = dd.get('duration_hours', 0) / 24
        print(f"{i+1}. Duration: {duration_days:.1f} days ({dd.get('duration_hours', 0):.1f}h)")
        print(f"   Magnitude: {dd['max_drawdown_pct']:.2f}%")
        print(f"   Period: {dd['start_date'].strftime('%Y-%m-%d')} to {dd.get('end_date', 'ongoing').strftime('%Y-%m-%d') if hasattr(dd.get('end_date', 'ongoing'), 'strftime') else 'ongoing'}")
        print(f"   Price: ${dd['peak_price']:.2f} → ${dd['min_price']:.2f}")
        print()
    
    print("\n=== TOP 5 DEEPEST DRAWDOWNS ===")
    for i, dd in enumerate(by_magnitude[:5]):
        duration_days = dd.get('duration_hours', 0) / 24
        print(f"{i+1}. Magnitude: {dd['max_drawdown_pct']:.2f}%")
        print(f"   Duration: {duration_days:.1f} days ({dd.get('duration_hours', 0):.1f}h)")
        print(f"   Period: {dd['start_date'].strftime('%Y-%m-%d')} to {dd.get('end_date', 'ongoing').strftime('%Y-%m-%d') if hasattr(dd.get('end_date', 'ongoing'), 'strftime') else 'ongoing'}")
        print(f"   Price: ${dd['peak_price']:.2f} → ${dd['min_price']:.2f}")
        print()
    
    return by_duration, by_magnitude

def analyze_entry_conditions_for_worst_drawdowns(backtester, worst_drawdowns):
    """Analyze what 3commas filters could have prevented the worst drawdowns"""
    
    print("\n=== ANALYZING 3COMMAS FILTER OPPORTUNITIES ===")
    
    filter_effectiveness = {}
    
    for i, dd in enumerate(worst_drawdowns[:3]):  # Analyze top 3 worst
        print(f"\n--- DRAWDOWN #{i+1} ---")
        duration_days = dd.get('duration_hours', 0) / 24
        print(f"Duration: {duration_days:.1f} days, Magnitude: {dd['max_drawdown_pct']:.2f}%")
        print(f"Period: {dd['start_date'].strftime('%Y-%m-%d %H:%M')} to {dd.get('end_date', 'ongoing')}")
        
        # Get the index where this drawdown started (the peak before the fall)
        start_idx = dd['start_idx']
        
        # Look at conditions around the start of the drawdown period
        # Check multiple points during the early drawdown phase
        analysis_points = []
        
        # Check at drawdown start
        if start_idx < len(backtester.prices):
            analysis_points.append(('Drawdown Start', start_idx))
        
        # Check 25%, 50%, 75% into the drawdown
        if 'max_drawdown_idx' in dd:
            max_dd_idx = dd['max_drawdown_idx']
            quarter_point = start_idx + (max_dd_idx - start_idx) // 4
            half_point = start_idx + (max_dd_idx - start_idx) // 2
            three_quarter_point = start_idx + 3 * (max_dd_idx - start_idx) // 4
            
            if quarter_point < len(backtester.prices):
                analysis_points.append(('25% into drawdown', quarter_point))
            if half_point < len(backtester.prices):
                analysis_points.append(('50% into drawdown', half_point))
            if three_quarter_point < len(backtester.prices):
                analysis_points.append(('75% into drawdown', three_quarter_point))
        
        for point_name, idx in analysis_points:
            print(f"\n  {point_name} (Index {idx}):")
            
            if idx >= len(backtester.prices):
                continue
                
            price = backtester.prices[idx]
            indicators = get_indicators_at_time(backtester, idx)
            
            # Get lookback prices for volatility and higher highs analysis
            lookback_start = max(0, idx - 30)
            lookback_prices = backtester.prices[lookback_start:idx+1]
            
            # Test 3commas filters
            triggered_filters = test_3commas_filters(price, indicators, lookback_prices)
            
            print(f"    Price: ${price:.2f}")
            print(f"    RSI: {indicators['rsi_1h']:.1f}")
            
            if triggered_filters:
                print(f"    🚫 FILTERS THAT WOULD PREVENT ENTRY:")
                for filter_name in triggered_filters:
                    print(f"      - {filter_name}")
                    
                    # Count filter effectiveness
                    if filter_name not in filter_effectiveness:
                        filter_effectiveness[filter_name] = 0
                    filter_effectiveness[filter_name] += 1
            else:
                print(f"    ✅ No 3commas filters would prevent entry at this point")
    
    # Summary of most effective filters
    if filter_effectiveness:
        print(f"\n=== MOST EFFECTIVE 3COMMAS FILTERS ===")
        sorted_filters = sorted(filter_effectiveness.items(), key=lambda x: x[1], reverse=True)
        
        for filter_name, count in sorted_filters:
            print(f"{count}x - {filter_name}")
    
    return filter_effectiveness

def main():
    """Main analysis function"""
    print("🔍 ANALYZING WORST DRAWDOWNS AND 3COMMAS FILTER OPPORTUNITIES")
    print("=" * 70)
    
    # Run simulation with best parameters
    backtester, balances, timestamps, best_params = run_simulation_with_best_params()
    
    # Analyze worst drawdown periods
    by_duration, by_magnitude = analyze_worst_periods(backtester, balances, timestamps)
    
    if not by_duration:
        print("No drawdowns to analyze!")
        return
    
    # Analyze what 3commas filters could have helped
    print("\n" + "=" * 70)
    print("TESTING 3COMMAS FILTERS FOR WORST DRAWDOWNS")
    print("=" * 70)
    
    # Analyze both longest and deepest drawdowns
    worst_by_duration = by_duration[:3]
    worst_by_magnitude = by_magnitude[:3]
    
    # Combine and deduplicate
    all_worst = []
    seen_periods = set()
    
    for dd in worst_by_duration + worst_by_magnitude:
        period_key = (dd['start_date'], dd.get('end_date'))
        if period_key not in seen_periods:
            all_worst.append(dd)
            seen_periods.add(period_key)
    
    # Analyze entry conditions
    filter_effectiveness = analyze_entry_conditions_for_worst_drawdowns(backtester, all_worst)
    
    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    analysis_results = {
        'timestamp': timestamp,
        'simulation_results': {
            'apy': float(backtester.simulate_strategy_fast(best_params)[0]),
            'max_drawdown': float(backtester.simulate_strategy_fast(best_params)[1]),
            'total_trades': int(backtester.simulate_strategy_fast(best_params)[2])
        },
        'worst_drawdowns': {
            'by_duration': [
                {
                    'duration_hours': dd.get('duration_hours', 0),
                    'magnitude_pct': dd['max_drawdown_pct'],
                    'start_date': dd['start_date'].isoformat(),
                    'end_date': dd.get('end_date').isoformat() if dd.get('end_date') else None,
                    'price_range': f"${dd['peak_price']:.2f} → ${dd['min_price']:.2f}"
                }
                for dd in by_duration[:5]
            ],
            'by_magnitude': [
                {
                    'duration_hours': dd.get('duration_hours', 0),
                    'magnitude_pct': dd['max_drawdown_pct'],
                    'start_date': dd['start_date'].isoformat(),
                    'end_date': dd.get('end_date').isoformat() if dd.get('end_date') else None,
                    'price_range': f"${dd['peak_price']:.2f} → ${dd['min_price']:.2f}"
                }
                for dd in by_magnitude[:5]
            ]
        },
        'filter_effectiveness': filter_effectiveness,
        'best_parameters': {
            'base_percent': best_params.base_percent,
            'initial_deviation': best_params.initial_deviation,
            'trailing_deviation': best_params.trailing_deviation,
            'tp_level1': best_params.tp_level1,
            'rsi_entry_threshold': best_params.rsi_entry_threshold,
            'rsi_safety_threshold': best_params.rsi_safety_threshold
        }
    }
    
    results_path = f"results/drawdown_analysis_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n📊 Analysis results saved to: {results_path}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS FOR NEXT OPTIMIZATION:")
    if filter_effectiveness:
        top_filters = sorted(filter_effectiveness.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Consider enabling these 3commas filters:")
        for filter_name, count in top_filters:
            print(f"  - {filter_name}")
    else:
        print("  - Current strategy seems robust, consider other optimization approaches")
    
    return analysis_results

if __name__ == "__main__":
    main()
