#!/usr/bin/env python3
"""
Test script using full dataset to ensure we have valid indicators
"""

import pandas as pd
import numpy as np
from strategy_config import StrategyParams
from dca_backtest import Backtester

def test_full_data():
    """Test with full dataset to get valid indicators"""
    print("=== TESTING WITH FULL DATASET ===")
    
    # Load full dataset
    data = pd.read_csv('data/SOLUSDT_1m.csv')
    data['ts'] = pd.to_datetime(data['ts'])
    data.set_index('ts', inplace=True)
    
    print(f"Full dataset: {len(data)} points")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Test with full data
    backtester = Backtester(data, 10000)
    params = StrategyParams()
    
    # Check indicators with more data
    indicators = backtester.indicators
    print("\n=== INDICATOR SUMMARY ===")
    for name, values in indicators.items():
        valid_count = (~np.isnan(values)).sum()
        total_count = len(values)
        print(f"{name}: {valid_count}/{total_count} valid ({valid_count/total_count*100:.1f}%)")
        
        if valid_count > 0:
            valid_values = values.dropna()
            print(f"  Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
            print(f"  Mean: {valid_values.mean():.2f}")
    
    # Find first valid entry point
    print("\n=== FINDING FIRST VALID ENTRY ===")
    entry_count = 0
    
    # Skip the first 26 hours to ensure SMA_26 has valid data
    start_idx = 26 * 60  # 26 hours * 60 minutes
    test_data = data.iloc[start_idx:]
    
    for timestamp, row in test_data.iterrows():
        indicators = {
            'rsi_1h': backtester.get_indicator_value(timestamp, 'rsi_1h'),
            'sma_fast_1h': backtester.get_indicator_value(timestamp, 'sma_fast_1h'),
            'sma_slow_1h': backtester.get_indicator_value(timestamp, 'sma_slow_1h'),
        }
        
        # Skip if any indicators are NaN
        if any(np.isnan([indicators['rsi_1h'], indicators['sma_fast_1h'], indicators['sma_slow_1h']])):
            continue
            
        rsi_ok = indicators['rsi_1h'] < params.rsi_entry_threshold
        sma_ok = indicators['sma_fast_1h'] > indicators['sma_slow_1h']
        
        if rsi_ok and sma_ok:
            entry_count += 1
            print(f"Entry #{entry_count} at {timestamp}:")
            print(f"  RSI: {indicators['rsi_1h']:.1f} < {params.rsi_entry_threshold}")
            print(f"  SMA: {indicators['sma_fast_1h']:.2f} > {indicators['sma_slow_1h']:.2f}")
            print(f"  Price: ${row['close']:.2f}")
            
            if entry_count >= 5:
                break
    
    if entry_count == 0:
        print("No entry conditions found in full dataset")
        
        # Let's check the indicator ranges
        print("\n=== INDICATOR ANALYSIS ===")
        valid_rsi = []
        valid_sma_fast = []
        valid_sma_slow = []
        
        for timestamp, row in test_data.iterrows():
            rsi = backtester.get_indicator_value(timestamp, 'rsi_1h')
            sma_fast = backtester.get_indicator_value(timestamp, 'sma_fast_1h')
            sma_slow = backtester.get_indicator_value(timestamp, 'sma_slow_1h')
            
            if not any(np.isnan([rsi, sma_fast, sma_slow])):
                valid_rsi.append(rsi)
                valid_sma_fast.append(sma_fast)
                valid_sma_slow.append(sma_slow)
        
        if valid_rsi:
            print(f"RSI range: {min(valid_rsi):.1f} - {max(valid_rsi):.1f}")
            print(f"RSI < 40 count: {sum(r < 40 for r in valid_rsi)}")
            print(f"SMA cross count: {sum(f > s for f, s in zip(valid_sma_fast, valid_sma_slow))}")
    
    # Run full simulation
    print("\n=== RUNNING FULL SIMULATION ===")
    apy, dd, balance_history, trades = backtester.simulate_strategy(params)
    print(f"APY: {apy:.2f}%")
    print(f"Max DD: {dd:.2f}%")
    print(f"Total trades: {len(trades)}")
    
    if trades:
        print(f"First trade: {trades[0].timestamp} - {trades[0].action} {trades[0].amount_coin:.4f} at ${trades[0].price:.2f}")
        print(f"Last trade: {trades[-1].timestamp} - {trades[-1].action} {trades[-1].amount_coin:.4f} at ${trades[-1].price:.2f}")
    else:
        print("No trades executed")

if __name__ == "__main__":
    test_full_data()
