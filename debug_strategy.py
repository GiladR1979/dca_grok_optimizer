#!/usr/bin/env python3
"""
Debug script to identify why no trades are being executed
"""

import pandas as pd
import numpy as np
from strategy_config import StrategyParams
from dca_backtest import Backtester, DataProcessor
from fast_dca_backtest import FastBacktester

def debug_indicators():
    """Debug indicator calculation issues"""
    print("=== DEBUGGING INDICATORS ===")
    
    # Load more data for better indicator calculation
    data = pd.read_csv('data/SOLUSDT_1m.csv')
    data['ts'] = pd.to_datetime(data['ts'])
    data.set_index('ts', inplace=True)
    
    # Use more data for proper indicator calculation
    data = data.tail(5000)  # ~3.5 days
    
    print(f"Data points: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Test original implementation
    print("\n=== ORIGINAL IMPLEMENTATION ===")
    backtester = Backtester(data, 10000)
    
    # Check indicators
    indicators = backtester.indicators
    for name, values in indicators.items():
        valid_count = (~np.isnan(values)).sum()
        print(f"{name}: {valid_count}/{len(values)} valid ({valid_count/len(values)*100:.1f}%)")
        if valid_count > 0:
            print(f"  Range: {np.nanmin(values):.2f} - {np.nanmax(values):.2f}")
            print(f"  First 10: {values[:10]}")
    
    # Test a few data points manually
    params = StrategyParams()
    print(f"\n=== STRATEGY PARAMETERS ===")
    print(f"RSI entry threshold: {params.rsi_entry_threshold}")
    print(f"RSI safety threshold: {params.rsi_safety_threshold}")
    
    # Check first few rows
    print(f"\n=== SAMPLE DATA CHECK ===")
    for i, (timestamp, row) in enumerate(data.head(10).iterrows()):
        indicators = {
            'rsi_1h': backtester.get_indicator_value(timestamp, 'rsi_1h'),
            'rsi_4h': backtester.get_indicator_value(timestamp, 'rsi_4h'),
            'sma_fast_1h': backtester.get_indicator_value(timestamp, 'sma_fast_1h'),
            'sma_slow_1h': backtester.get_indicator_value(timestamp, 'sma_slow_1h'),
        }
        
        rsi_ok = indicators['rsi_1h'] < params.rsi_entry_threshold
        sma_ok = indicators['sma_fast_1h'] > indicators['sma_slow_1h']
        
        print(f"{timestamp}: RSI={indicators['rsi_1h']:.1f}, SMA_fast={indicators['sma_fast_1h']:.2f}, SMA_slow={indicators['sma_slow_1h']:.2f}")
        print(f"  Entry conditions: RSI<{params.rsi_entry_threshold}={rsi_ok}, SMA_cross={sma_ok}")
        
        if rsi_ok and sma_ok:
            print("  ✓ ENTRY CONDITIONS MET!")
            break
    
    # Run full simulation
    print(f"\n=== FULL SIMULATION ===")
    apy, dd, balance_history, trades = backtester.simulate_strategy(params)
    print(f"APY: {apy:.2f}%")
    print(f"Max DD: {dd:.2f}%")
    print(f"Total trades: {len(trades)}")
    
    if trades:
        print("First few trades:")
        for trade in trades[:5]:
            print(f"  {trade.timestamp}: {trade.action} {trade.amount_coin:.4f} at ${trade.price:.2f} ({trade.reason})")
    else:
        print("No trades executed - checking why...")
        
        # Check if we have any valid entry conditions
        entry_count = 0
        for timestamp, row in data.iterrows():
            indicators = {
                'rsi_1h': backtester.get_indicator_value(timestamp, 'rsi_1h'),
                'sma_fast_1h': backtester.get_indicator_value(timestamp, 'sma_fast_1h'),
                'sma_slow_1h': backtester.get_indicator_value(timestamp, 'sma_slow_1h'),
            }
            
            rsi_ok = indicators['rsi_1h'] < params.rsi_entry_threshold
            sma_ok = indicators['sma_fast_1h'] > indicators['sma_slow_1h']
            
            if rsi_ok and sma_ok:
                entry_count += 1
                if entry_count <= 3:
                    print(f"Entry condition met at {timestamp}: RSI={indicators['rsi_1h']:.1f}, SMA={indicators['sma_fast_1h']:.2f}>{indicators['sma_slow_1h']:.2f}")
        
        print(f"Total entry conditions met: {entry_count}")

if __name__ == "__main__":
    debug_indicators()
