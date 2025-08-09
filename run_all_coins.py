#!/usr/bin/env python3
"""
Simple script to run run_supertrend_backtest for all coins
"""

import os
from pathlib import Path


def get_coins():
    """Get list of available coins"""
    data_dir = Path('data')
    coins = []
    for file in data_dir.glob('*_1m.csv'):
        coin = file.stem.replace('_1m', '')
        coins.append(coin)
    return sorted(coins)


def main():
    trials = "100000"

    coins = get_coins()
    print(f"🚀 Running optimization for {len(coins)} coins with {trials} trials each")
    print(f"Coins: {', '.join(coins)}")
    print()

    for i, coin in enumerate(coins, 1):
        print(f"📈 {i}/{len(coins)}: {coin}")
        cmd = f'python run_supertrend_backtest.py --coin {coin} --data_path data/{coin}_1m.csv --optimize --trials {trials}'
        print(f"Running: {cmd}")
        print("-" * 60)

        # Just run the command directly
        os.system(cmd)

        print(f"✅ {coin} completed")
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
