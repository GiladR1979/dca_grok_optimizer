#!/usr/bin/env python3
"""
Run fast_dca_backtest.py with ultra-aggressive parameters directly
"""

import subprocess
import sys

# Build the command with all ultra-aggressive parameters
cmd = [
    sys.executable,
    'fast_dca_backtest.py',
    '--data_path', 'data/SOLUSDT_1m.csv',
    '--coin', 'SOL',
    '--initial_balance', '10000',
    '--preset', 'aggressive'  # This will use the aggressive preset as a base
]

print("Running fast_dca_backtest.py with aggressive preset...")
print("This will generate a chart with ALL trade markers visible")
print("-" * 60)

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print("\nNote: The chart should now show ALL trade markers as requested.")
print("Check the results folder for the generated PNG file.")
