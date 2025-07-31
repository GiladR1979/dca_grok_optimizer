#!/usr/bin/env python3
"""
Optimize DCA strategy for all available coins
Usage: python optimize_all_coins.py [options]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import subprocess
import json
import pandas as pd

def get_available_coins():
    """Get list of available coin data files"""
    data_dir = Path('data')
    if not data_dir.exists():
        print("❌ Data directory not found")
        return []
    
    coins = []
    for file in data_dir.glob('*_1m.csv'):
        # Extract coin name from filename (e.g., SOLUSDT_1m.csv -> SOLUSDT)
        coin_name = file.stem.replace('_1m', '')
        coins.append({
            'coin': coin_name,
            'file': str(file),
            'size_mb': file.stat().st_size / (1024 * 1024)
        })
    
    return sorted(coins, key=lambda x: x['coin'])

def run_optimization(coin_data, trials=5000, timeout_minutes=30):
    """Run optimization for a single coin"""
    coin = coin_data['coin']
    data_path = coin_data['file']
    
    print(f"\n🚀 OPTIMIZING {coin}")
    print("=" * 60)
    print(f"Data file: {data_path}")
    print(f"File size: {coin_data['size_mb']:.1f} MB")
    print(f"Trials: {trials:,}")
    print(f"Timeout: {timeout_minutes} minutes")
    
    # Build command
    cmd = [
        'python', 'fast_dca_backtest.py',
        '--data_path', data_path,
        '--coin', coin,
        '--optimize',
        '--trials', str(trials)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\n🔄 Starting optimization...")
    
    start_time = time.time()
    
    try:
        # Run with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=os.getcwd()
        )
        
        # Capture output in real-time
        output_lines = []
        apy = None
        drawdown = None
        trades = None
        
        print("📊 Live output:")
        print("-" * 40)
        
        # Read output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                output_lines.append(line)
                
                # Show important lines immediately
                if any(keyword in line for keyword in ['Best:', 'APY:', 'GPU detected', 'FAST BACKTEST RESULTS', 'Final Balance:', 'Max Drawdown:', 'Total Trades:']):
                    print(f"   {line}")
                elif 'trials' in line.lower() and '%' in line:
                    print(f"   {line}")
                
                # Extract metrics as they come
                if 'APY:' in line:
                    try:
                        apy = float(line.split('APY:')[1].split('%')[0].strip())
                    except:
                        pass
                elif 'Max Drawdown:' in line:
                    try:
                        drawdown = float(line.split('Max Drawdown:')[1].split('%')[0].strip())
                    except:
                        pass
                elif 'Total Trades:' in line:
                    try:
                        trades = int(line.split('Total Trades:')[1].strip().replace(',', ''))
                    except:
                        pass
        
        # Wait for process to complete
        stderr_output = process.stderr.read()
        return_code = process.poll()
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            print("-" * 40)
            print(f"✅ {coin} completed in {elapsed_time/60:.1f} minutes")
            if apy is not None:
                print(f"   📈 Final APY: {apy:.2f}%")
            if drawdown is not None:
                print(f"   📉 Max Drawdown: {drawdown:.2f}%")
            if trades is not None:
                print(f"   🔄 Total Trades: {trades:,}")
            
            return {
                'coin': coin,
                'status': 'success',
                'apy': apy,
                'max_drawdown': drawdown,
                'total_trades': trades,
                'elapsed_minutes': elapsed_time / 60,
                'trials': trials,
                'output': '\n'.join(output_lines)
            }
        else:
            print(f"❌ {coin} failed with return code {return_code}")
            print(f"Error: {stderr_output}")
            return {
                'coin': coin,
                'status': 'failed',
                'error': stderr_output,
                'elapsed_minutes': elapsed_time / 60,
                'trials': trials
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {coin} timed out after {timeout_minutes} minutes")
        return {
            'coin': coin,
            'status': 'timeout',
            'elapsed_minutes': timeout_minutes,
            'trials': trials
        }
    except Exception as e:
        print(f"💥 {coin} crashed: {e}")
        return {
            'coin': coin,
            'status': 'crashed',
            'error': str(e),
            'elapsed_minutes': (time.time() - start_time) / 60,
            'trials': trials
        }

def save_summary_report(results, trials):
    """Save a summary report of all optimizations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed JSON report
    json_file = results_dir / f"all_coins_optimization_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'trials_per_coin': trials,
            'total_coins': len(results),
            'results': results
        }, f, indent=2, default=str)
    
    # Create summary CSV
    csv_data = []
    for result in results:
        csv_data.append({
            'Coin': result['coin'],
            'Status': result['status'],
            'APY (%)': result.get('apy', 'N/A'),
            'Max Drawdown (%)': result.get('max_drawdown', 'N/A'),
            'Total Trades': result.get('total_trades', 'N/A'),
            'Time (min)': f"{result['elapsed_minutes']:.1f}",
            'Trials': result['trials']
        })
    
    csv_file = results_dir / f"all_coins_summary_{timestamp}.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    if successful:
        print(f"\n✅ SUCCESSFUL OPTIMIZATIONS ({len(successful)}):")
        print("-" * 80)
        print(f"{'Coin':<12} {'APY (%)':<8} {'Drawdown (%)':<12} {'Trades':<8} {'Time (min)':<10}")
        print("-" * 80)
        
        # Sort by APY descending
        successful_sorted = sorted(successful, key=lambda x: x.get('apy', 0), reverse=True)
        
        for result in successful_sorted:
            apy = f"{result.get('apy', 0):.2f}" if result.get('apy') else "N/A"
            dd = f"{result.get('max_drawdown', 0):.2f}" if result.get('max_drawdown') else "N/A"
            trades = f"{result.get('total_trades', 0):,}" if result.get('total_trades') else "N/A"
            time_str = f"{result['elapsed_minutes']:.1f}"
            
            print(f"{result['coin']:<12} {apy:<8} {dd:<12} {trades:<8} {time_str:<10}")
        
        # Best performers
        best_apy = max(successful, key=lambda x: x.get('apy', 0))
        best_risk_adj = min([r for r in successful if r.get('apy') and r.get('max_drawdown') and r.get('max_drawdown') > 0], 
                           key=lambda x: x.get('max_drawdown', 100) / max(x.get('apy', 1), 1), default=None)
        
        print(f"\n🏆 BEST APY: {best_apy['coin']} - {best_apy.get('apy', 0):.2f}%")
        if best_risk_adj:
            risk_ratio = best_risk_adj.get('apy', 0) / best_risk_adj.get('max_drawdown', 1)
            print(f"⚖️  BEST RISK-ADJUSTED: {best_risk_adj['coin']} - {risk_ratio:.3f} (APY/DD)")
    
    if failed:
        print(f"\n❌ FAILED OPTIMIZATIONS ({len(failed)}):")
        print("-" * 40)
        for result in failed:
            status = result['status'].upper()
            time_str = f"{result['elapsed_minutes']:.1f}min"
            print(f"{result['coin']:<12} {status:<10} {time_str}")
    
    total_time = sum(r['elapsed_minutes'] for r in results)
    print(f"\n⏱️  TOTAL TIME: {total_time/60:.1f} hours ({total_time:.1f} minutes)")
    print(f"📁 Results saved to:")
    print(f"   📄 {json_file}")
    print(f"   📊 {csv_file}")
    
    return json_file, csv_file

def main():
    parser = argparse.ArgumentParser(description='Optimize DCA strategy for all available coins')
    parser.add_argument('--trials', type=int, default=5000, help='Number of optimization trials per coin (default: 5000)')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout per coin in minutes (default: 30)')
    parser.add_argument('--coins', nargs='+', help='Specific coins to optimize (default: all available)')
    parser.add_argument('--list', action='store_true', help='List available coins and exit')
    parser.add_argument('--skip', nargs='+', help='Coins to skip')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes (default: 1)')
    
    args = parser.parse_args()
    
    # Get available coins
    available_coins = get_available_coins()
    
    if not available_coins:
        print("❌ No coin data files found in data/ directory")
        return
    
    # List coins and exit if requested
    if args.list:
        print("📁 Available coins:")
        for coin_data in available_coins:
            print(f"   {coin_data['coin']:<12} ({coin_data['size_mb']:.1f} MB)")
        return
    
    # Filter coins if specific ones requested
    if args.coins:
        available_coins = [c for c in available_coins if c['coin'] in args.coins]
        if not available_coins:
            print(f"❌ None of the requested coins found: {args.coins}")
            return
    
    # Skip coins if requested
    if args.skip:
        available_coins = [c for c in available_coins if c['coin'] not in args.skip]
        print(f"⏭️  Skipping coins: {args.skip}")
    
    print("🚀 MULTI-COIN DCA OPTIMIZATION")
    print("=" * 60)
    print(f"Coins to optimize: {len(available_coins)}")
    print(f"Trials per coin: {args.trials:,}")
    print(f"Timeout per coin: {args.timeout} minutes")
    print(f"Estimated total time: {len(available_coins) * args.timeout / 60:.1f} hours (worst case)")
    print()
    
    for i, coin_data in enumerate(available_coins, 1):
        print(f"   {i}. {coin_data['coin']} ({coin_data['size_mb']:.1f} MB)")
    
    # Confirm before starting
    response = input(f"\n🤔 Proceed with optimization? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Optimization cancelled")
        return
    
    # Run optimizations
    print(f"\n🏁 Starting optimization of {len(available_coins)} coins...")
    start_time = time.time()
    
    results = []
    for i, coin_data in enumerate(available_coins, 1):
        print(f"\n📈 Progress: {i}/{len(available_coins)} coins")
        result = run_optimization(coin_data, args.trials, args.timeout)
        results.append(result)
        
        # Show running summary
        successful = len([r for r in results if r['status'] == 'success'])
        print(f"   ✅ Successful: {successful}/{i}")
    
    # Save and display final summary
    total_elapsed = time.time() - start_time
    print(f"\n🏁 ALL OPTIMIZATIONS COMPLETE!")
    print(f"⏱️  Total time: {total_elapsed/3600:.1f} hours")
    
    save_summary_report(results, args.trials)

if __name__ == "__main__":
    main()
