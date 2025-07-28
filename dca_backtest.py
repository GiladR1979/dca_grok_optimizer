#!/usr/bin/env python3
"""
DCA Backtesting Program
A comprehensive backtesting system for Dollar-Cost Averaging trading strategies
"""

import pandas as pd
import numpy as np
import argparse
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import ta
except ImportError:
    print("Installing ta-lib alternative...")
    import subprocess

    subprocess.check_call(['pip', 'install', 'ta'])
    import ta

try:
    import optuna
except ImportError:
    print("Installing optuna...")
    import subprocess

    subprocess.check_call(['pip', 'install', 'optuna'])
    import optuna

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bars...")
    import subprocess

    subprocess.check_call(['pip', 'install', 'tqdm'])
    from tqdm import tqdm


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    action: str  # 'buy' or 'sell'
    amount_coin: float
    price: float
    usdt_amount: float
    average_entry: float
    position_size: float
    reason: str


@dataclass
class StrategyParams:
    """Strategy parameters for optimization"""
    base_percent: float = 1.0  # Base order as % of balance
    initial_deviation: float = 3.0  # Initial safety order deviation %
    step_multiplier: float = 1.5  # Geometric step multiplier
    volume_multiplier: float = 1.2  # Volume scaling multiplier
    max_safeties: int = 8  # Maximum safety orders
    trailing_deviation: float = 7.0  # Trailing stop %
    tp_level1: float = 5.0  # First TP level %
    tp_level2: float = 10.0  # Second TP level %
    tp_level3: float = 15.0  # Third TP level %
    tp_percent1: float = 50.0  # % to sell at TP1
    tp_percent2: float = 30.0  # % to sell at TP2
    tp_percent3: float = 20.0  # % to sell at TP3
    rsi_entry_threshold: float = 30.0  # RSI entry threshold
    rsi_safety_threshold: float = 25.0  # RSI safety threshold
    rsi_exit_threshold: float = 70.0  # RSI exit threshold
    fees: float = 0.0  # Trading fees %


class DCAStrategy:
    """Main DCA strategy implementation"""

    def __init__(self, params: StrategyParams, initial_balance: float = 10000):
        self.params = params
        self.initial_balance = initial_balance
        self.reset_state()

    def reset_state(self):
        """Reset strategy state for new simulation"""
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.average_entry = 0.0
        self.total_spent = 0.0
        self.active_deal = False
        self.safety_count = 0
        self.last_entry_price = 0.0
        self.peak_price = 0.0
        self.trailing_active = False
        self.last_trade_time = None
        self.trades = []
        self.balance_history = []
        self.tp_levels_hit = {'tp1': False, 'tp2': False, 'tp3': False}

    def add_trade(self, timestamp: datetime, action: str, amount_coin: float,
                  price: float, usdt_amount: float, reason: str):
        """Add a trade to the history"""
        trade = Trade(
            timestamp=timestamp,
            action=action,
            amount_coin=amount_coin,
            price=price,
            usdt_amount=usdt_amount,
            average_entry=self.average_entry,
            position_size=self.position_size,
            reason=reason
        )
        self.trades.append(trade)

    def can_enter_new_deal(self, current_time: datetime) -> bool:
        """Check if we can enter a new deal (cooldown logic)"""
        if not self.active_deal and self.last_trade_time:
            cooldown = timedelta(minutes=5)
            return current_time - self.last_trade_time >= cooldown
        return not self.active_deal

    def check_entry_conditions(self, row: pd.Series, indicators: Dict) -> bool:
        """Check if entry conditions are met"""
        try:
            rsi_4h = indicators.get('rsi_4h', 50)
            sma_fast_1h = indicators.get('sma_fast_1h', row['close'])
            sma_slow_1h = indicators.get('sma_slow_1h', row['close'])
            sma_50_1h = indicators.get('sma_50_1h', row['close'])

            # Entry conditions
            rsi_condition = rsi_4h < self.params.rsi_entry_threshold
            sma_condition = sma_fast_1h > sma_slow_1h
            momentum_condition = row['close'] > sma_50_1h

            return rsi_condition and sma_condition and momentum_condition
        except:
            return False

    def execute_base_order(self, row: pd.Series):
        """Execute base order"""
        base_amount_usdt = self.balance * (self.params.base_percent / 100)
        amount_coin = base_amount_usdt / row['close']

        # Apply fees
        fee_amount = base_amount_usdt * (self.params.fees / 100)
        net_amount_usdt = base_amount_usdt - fee_amount
        amount_coin = net_amount_usdt / row['close']

        self.balance -= base_amount_usdt
        self.position_size = amount_coin
        self.total_spent = base_amount_usdt
        self.average_entry = row['close']
        self.last_entry_price = row['close']
        self.active_deal = True
        self.safety_count = 0
        self.peak_price = row['close']
        self.tp_levels_hit = {'tp1': False, 'tp2': False, 'tp3': False}

        self.add_trade(row.name, 'buy', amount_coin, row['close'],
                       base_amount_usdt, 'base_order')

    def check_safety_conditions(self, row: pd.Series, indicators: Dict) -> bool:
        """Check if safety order should be triggered"""
        if self.safety_count >= self.params.max_safeties:
            return False

        # Calculate current deviation
        current_deviation = self.params.initial_deviation
        for i in range(self.safety_count):
            current_deviation *= self.params.step_multiplier

        price_drop_threshold = self.last_entry_price * (1 - current_deviation / 100)
        price_condition = row['close'] <= price_drop_threshold

        # RSI condition for safety
        rsi_1h = indicators.get('rsi_1h', 50)
        rsi_condition = rsi_1h < self.params.rsi_safety_threshold

        return price_condition and rsi_condition

    def execute_safety_order(self, row: pd.Series):
        """Execute safety order"""
        # Calculate safety order volume
        base_amount = self.initial_balance * (self.params.base_percent / 100)
        safety_multiplier = self.params.volume_multiplier ** self.safety_count
        safety_amount_usdt = base_amount * safety_multiplier

        # Check if we have enough balance
        if safety_amount_usdt > self.balance:
            safety_amount_usdt = self.balance * 0.95  # Use 95% of remaining balance

        if safety_amount_usdt < 1:  # Minimum order size
            return

        # Apply fees
        fee_amount = safety_amount_usdt * (self.params.fees / 100)
        net_amount_usdt = safety_amount_usdt - fee_amount
        amount_coin = net_amount_usdt / row['close']

        self.balance -= safety_amount_usdt
        self.position_size += amount_coin
        self.total_spent += safety_amount_usdt
        self.average_entry = self.total_spent / self.position_size
        self.last_entry_price = row['close']
        self.safety_count += 1

        self.add_trade(row.name, 'buy', amount_coin, row['close'],
                       safety_amount_usdt, f'safety_order_{self.safety_count}')

    def check_take_profit_conditions(self, row: pd.Series) -> List[Tuple[str, float, float]]:
        """Check take profit conditions and return list of (level, percent, price_threshold)"""
        if self.position_size <= 0:
            return []

        profit_percent = (row['close'] - self.average_entry) / self.average_entry * 100
        tp_triggers = []

        # Check TP levels
        if profit_percent >= self.params.tp_level1 and not self.tp_levels_hit['tp1']:
            tp_triggers.append(('tp1', self.params.tp_percent1 / 100, self.params.tp_level1))

        if profit_percent >= self.params.tp_level2 and not self.tp_levels_hit['tp2']:
            tp_triggers.append(('tp2', self.params.tp_percent2 / 100, self.params.tp_level2))

        if profit_percent >= self.params.tp_level3 and not self.tp_levels_hit['tp3']:
            tp_triggers.append(('tp3', self.params.tp_percent3 / 100, self.params.tp_level3))

        return tp_triggers

    def execute_take_profit(self, row: pd.Series, tp_level: str, sell_percent: float):
        """Execute take profit order"""
        amount_to_sell = self.position_size * sell_percent
        usdt_received = amount_to_sell * row['close']

        # Apply fees
        fee_amount = usdt_received * (self.params.fees / 100)
        net_usdt = usdt_received - fee_amount

        self.balance += net_usdt
        self.position_size -= amount_to_sell
        self.tp_levels_hit[tp_level] = True

        # Enable trailing after first TP
        if tp_level == 'tp1':
            self.trailing_active = True
            self.peak_price = row['close']

        self.add_trade(row.name, 'sell', amount_to_sell, row['close'],
                       usdt_received, f'take_profit_{tp_level}')

        # Close deal if position is empty
        if self.position_size < 0.0001:
            self.close_deal(row.name)

    def check_trailing_stop(self, row: pd.Series) -> bool:
        """Check trailing stop condition"""
        if not self.trailing_active or self.position_size <= 0:
            return False

        # Update peak price
        if row['close'] > self.peak_price:
            self.peak_price = row['close']

        # Check trailing stop
        trailing_threshold = self.peak_price * (1 - self.params.trailing_deviation / 100)
        return row['close'] <= trailing_threshold

    def check_force_exit_conditions(self, row: pd.Series, indicators: Dict) -> bool:
        """Check conditions that force exit"""
        rsi_4h = indicators.get('rsi_4h', 50)
        return rsi_4h > self.params.rsi_exit_threshold

    def execute_full_exit(self, row: pd.Series, reason: str):
        """Execute full position exit"""
        if self.position_size <= 0:
            return

        usdt_received = self.position_size * row['close']
        fee_amount = usdt_received * (self.params.fees / 100)
        net_usdt = usdt_received - fee_amount

        self.balance += net_usdt

        self.add_trade(row.name, 'sell', self.position_size, row['close'],
                       usdt_received, reason)

        self.close_deal(row.name)

    def close_deal(self, timestamp: datetime):
        """Close the current deal"""
        self.active_deal = False
        self.position_size = 0.0
        self.average_entry = 0.0
        self.total_spent = 0.0
        self.safety_count = 0
        self.trailing_active = False
        self.last_trade_time = timestamp
        self.tp_levels_hit = {'tp1': False, 'tp2': False, 'tp3': False}


class DataProcessor:
    """Handle data processing and indicator calculation"""

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and preprocess data from CSV"""
        df = pd.read_csv(file_path)
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators"""
        indicators = {}

        # Resample to different timeframes
        df_1h = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        }).dropna()

        df_4h = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        }).dropna()

        # Calculate indicators
        try:
            # RSI indicators
            indicators['rsi_1h'] = ta.momentum.RSIIndicator(df_1h['close'], window=14).rsi()
            indicators['rsi_4h'] = ta.momentum.RSIIndicator(df_4h['close'], window=14).rsi()

            # SMA indicators
            indicators['sma_fast_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=12).sma_indicator()
            indicators['sma_slow_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=26).sma_indicator()
            indicators['sma_50_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=50).sma_indicator()

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # Fallback to simple calculations
            indicators['rsi_1h'] = pd.Series(50, index=df_1h.index)
            indicators['rsi_4h'] = pd.Series(50, index=df_4h.index)
            indicators['sma_fast_1h'] = df_1h['close'].rolling(12).mean()
            indicators['sma_slow_1h'] = df_1h['close'].rolling(26).mean()
            indicators['sma_50_1h'] = df_1h['close'].rolling(50).mean()

        return indicators, df_1h, df_4h


class Backtester:
    """Main backtesting engine"""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance
        self.indicators, self.df_1h, self.df_4h = DataProcessor.calculate_indicators(data)

    def get_indicator_value(self, timestamp: datetime, indicator_name: str) -> float:
        """Get indicator value at specific timestamp with forward fill"""
        try:
            if indicator_name.endswith('_1h'):
                series = self.indicators[indicator_name]
                # Find the most recent value before or at timestamp
                valid_times = series.index[series.index <= timestamp]
                if len(valid_times) > 0:
                    return series.loc[valid_times[-1]]
            elif indicator_name.endswith('_4h'):
                series = self.indicators[indicator_name]
                valid_times = series.index[series.index <= timestamp]
                if len(valid_times) > 0:
                    return series.loc[valid_times[-1]]
        except:
            pass
        return 50.0  # Default fallback value

    def simulate_strategy(self, params: StrategyParams) -> Tuple[float, float, List, List]:
        """Simulate strategy with given parameters"""
        strategy = DCAStrategy(params, self.initial_balance)

        for timestamp, row in self.data.iterrows():
            # Get current indicator values
            current_indicators = {
                'rsi_1h': self.get_indicator_value(timestamp, 'rsi_1h'),
                'rsi_4h': self.get_indicator_value(timestamp, 'rsi_4h'),
                'sma_fast_1h': self.get_indicator_value(timestamp, 'sma_fast_1h'),
                'sma_slow_1h': self.get_indicator_value(timestamp, 'sma_slow_1h'),
                'sma_50_1h': self.get_indicator_value(timestamp, 'sma_50_1h')
            }

            # Check for new entry
            if strategy.can_enter_new_deal(timestamp) and strategy.check_entry_conditions(row, current_indicators):
                strategy.execute_base_order(row)

            # Check active deal logic
            if strategy.active_deal:
                # Check safety orders
                if strategy.check_safety_conditions(row, current_indicators):
                    strategy.execute_safety_order(row)

                # Check take profit
                tp_triggers = strategy.check_take_profit_conditions(row)
                for tp_level, sell_percent, _ in tp_triggers:
                    strategy.execute_take_profit(row, tp_level, sell_percent)

                # Check trailing stop
                if strategy.check_trailing_stop(row):
                    strategy.execute_full_exit(row, 'trailing_stop')

                # Check force exit
                elif strategy.check_force_exit_conditions(row, current_indicators):
                    strategy.execute_full_exit(row, 'rsi_force_exit')

            # Record balance history
            total_value = strategy.balance
            if strategy.position_size > 0:
                total_value += strategy.position_size * row['close']
            strategy.balance_history.append((timestamp, total_value))

        # Close any remaining position
        if strategy.active_deal:
            last_row = self.data.iloc[-1]
            strategy.execute_full_exit(last_row, 'end_of_data')

        # Calculate metrics
        final_balance = strategy.balance
        apy = self.calculate_apy(self.initial_balance, final_balance,
                                 self.data.index[0], self.data.index[-1])
        max_drawdown = self.calculate_max_drawdown(strategy.balance_history)

        return apy, max_drawdown, strategy.balance_history, strategy.trades

    @staticmethod
    def calculate_apy(initial: float, final: float, start_date: datetime, end_date: datetime) -> float:
        """Calculate annualized percentage yield"""
        if initial <= 0:
            return 0.0
        days = (end_date - start_date).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        return (pow(final / initial, 1 / years) - 1) * 100

    @staticmethod
    def calculate_max_drawdown(balance_history: List[Tuple[datetime, float]]) -> float:
        """Calculate maximum drawdown percentage"""
        if not balance_history:
            return 0.0

        peak = balance_history[0][1]
        max_dd = 0.0

        for _, balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_dd = max(max_dd, drawdown)

        return max_dd


class Optimizer:
    """Strategy parameter optimization"""

    def __init__(self, backtester: Backtester):
        self.backtester = backtester
        self.best_fitness = -1000
        self.best_apy = 0
        self.best_drawdown = 100
        self.best_params = {}
        self.trial_count = 0
        self.progress_bar = None

    def objective(self, trial):
        """Optuna objective function"""
        params = StrategyParams(
            base_percent=trial.suggest_float('base_percent', 0.5, 2.0),
            initial_deviation=trial.suggest_float('initial_deviation', 2.0, 5.0),
            step_multiplier=trial.suggest_float('step_multiplier', 1.1, 1.6),
            volume_multiplier=trial.suggest_float('volume_multiplier', 1.1, 1.6),
            max_safeties=trial.suggest_int('max_safeties', 5, 10),
            trailing_deviation=trial.suggest_float('trailing_deviation', 5.0, 10.0),
            tp_level1=trial.suggest_float('tp_level1', 3.0, 8.0),
            tp_level2=trial.suggest_float('tp_level2', 8.0, 15.0),
            tp_level3=trial.suggest_float('tp_level3', 15.0, 25.0),
            rsi_entry_threshold=trial.suggest_float('rsi_entry_threshold', 25.0, 35.0),
            rsi_safety_threshold=trial.suggest_float('rsi_safety_threshold', 20.0, 30.0),
            rsi_exit_threshold=trial.suggest_float('rsi_exit_threshold', 65.0, 75.0),
            fees=trial.suggest_float('fees', 0.0, 0.1)
        )

        try:
            apy, max_drawdown, _, _ = self.backtester.simulate_strategy(params)

            # Multi-objective: maximize APY, minimize drawdown
            # Using weighted sum approach
            fitness = 0.6 * apy - 0.4 * max_drawdown

            # Update best results
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_apy = apy
                self.best_drawdown = max_drawdown
                self.best_params = {
                    'base_percent': params.base_percent,
                    'initial_deviation': params.initial_deviation,
                    'step_multiplier': params.step_multiplier,
                    'volume_multiplier': params.volume_multiplier,
                    'max_safeties': params.max_safeties,
                    'trailing_deviation': params.trailing_deviation,
                    'tp_level1': params.tp_level1,
                    'tp_level2': params.tp_level2,
                    'tp_level3': params.tp_level3,
                    'rsi_entry_threshold': params.rsi_entry_threshold,
                    'rsi_safety_threshold': params.rsi_safety_threshold,
                    'rsi_exit_threshold': params.rsi_exit_threshold,
                    'fees': params.fees
                }

            # Update progress bar description with best results
            self.trial_count += 1
            if self.progress_bar:
                desc = f"Best: APY={self.best_apy:.1f}% DD={self.best_drawdown:.1f}% Fitness={self.best_fitness:.1f}"
                self.progress_bar.set_description(desc)
                self.progress_bar.update(1)

            return fitness
        except Exception as e:
            if self.progress_bar:
                self.progress_bar.update(1)
            print(f"\nError in trial {self.trial_count}: {e}")
            return -1000  # Heavy penalty for failed trials

    def optimize(self, n_trials: int = 100, n_jobs: int = 1) -> Dict:
        """Run optimization with progress tracking"""
        print(f"Starting optimization with {n_trials} trials...")
        print("Optimizing for: 60% APY + 40% Drawdown reduction")
        print("=" * 80)

        # Create progress bar
        self.progress_bar = tqdm(
            total=n_trials,
            desc="Initializing...",
            bar_format='{desc}\nProgress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} trials [{elapsed}<{remaining}]',
            ncols=80
        )

        # Set up optuna with reduced logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction='maximize')

        try:
            # Use parallel processing if requested
            if n_jobs == 1:
                study.optimize(self.objective, n_trials=n_trials)
            else:
                study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        finally:
            self.progress_bar.close()

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"Best Fitness Score: {self.best_fitness:.2f}")
        print(f"Best APY: {self.best_apy:.2f}%")
        print(f"Best Max Drawdown: {self.best_drawdown:.2f}%")
        print(f"Trials Completed: {len(study.trials)}")

        print("\nBest Parameters:")
        print("-" * 40)
        for key, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.3f}")
            else:
                print(f"  {key:20s}: {value}")

        # Show top 5 trials
        print(f"\nTop 5 Trials:")
        print("-" * 40)
        sorted_trials = sorted(study.trials, key=lambda x: x.value if x.value else -1000, reverse=True)
        for i, trial in enumerate(sorted_trials[:5]):
            if trial.value:
                print(f"  #{i + 1}: Fitness={trial.value:.2f}")

        return self.best_params, self.best_fitness


class Visualizer:
    """Create visualizations and reports"""

    @staticmethod
    def plot_results(balance_history: List[Tuple[datetime, float]],
                     trades: List[Trade], coin: str, save_path: str):
        """Create balance and trades visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Balance chart
        times, balances = zip(*balance_history)
        ax1.plot(times, balances, 'b-', linewidth=2, label='Portfolio Value')

        # Add trade markers
        buy_times = [t.timestamp for t in trades if t.action == 'buy']
        buy_balances = []
        sell_times = [t.timestamp for t in trades if t.action == 'sell']
        sell_balances = []

        for trade in trades:
            # Find corresponding balance
            balance_idx = next((i for i, (t, _) in enumerate(balance_history)
                                if t >= trade.timestamp), -1)
            if balance_idx >= 0:
                if trade.action == 'buy':
                    buy_balances.append(balance_history[balance_idx][1])
                else:
                    sell_balances.append(balance_history[balance_idx][1])

        if buy_times and buy_balances:
            ax1.scatter(buy_times, buy_balances, color='green', marker='^',
                        s=50, alpha=0.7, label='Buys')
        if sell_times and sell_balances:
            ax1.scatter(sell_times, sell_balances, color='red', marker='v',
                        s=50, alpha=0.7, label='Sells')

        ax1.set_title(f'{coin} - Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('USDT Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Trade count over time
        daily_trades = {}
        for trade in trades:
            date = trade.timestamp.date()
            daily_trades[date] = daily_trades.get(date, 0) + 1

        if daily_trades:
            dates, counts = zip(*sorted(daily_trades.items()))
            ax2.bar(dates, counts, alpha=0.7, color='orange')
            ax2.set_title('Daily Trade Count')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Trades')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def save_trades_log(trades: List[Trade], save_path: str):
        """Save detailed trades log to CSV"""
        trades_data = []
        for trade in trades:
            trades_data.append({
                'timestamp': trade.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'action': trade.action,
                'amount_coin': round(trade.amount_coin, 6),
                'price': round(trade.price, 6),
                'usdt_amount': round(trade.usdt_amount, 2),
                'average_entry': round(trade.average_entry, 6),
                'position_size': round(trade.position_size, 6),
                'reason': trade.reason
            })

        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(save_path, index=False)

    @staticmethod
    def save_results(results: Dict, save_path: str):
        """Save results to JSON"""
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='DCA Strategy Backtester')
    parser.add_argument('--data_path', required=True, help='Path to CSV data file')
    parser.add_argument('--coin', required=True, help='Coin symbol (e.g., BTCUSDT)')
    parser.add_argument('--initial_balance', type=float, default=10000,
                        help='Initial balance in USDT')
    parser.add_argument('--optimize', action='store_true',
                        help='Run parameter optimization')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of parallel jobs for optimization (1=sequential)')
    parser.add_argument('--output_dir', default='./results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading data from {args.data_path}...")
    try:
        data = DataProcessor.load_data(args.data_path)
        print(f"Loaded {len(data)} data points from {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Initialize backtester
    backtester = Backtester(data, args.initial_balance)

    if args.optimize:
        # Run optimization
        print(f"Data loaded successfully! Starting optimization...")
        print(
            f"Dataset: {len(data)} rows from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Timespan: {(data.index[-1] - data.index[0]).days} days")
        print()

        optimizer = Optimizer(backtester)
        best_params, best_fitness = optimizer.optimize(args.trials, args.jobs)

        # Convert to StrategyParams object
        strategy_params = StrategyParams(**best_params)
    else:
        # Use default parameters
        strategy_params = StrategyParams()

    print("Running final simulation with best parameters...")
    apy, max_drawdown, balance_history, trades = backtester.simulate_strategy(strategy_params)

    # Calculate additional metrics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.action == 'sell' and
                          t.price > t.average_entry])
    win_rate = (winning_trades / max(1, len([t for t in trades if t.action == 'sell']))) * 100

    # Prepare results
    results = {
        'coin': args.coin,
        'initial_balance': args.initial_balance,
        'final_balance': balance_history[-1][1] if balance_history else args.initial_balance,
        'apy': round(apy, 2),
        'max_drawdown': round(max_drawdown, 2),
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'parameters': {
            'base_percent': strategy_params.base_percent,
            'initial_deviation': strategy_params.initial_deviation,
            'step_multiplier': strategy_params.step_multiplier,
            'volume_multiplier': strategy_params.volume_multiplier,
            'max_safeties': strategy_params.max_safeties,
            'trailing_deviation': strategy_params.trailing_deviation,
            'tp_level1': strategy_params.tp_level1,
            'tp_level2': strategy_params.tp_level2,
            'tp_level3': strategy_params.tp_level3,
            'rsi_entry_threshold': strategy_params.rsi_entry_threshold,
            'rsi_safety_threshold': strategy_params.rsi_safety_threshold,
            'rsi_exit_threshold': strategy_params.rsi_exit_threshold,
            'fees': strategy_params.fees
        },
        'data_period': {
            'start': data.index[0].isoformat(),
            'end': data.index[-1].isoformat(),
            'total_days': (data.index[-1] - data.index[0]).days
        }
    }

    # Print results
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS FOR {args.coin}")
    print("=" * 60)
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Final Balance: ${results['final_balance']:,.2f}")
    print(f"APY: {apy:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Data Period: {results['data_period']['start']} to {results['data_period']['end']}")
    print("=" * 60)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{args.coin}_{timestamp}"

    # Save results JSON
    results_path = output_dir / f"{base_filename}_results.json"
    Visualizer.save_results(results, str(results_path))
    print(f"Results saved to: {results_path}")

    # Save trades log
    trades_path = output_dir / f"{base_filename}_trades.csv"
    Visualizer.save_trades_log(trades, str(trades_path))
    print(f"Trades log saved to: {trades_path}")

    # Create and save visualization
    chart_path = output_dir / f"{base_filename}_chart.png"
    Visualizer.plot_results(balance_history, trades, args.coin, str(chart_path))
    print(f"Chart saved to: {chart_path}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()