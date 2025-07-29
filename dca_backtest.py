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
    print("Please install ta: pip install ta")
    exit(1)

try:
    import optuna
except ImportError:
    print("Please install optuna: pip install optuna")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    exit(1)

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("No GPU detected, using CPU only")
except ImportError:
    print("Please install PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    exit(1)


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
    base_percent: float = 1.0              # Base order as % of balance (constant)
    initial_deviation: float = 3.0          # Initial safety order deviation % (optimizable)
    step_multiplier: float = 1.5            # Geometric step multiplier (CONSTANT)
    volume_multiplier: float = 1.2          # Volume scaling multiplier (CONSTANT)
    max_safeties: int = 8                   # Maximum safety orders (CONSTANT)
    trailing_deviation: float = 7.0         # Trailing stop % (optimizable)
    tp_level1: float = 5.0                  # First TP level % (optimizable)
    tp_level2: float = 10.0                 # Second TP level % (optimizable)
    tp_level3: float = 15.0                 # Third TP level % (optimizable)
    tp_percent1: float = 50.0               # % to sell at TP1 (optimizable)
    tp_percent2: float = 30.0               # % to sell at TP2 (optimizable)
    tp_percent3: float = 20.0               # % to sell at TP3 (optimizable)
    rsi_entry_threshold: float = 30.0       # RSI entry threshold (CONSTANT)
    rsi_safety_threshold: float = 30.0      # RSI safety threshold (CONSTANT)
    rsi_exit_threshold: float = 70.0        # RSI exit threshold (CONSTANT)
    fees: float = 0.075                     # Trading fees % (CONSTANT - 0.075% realistic)


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
        print(f"Calculating technical indicators for {len(df)} data points...")
        indicators = {}

        # Resample to different timeframes
        df_1h = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        }).dropna()
        print(f"  1H data: {len(df_1h)} candles")

        df_4h = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'vol': 'sum'
        }).dropna()
        print(f"  4H data: {len(df_4h)} candles")

        # Calculate indicators
        print("  Computing RSI indicators...")
        try:
            # RSI indicators
            indicators['rsi_1h'] = ta.momentum.RSIIndicator(df_1h['close'], window=14).rsi()
            indicators['rsi_4h'] = ta.momentum.RSIIndicator(df_4h['close'], window=14).rsi()

            print("  Computing SMA indicators...")
            # SMA indicators
            indicators['sma_fast_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=12).sma_indicator()
            indicators['sma_slow_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=26).sma_indicator()
            indicators['sma_50_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=50).sma_indicator()
            print("  ✓ Technical indicators calculated successfully")

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            print("  Using fallback calculations...")
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


class GPUBatchSimulator:
    """GPU-accelerated batch strategy simulation"""

    def __init__(self, data: pd.DataFrame, indicators: Dict, initial_balance: float = 10000):
        print("Initializing GPU Batch Simulator...")
        self.device = torch.device('cuda' if GPU_AVAILABLE else 'cpu')
        self.initial_balance = initial_balance

        print(f"  Converting {len(data)} price points to GPU tensors...")
        # Convert data to tensors
        self.prices = torch.tensor(data['close'].values, dtype=torch.float32, device=self.device)
        self.timestamps = data.index.values

        print("  Preparing indicator tensors with forward fill...")
        # Prepare indicator tensors (with forward fill)
        self.rsi_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('rsi_1h', pd.Series()))
        print("    ✓ RSI 1H tensor ready")
        self.rsi_4h_values = self._prepare_indicator_tensor(data.index, indicators.get('rsi_4h', pd.Series()))
        print("    ✓ RSI 4H tensor ready")
        self.sma_fast_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('sma_fast_1h', pd.Series()))
        print("    ✓ SMA Fast 1H tensor ready")
        self.sma_slow_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('sma_slow_1h', pd.Series()))
        print("    ✓ SMA Slow 1H tensor ready")
        self.sma_50_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('sma_50_1h', pd.Series()))
        print("    ✓ SMA 50 1H tensor ready")

        print("  Calculating optimal batch size for GPU memory...")
        # Memory management
        self.max_batch_size = self._calculate_max_batch_size()
        print(f"✓ GPU Batch Simulator initialized on {self.device}")
        print(f"✓ Max batch size: {self.max_batch_size} (to avoid OOM)")

    def _calculate_max_batch_size(self) -> int:
        """Calculate maximum batch size to avoid GPU OOM"""
        if not GPU_AVAILABLE:
            return 32  # Conservative CPU batch size

        try:
            # Get available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = gpu_memory * 0.7  # Use 70% to be safe

            # Estimate memory per simulation (rough calculation)
            data_points = len(self.prices)
            memory_per_sim = data_points * 4 * 20  # 20 float32 arrays per simulation

            max_batch = int(available_memory / memory_per_sim)

            # Cap between reasonable bounds
            return max(8, min(max_batch, 512))
        except:
            return 32  # Safe fallback

    def _prepare_indicator_tensor(self, timestamps: pd.DatetimeIndex, indicator_series: pd.Series) -> torch.Tensor:
        """Convert indicator series to forward-filled tensor"""
        try:
            if len(indicator_series) == 0:
                return torch.full((len(timestamps),), 50.0, dtype=torch.float32, device=self.device)

            # Forward fill indicator values to match data timestamps
            aligned_values = []
            last_value = 50.0

            print(f"      Processing {len(timestamps)} timestamps...")

            for i, ts in enumerate(timestamps):
                if i % 20000 == 0:  # Progress every 200k points
                    print(f"        Progress: {i}/{len(timestamps)} ({i / len(timestamps) * 100:.1f}%)")

                # Find most recent indicator value
                valid_indices = indicator_series.index <= ts
                if valid_indices.any():
                    recent_idx = indicator_series.index[valid_indices][-1]
                    last_value = indicator_series.loc[recent_idx]
                aligned_values.append(last_value)

            return torch.tensor(aligned_values, dtype=torch.float32, device=self.device)
        except:
            return torch.full((len(timestamps),), 50.0, dtype=torch.float32, device=self.device)

    def simulate_batch(self, param_batch: List[StrategyParams]) -> List[Tuple[float, float]]:
        """Simulate multiple parameter sets in parallel on GPU"""
        if not param_batch:
            return []

        batch_size = len(param_batch)
        data_length = len(self.prices)

        try:
            # Initialize batch state tensors
            balances = torch.full((batch_size,), self.initial_balance, dtype=torch.float32, device=self.device)
            position_sizes = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            average_entries = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            total_spent = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            active_deals = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            safety_counts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            last_entry_prices = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            peak_prices = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            trailing_active = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # TP level tracking
            tp1_hit = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            tp2_hit = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            tp3_hit = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            # Extract parameters as tensors
            base_percents = torch.tensor([p.base_percent for p in param_batch], dtype=torch.float32, device=self.device)
            initial_deviations = torch.tensor([p.initial_deviation for p in param_batch], dtype=torch.float32,
                                              device=self.device)
            step_multipliers = torch.tensor([p.step_multiplier for p in param_batch], dtype=torch.float32,
                                            device=self.device)
            volume_multipliers = torch.tensor([p.volume_multiplier for p in param_batch], dtype=torch.float32,
                                              device=self.device)
            max_safeties = torch.tensor([p.max_safeties for p in param_batch], dtype=torch.int32, device=self.device)
            trailing_deviations = torch.tensor([p.trailing_deviation for p in param_batch], dtype=torch.float32,
                                               device=self.device)
            tp_level1s = torch.tensor([p.tp_level1 for p in param_batch], dtype=torch.float32, device=self.device)
            tp_level2s = torch.tensor([p.tp_level2 for p in param_batch], dtype=torch.float32, device=self.device)
            tp_level3s = torch.tensor([p.tp_level3 for p in param_batch], dtype=torch.float32, device=self.device)
            tp_percent1s = torch.tensor([p.tp_percent1 for p in param_batch], dtype=torch.float32, device=self.device)
            tp_percent2s = torch.tensor([p.tp_percent2 for p in param_batch], dtype=torch.float32, device=self.device)
            tp_percent3s = torch.tensor([p.tp_percent3 for p in param_batch], dtype=torch.float32, device=self.device)
            rsi_entry_thresholds = torch.tensor([p.rsi_entry_threshold for p in param_batch], dtype=torch.float32,
                                                device=self.device)
            rsi_safety_thresholds = torch.tensor([p.rsi_safety_threshold for p in param_batch], dtype=torch.float32,
                                                 device=self.device)
            rsi_exit_thresholds = torch.tensor([p.rsi_exit_threshold for p in param_batch], dtype=torch.float32,
                                               device=self.device)
            fees = torch.tensor([p.fees for p in param_batch], dtype=torch.float32, device=self.device)

            # Track balance history for drawdown calculation
            balance_history = torch.zeros((batch_size, data_length), dtype=torch.float32, device=self.device)

            # Main simulation loop
            for i in range(data_length):
                current_price = self.prices[i]
                current_rsi_1h = self.rsi_1h_values[i]
                current_rsi_4h = self.rsi_4h_values[i]
                current_sma_fast = self.sma_fast_1h_values[i]
                current_sma_slow = self.sma_slow_1h_values[i]
                current_sma_50 = self.sma_50_1h_values[i]

                # Entry conditions
                rsi_entry_ok = current_rsi_4h < rsi_entry_thresholds
                sma_ok = current_sma_fast > current_sma_slow
                momentum_ok = current_price > current_sma_50
                can_enter = ~active_deals & rsi_entry_ok & sma_ok & momentum_ok

                # Execute base orders
                base_amounts = balances * (base_percents / 100.0)
                coin_amounts = base_amounts / current_price

                # Apply entry logic
                entering = can_enter & (base_amounts > 1.0)  # Minimum order size
                balances = torch.where(entering, balances - base_amounts, balances)
                position_sizes = torch.where(entering, coin_amounts, position_sizes)
                total_spent = torch.where(entering, base_amounts, total_spent)
                average_entries = torch.where(entering, current_price, average_entries)
                last_entry_prices = torch.where(entering, current_price, last_entry_prices)
                active_deals = torch.where(entering, True, active_deals)
                safety_counts = torch.where(entering, 0, safety_counts)
                peak_prices = torch.where(entering, current_price, peak_prices)
                tp1_hit = torch.where(entering, False, tp1_hit)
                tp2_hit = torch.where(entering, False, tp2_hit)
                tp3_hit = torch.where(entering, False, tp3_hit)

                # Safety order logic (simplified for GPU)
                safety_deviations = initial_deviations * (step_multipliers ** safety_counts.float())
                safety_thresholds = last_entry_prices * (1.0 - safety_deviations / 100.0)
                safety_rsi_ok = current_rsi_1h < rsi_safety_thresholds
                can_safety = active_deals & (current_price <= safety_thresholds) & safety_rsi_ok & (
                            safety_counts < max_safeties)

                # Execute safety orders
                safety_multipliers = volume_multipliers ** safety_counts.float()
                safety_base = self.initial_balance * (base_percents / 100.0)
                safety_amounts = safety_base * safety_multipliers
                safety_amounts = torch.min(safety_amounts, balances * 0.95)  # Don't exceed balance
                safety_coins = safety_amounts / current_price

                executing_safety = can_safety & (safety_amounts > 1.0)
                balances = torch.where(executing_safety, balances - safety_amounts, balances)
                position_sizes = torch.where(executing_safety, position_sizes + safety_coins, position_sizes)
                total_spent = torch.where(executing_safety, total_spent + safety_amounts, total_spent)
                average_entries = torch.where(executing_safety, total_spent / position_sizes, average_entries)
                last_entry_prices = torch.where(executing_safety, current_price, last_entry_prices)
                safety_counts = torch.where(executing_safety, safety_counts + 1, safety_counts)

                # Take profit logic
                profit_pcts = (current_price - average_entries) / average_entries * 100.0

                # TP1
                tp1_trigger = active_deals & ~tp1_hit & (profit_pcts >= tp_level1s)
                tp1_sell_amounts = position_sizes * (tp_percent1s / 100.0)
                tp1_usdt = tp1_sell_amounts * current_price
                balances = torch.where(tp1_trigger, balances + tp1_usdt, balances)
                position_sizes = torch.where(tp1_trigger, position_sizes - tp1_sell_amounts, position_sizes)
                tp1_hit = torch.where(tp1_trigger, True, tp1_hit)
                trailing_active = torch.where(tp1_trigger, True, trailing_active)
                peak_prices = torch.where(tp1_trigger, current_price, peak_prices)

                # TP2
                tp2_trigger = active_deals & ~tp2_hit & (profit_pcts >= tp_level2s)
                tp2_sell_amounts = position_sizes * (tp_percent2s / 100.0)
                tp2_usdt = tp2_sell_amounts * current_price
                balances = torch.where(tp2_trigger, balances + tp2_usdt, balances)
                position_sizes = torch.where(tp2_trigger, position_sizes - tp2_sell_amounts, position_sizes)
                tp2_hit = torch.where(tp2_trigger, True, tp2_hit)

                # TP3
                tp3_trigger = active_deals & ~tp3_hit & (profit_pcts >= tp_level3s)
                tp3_sell_amounts = position_sizes * (tp_percent3s / 100.0)
                tp3_usdt = tp3_sell_amounts * current_price
                balances = torch.where(tp3_trigger, balances + tp3_usdt, balances)
                position_sizes = torch.where(tp3_trigger, position_sizes - tp3_sell_amounts, position_sizes)
                tp3_hit = torch.where(tp3_trigger, True, tp3_hit)

                # Update peak prices for trailing
                peak_prices = torch.where(trailing_active & (current_price > peak_prices), current_price, peak_prices)

                # Trailing stop
                trailing_thresholds = peak_prices * (1.0 - trailing_deviations / 100.0)
                trailing_trigger = trailing_active & (current_price <= trailing_thresholds)

                # Force exit conditions
                force_exit = active_deals & (current_rsi_4h > rsi_exit_thresholds)

                # Execute full exits
                exit_trigger = trailing_trigger | force_exit
                exit_usdt = position_sizes * current_price
                balances = torch.where(exit_trigger, balances + exit_usdt, balances)
                position_sizes = torch.where(exit_trigger, 0.0, position_sizes)
                active_deals = torch.where(exit_trigger, False, active_deals)
                trailing_active = torch.where(exit_trigger, False, trailing_active)

                # Close deals with zero position
                zero_position = active_deals & (position_sizes < 0.0001)
                active_deals = torch.where(zero_position, False, active_deals)

                # Record total portfolio values
                portfolio_values = balances + position_sizes * current_price
                balance_history[:, i] = portfolio_values

            # Final exit for any remaining positions
            final_exit_usdt = position_sizes * self.prices[-1]
            final_balances = balances + final_exit_usdt

            # Calculate metrics
            results = []
            for b in range(batch_size):
                final_balance = final_balances[b].item()
                balance_series = balance_history[b, :].cpu().numpy()

                # Calculate APY
                days = (self.timestamps[-1] - self.timestamps[0]) / np.timedelta64(1, 'D')
                years = days / 365.25
                apy = (pow(final_balance / self.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

                # Calculate max drawdown
                peak = balance_series[0]
                max_dd = 0.0
                for balance in balance_series:
                    if balance > peak:
                        peak = balance
                    drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, drawdown)

                results.append((apy, max_dd))

            return results

        except torch.cuda.OutOfMemoryError:
            print(f"GPU OOM with batch size {batch_size}, falling back to CPU")
            torch.cuda.empty_cache()
            return self._simulate_batch_cpu(param_batch)
        except Exception as e:
            print(f"GPU simulation error: {e}, falling back to CPU")
            if GPU_AVAILABLE:
                torch.cuda.empty_cache()
            return self._simulate_batch_cpu(param_batch)

    def _simulate_batch_cpu(self, param_batch: List[StrategyParams]) -> List[Tuple[float, float]]:
        """Fallback CPU simulation for failed GPU runs"""
        results = []
        for params in param_batch:
            try:
                # Use the original backtester for individual simulation
                from copy import deepcopy
                data_df = pd.DataFrame({
                    'close': self.prices.cpu().numpy()
                }, index=pd.to_datetime(self.timestamps))

                # Create a simple backtester instance
                simple_backtester = Backtester(data_df, self.initial_balance)
                apy, max_drawdown, _, _ = simple_backtester.simulate_strategy(params)
                results.append((apy, max_drawdown))
            except:
                results.append((0.0, 100.0))  # Failed simulation
        return results


class Optimizer:
    """Strategy parameter optimization with GPU acceleration"""

    def __init__(self, backtester: Backtester, use_gpu: bool = True):
        self.backtester = backtester
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.best_fitness = -1000
        self.best_apy = 0
        self.best_drawdown = 100
        self.best_params = {}
        self.trial_count = 0
        self.progress_bar = None

        # Initialize GPU simulator if available
        if self.use_gpu:
            try:
                print("Setting up GPU acceleration...")
                self.gpu_simulator = GPUBatchSimulator(
                    backtester.data,
                    backtester.indicators,
                    backtester.initial_balance
                )
                print(f"✓ GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"✗ GPU initialization failed: {e}")
                self.use_gpu = False

    def objective(self, trial):
        """Optuna objective function with discrete values"""
        params = StrategyParams(
            # CONSTANT PARAMETERS (not optimized)
            base_percent=1.0,
            step_multiplier=1.5,
            volume_multiplier=1.2,
            max_safeties=8,
            rsi_entry_threshold=30.0,
            rsi_safety_threshold=30.0,
            rsi_exit_threshold=70.0,
            fees=0.075,

            # OPTIMIZABLE PARAMETERS (discrete values)
            initial_deviation=trial.suggest_categorical('initial_deviation',
                                                        [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
                                                         3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8,
                                                         4.9, 5.0]),
            trailing_deviation=trial.suggest_categorical('trailing_deviation', [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            tp_level1=trial.suggest_categorical('tp_level1', [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            tp_level2=trial.suggest_categorical('tp_level2', [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
            tp_level3=trial.suggest_categorical('tp_level3', [15.0, 17.0, 20.0, 22.0, 25.0]),
            tp_percent1=trial.suggest_categorical('tp_percent1', [30.0, 40.0, 50.0, 60.0, 70.0]),
            tp_percent2=trial.suggest_categorical('tp_percent2', [20.0, 25.0, 30.0, 35.0, 40.0]),
            tp_percent3=trial.suggest_categorical('tp_percent3', [10.0, 15.0, 20.0, 25.0, 30.0])
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
                    # Only track optimizable parameters
                    'initial_deviation': params.initial_deviation,
                    'trailing_deviation': params.trailing_deviation,
                    'tp_level1': params.tp_level1,
                    'tp_level2': params.tp_level2,
                    'tp_level3': params.tp_level3,
                    'tp_percent1': params.tp_percent1,
                    'tp_percent2': params.tp_percent2,
                    'tp_percent3': params.tp_percent3,

                    # Include constants for reference
                    'base_percent': 1.0,
                    'step_multiplier': 1.5,
                    'volume_multiplier': 1.2,
                    'max_safeties': 8,
                    'rsi_entry_threshold': 30.0,
                    'rsi_safety_threshold': 30.0,
                    'rsi_exit_threshold': 70.0,
                    'fees': 0.075
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

    def optimize_gpu_batch(self, n_trials: int = 100) -> Dict:
        """GPU-accelerated batch optimization"""
        if not self.use_gpu:
            return self.optimize(n_trials, 1)  # Fall back to CPU

        print(f"Starting GPU batch optimization with {n_trials} trials...")
        print("Optimizing for: 60% APY + 40% Drawdown reduction")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {self.gpu_simulator.max_batch_size}")
        print("=" * 80)

        # Generate all parameter combinations
        all_params = []
        for _ in range(n_trials):
            params = StrategyParams(
                # CONSTANT PARAMETERS
                base_percent=1.0,
                step_multiplier=1.5,
                volume_multiplier=1.2,
                max_safeties=8,
                rsi_entry_threshold=30.0,
                rsi_safety_threshold=30.0,
                rsi_exit_threshold=70.0,
                fees=0.075,

                # OPTIMIZABLE PARAMETERS (discrete random selection)
                initial_deviation=np.random.choice(
                    [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4,
                     4.5, 4.6, 4.7, 4.8, 4.9, 5.0]),
                trailing_deviation=np.random.choice([5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
                tp_level1=np.random.choice([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                tp_level2=np.random.choice([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]),
                tp_level3=np.random.choice([15.0, 17.0, 20.0, 22.0, 25.0]),
                tp_percent1=np.random.choice([30.0, 40.0, 50.0, 60.0, 70.0]),
                tp_percent2=np.random.choice([20.0, 25.0, 30.0, 35.0, 40.0]),
                tp_percent3=np.random.choice([10.0, 15.0, 20.0, 25.0, 30.0])
            )
            all_params.append(params)

        # Process in batches
        batch_size = self.gpu_simulator.max_batch_size
        all_results = []

        self.progress_bar = tqdm(
            total=n_trials,
            desc="Initializing GPU batches...",
            bar_format='{desc}\nProgress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} trials [{elapsed}<{remaining}]',
            ncols=80
        )

        try:
            for i in range(0, n_trials, batch_size):
                batch_params = all_params[i:i + batch_size]

                # GPU batch simulation with memory management
                try:
                    if GPU_AVAILABLE:
                        torch.cuda.empty_cache()  # Clear cache before each batch

                    batch_results = self.gpu_simulator.simulate_batch(batch_params)

                    # Update best results
                    for j, (apy, max_drawdown) in enumerate(batch_results):
                        fitness = 0.6 * apy - 0.4 * max_drawdown

                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_apy = apy
                            self.best_drawdown = max_drawdown
                            param_idx = i + j
                            self.best_params = {
                                # Only track optimizable parameters
                                'initial_deviation': params.initial_deviation,
                                'trailing_deviation': params.trailing_deviation,
                                'tp_level1': params.tp_level1,
                                'tp_level2': params.tp_level2,
                                'tp_level3': params.tp_level3,
                                'tp_percent1': params.tp_percent1,
                                'tp_percent2': params.tp_percent2,
                                'tp_percent3': params.tp_percent3,

                                # Include constants for reference
                                'base_percent': 1.0,
                                'step_multiplier': 1.5,
                                'volume_multiplier': 1.2,
                                'max_safeties': 8,
                                'rsi_entry_threshold': 30.0,
                                'rsi_safety_threshold': 30.0,
                                'rsi_exit_threshold': 70.0,
                                'fees': 0.075
                            }

                    all_results.extend(batch_results)

                    # Update progress
                    desc = f"Best: APY={self.best_apy:.1f}% DD={self.best_drawdown:.1f}% Fitness={self.best_fitness:.1f}"
                    self.progress_bar.set_description(desc)
                    self.progress_bar.update(len(batch_params))

                except torch.cuda.OutOfMemoryError:
                    print(f"\nGPU OOM at batch {i // batch_size + 1}, reducing batch size...")
                    # Reduce batch size and retry
                    self.gpu_simulator.max_batch_size = max(4, self.gpu_simulator.max_batch_size // 2)
                    batch_size = self.gpu_simulator.max_batch_size
                    torch.cuda.empty_cache()

                    # Retry with smaller batch
                    batch_results = self.gpu_simulator.simulate_batch(batch_params[:batch_size])
                    all_results.extend(batch_results)
                    self.progress_bar.update(batch_size)

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
        finally:
            self.progress_bar.close()
            if GPU_AVAILABLE:
                torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print("GPU BATCH OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"Best Fitness Score: {self.best_fitness:.2f}")
        print(f"Best APY: {self.best_apy:.2f}%")
        print(f"Best Max Drawdown: {self.best_drawdown:.2f}%")
        print(f"Trials Completed: {len(all_results)}")

        print("\nBest Parameters:")
        print("-" * 40)
        for key, value in self.best_params.items():
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.3f}")
            else:
                print(f"  {key:20s}: {value}")

        return self.best_params, self.best_fitness

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
    parser.add_argument('--gpu_batch', action='store_true',
                        help='Use GPU batch optimization (much faster if GPU available)')
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

        optimizer = Optimizer(backtester, use_gpu=args.gpu_batch)

        if args.gpu_batch and GPU_AVAILABLE:
            best_params, best_fitness = optimizer.optimize_gpu_batch(args.trials)
        else:
            if args.gpu_batch and not GPU_AVAILABLE:
                print("GPU requested but not available, falling back to CPU")
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
