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
import signal
from contextlib import contextmanager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@contextmanager
def timeout(duration):
    """Windows-compatible timeout using threading"""
    import threading
    import time

    class TimeoutError(Exception):
        pass

    def timeout_handler():
        raise TimeoutError(f"Operation timed out after {duration} seconds")

    timer = threading.Timer(duration, timeout_handler)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
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
    base_percent: float = 1.0  # Base order as % of balance (constant)
    initial_deviation: float = 3.0  # Initial safety order deviation % (optimizable)
    step_multiplier: float = 1.5  # Geometric step multiplier (CONSTANT)
    volume_multiplier: float = 1.2  # Volume scaling multiplier (CONSTANT)
    max_safeties: int = 8  # Maximum safety orders (CONSTANT)
    trailing_deviation: float = 3.0  # Trailing stop % - NOW LIMITED TO TP1 MAX
    tp_level1: float = 3.0  # First TP level % (optimizable 1-5)
    # tp_level2 and tp_level3 are now calculated dynamically
    tp_percent1: float = 50.0  # % to sell at TP1 (optimizable)
    tp_percent2: float = 30.0  # % to sell at TP2 (optimizable)
    tp_percent3: float = 20.0  # % to sell at TP3 (optimizable)
    rsi_entry_threshold: float = 40.0  # RSI entry threshold - 1H RSI < 40 (CONSTANT)
    rsi_safety_threshold: float = 30.0  # RSI safety threshold (CONSTANT)
    rsi_exit_threshold: float = 70.0  # RSI exit threshold (CONSTANT)
    fees: float = 0.075  # Trading fees % (CONSTANT - 0.075% realistic)

    def __post_init__(self):
        """Ensure trailing deviation doesn't exceed TP1"""
        if self.trailing_deviation > self.tp_level1:
            self.trailing_deviation = self.tp_level1

    @property
    def tp_level2(self) -> float:
        """TP2 = TP1 × 2"""
        return self.tp_level1 * 2

    @property
    def tp_level3(self) -> float:
        """TP3 = TP1 × 3"""
        return self.tp_level1 * 3


class DCAStrategy:
    """Main DCA strategy implementation"""

    def execute_partial_trailing_exit(self, row: pd.Series, amount_to_sell: float):
        """Execute partial trailing stop - only TP1 amount"""
        if amount_to_sell > self.position_size:
            amount_to_sell = self.position_size

        usdt_received = amount_to_sell * row['close']

        # Apply fees
        fee_amount = usdt_received * (self.params.fees / 100)
        net_usdt = usdt_received - fee_amount

        self.balance += net_usdt
        self.position_size -= amount_to_sell

        # Disable trailing after execution
        self.trailing_active = False

        self.add_trade(row.name, 'sell', amount_to_sell, row['close'],
                       usdt_received, 'trailing_stop_partial')

        # Close deal if position is very small
        if self.position_size < 0.0001:
            self.close_deal(row.name)

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
        """Check if entry conditions are met - REVISED FOR HIGHER FREQUENCY"""
        try:
            # NEW CONDITIONS:
            # 1. RSI(14) < 40 on 1H (instead of RSI < 30 on 4H)
            rsi_1h = indicators.get('rsi_1h', 50)
            rsi_condition = rsi_1h < self.params.rsi_entry_threshold  # Now 40 on 1H

            # 2. Fast SMA(12) > Slow SMA(26) on 1H (unchanged - bullish bias)
            sma_fast_1h = indicators.get('sma_fast_1h', row['close'])
            sma_slow_1h = indicators.get('sma_slow_1h', row['close'])
            sma_condition = sma_fast_1h > sma_slow_1h

            return rsi_condition and sma_condition
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
        """Check trailing stop condition - deviation limited to TP1 max"""
        if not self.trailing_active or self.position_size <= 0:
            return False

        # Update peak price
        if row['close'] > self.peak_price:
            self.peak_price = row['close']

        # Limit trailing deviation to TP1 level maximum
        effective_trailing_deviation = min(self.params.trailing_deviation, self.params.tp_level1)

        # Check trailing stop with limited deviation
        trailing_threshold = self.peak_price * (1 - effective_trailing_deviation / 100)
        return row['close'] <= trailing_threshold


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
            # RSI indicators - NOW ONLY 1H NEEDED FOR ENTRY
            indicators['rsi_1h'] = ta.momentum.RSIIndicator(df_1h['close'], window=14).rsi()
            indicators['rsi_4h'] = ta.momentum.RSIIndicator(df_4h['close'], window=14).rsi()  # Still needed for exit

            print("  Computing SMA indicators...")
            # SMA indicators for trend confirmation
            indicators['sma_fast_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=12).sma_indicator()
            indicators['sma_slow_1h'] = ta.trend.SMAIndicator(df_1h['close'], window=26).sma_indicator()

            # DEBUG: Check for NaN values
            for name, series in indicators.items():
                nan_count = series.isna().sum()
                valid_count = len(series) - nan_count
                if valid_count > 0:
                    first_valid = series.dropna().iloc[0]
                    print(f"    {name}: {valid_count} valid, {nan_count} NaN, first_valid: {first_valid:.4f}")
                else:
                    print(f"    {name}: {valid_count} valid, {nan_count} NaN, NO VALID VALUES")

            print("  ✓ Technical indicators calculated successfully")

        except Exception as e:
            print(f"Error calculating indicators: {e}")
            print("  Using fallback calculations...")
            # Fallback to simple calculations
            indicators['rsi_1h'] = pd.Series(50, index=df_1h.index)
            indicators['rsi_4h'] = pd.Series(50, index=df_4h.index)
            indicators['sma_fast_1h'] = df_1h['close'].rolling(12).mean()
            indicators['sma_slow_1h'] = df_1h['close'].rolling(26).mean()

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
            # Get current indicator values - UPDATED FOR NEW CONDITIONS
            current_indicators = {
                'rsi_1h': self.get_indicator_value(timestamp, 'rsi_1h'),
                'rsi_4h': self.get_indicator_value(timestamp, 'rsi_4h'),  # Still needed for exit
                'sma_fast_1h': self.get_indicator_value(timestamp, 'sma_fast_1h'),
                'sma_slow_1h': self.get_indicator_value(timestamp, 'sma_slow_1h'),
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


            # Record balance history
            total_value = strategy.balance
            if strategy.position_size > 0:
                total_value += strategy.position_size * row['close']
            strategy.balance_history.append((timestamp, total_value))

        # # Close any remaining position
        # if strategy.active_deal:
        #     last_row = self.data.iloc[-1]
        #     strategy.execute_full_exit(last_row, 'end_of_data')

        # Calculate metrics
        # FIXED: Use total portfolio value (including open positions) for APY, not just cash balance
        final_balance = strategy.balance_history[-1][1] if strategy.balance_history else self.initial_balance
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
        self.rsi_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('rsi_1h', pd.Series()))
        print("    ✓ RSI 1H tensor ready")
        self.rsi_4h_values = self._prepare_indicator_tensor(data.index, indicators.get('rsi_4h', pd.Series()))
        print("    ✓ RSI 4H tensor ready")
        self.sma_fast_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('sma_fast_1h', pd.Series()))
        print("    ✓ SMA Fast 1H tensor ready")
        self.sma_slow_1h_values = self._prepare_indicator_tensor(data.index, indicators.get('sma_slow_1h', pd.Series()))
        print("    ✓ SMA Slow 1H tensor ready")

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
        """GPU-accelerated indicator tensor preparation with proper forward fill"""
        try:
            if len(indicator_series) == 0:
                return torch.full((len(timestamps),), 50.0, dtype=torch.float32, device=self.device)

            print(f"      GPU processing {len(timestamps)} timestamps...")

            # Remove NaN values from indicator series first
            clean_indicator_series = indicator_series.dropna()

            if len(clean_indicator_series) == 0:
                print(f"      Warning: No valid indicator values, using default 50.0")
                return torch.full((len(timestamps),), 50.0, dtype=torch.float32, device=self.device)

            # Convert to numpy arrays for processing
            timestamps_np = timestamps.values
            indicator_times_np = clean_indicator_series.index.values
            indicator_values_np = clean_indicator_series.values

            # Use pandas built-in reindex with forward fill for accuracy
            # This handles the complex timestamp alignment properly
            temp_series = pd.Series(indicator_values_np, index=indicator_times_np)

            # Reindex to match our timestamps with forward fill
            aligned_series = temp_series.reindex(timestamps, method='ffill')

            # Fill any remaining NaN values (leading ones) with 50.0
            # FIXED: Use 50.0 instead of first_valid to match CPU (prevents backfilling leading NaNs with first value)
            aligned_series = aligned_series.fillna(50.0)

            # Convert to tensor
            aligned_values = torch.tensor(aligned_series.values, dtype=torch.float32, device=self.device)

            print(
                f"      ✓ GPU tensor ready ({aligned_values.shape[0]} values, range: {aligned_values.min():.2f}-{aligned_values.max():.2f})")
            return aligned_values

        except Exception as e:
            print(f"      GPU tensor prep failed: {e}, using CPU fallback")
            return self._prepare_indicator_tensor_cpu_fallback(timestamps, indicator_series)

    def _prepare_indicator_tensor_cpu_fallback(self, timestamps: pd.DatetimeIndex,
                                               indicator_series: pd.Series) -> torch.Tensor:
        """CPU fallback for indicator tensor preparation"""
        try:
            if len(indicator_series) == 0:
                return torch.full((len(timestamps),), 50.0, dtype=torch.float32, device=self.device)

            aligned_values = []
            last_value = 50.0

            print(f"      CPU processing {len(timestamps)} timestamps...")

            for i, ts in enumerate(timestamps):
                if i % 10000 == 0:
                    print(f"        CPU Progress: {i}/{len(timestamps)} ({i / len(timestamps) * 100:.1f}%)")

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

            # FIXED: Add cooldown tracking (assume 1m data, use step index for minutes)
            last_close_step = torch.full((batch_size,), -999999, dtype=torch.int64, device=self.device)

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
            tp_level2s = torch.tensor([p.tp_level2 for p in param_batch], dtype=torch.float32, device=self.device)  # Auto-calculated
            tp_level3s = torch.tensor([p.tp_level3 for p in param_batch], dtype=torch.float32, device=self.device)  # Auto-calculated
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
            # FIXED: Precompute fee factor for buys/sells
            fee_factor = 1.0 - (fees / 100.0)

            # Track balance history for drawdown calculation
            balance_history = torch.zeros((batch_size, data_length), dtype=torch.float32, device=self.device)

            # Main simulation loop
            for i in range(data_length):
                try:
                    # Safely extract current values with bounds checking
                    if i >= len(self.prices):
                        break

                    current_price = self.prices[i]
                    current_rsi_1h = self.rsi_1h_values[i] if i < len(self.rsi_1h_values) else 50.0
                    current_rsi_4h = self.rsi_4h_values[i] if i < len(self.rsi_4h_values) else 50.0
                    current_sma_fast = self.sma_fast_1h_values[i] if i < len(self.sma_fast_1h_values) else current_price
                    current_sma_slow = self.sma_slow_1h_values[i] if i < len(self.sma_slow_1h_values) else current_price

                    # Add debug logging for first iteration
                    if i == 0:
                        print(f"    Debug first iteration - Price: {current_price:.4f}")
                        print(
                            f"    RSI1H: {current_rsi_1h:.1f}, SMA_fast: {current_sma_fast:.4f}, SMA_slow: {current_sma_slow:.4f}")
                        print(
                            f"    Entry conditions: RSI1H<40: {current_rsi_1h < 40.0}, SMA_cross: {current_sma_fast > current_sma_slow}")

                    if i % 50000 == 0 and i > 0:
                        # Aggressive memory cleanup for large datasets
                        if len(self.prices) > 1000000:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        print(f"    GPU simulation progress: {i}/{data_length} ({i / data_length * 100:.1f}%)")

                    # Entry conditions - SIMPLIFIED TO 2 CONDITIONS
                    rsi_entry_ok = current_rsi_1h < rsi_entry_thresholds  # RSI < 40 on 1H
                    sma_ok = current_sma_fast > current_sma_slow  # SMA cross bullish
                    # FIXED: Add cooldown check (5 min since last close)
                    in_cooldown = (i - last_close_step) < 5
                    can_enter = ~active_deals & rsi_entry_ok & sma_ok & ~in_cooldown  # REMOVED momentum_ok

                    # Execute base orders
                    base_amounts = balances * (base_percents / 100.0)
                    # FIXED: Apply fees to base buy
                    net_base = base_amounts * fee_factor
                    coin_amounts = net_base / current_price

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
                    # FIXED: Apply fees to safety buy
                    net_safety = safety_amounts * fee_factor
                    safety_coins = net_safety / current_price

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
                    # FIXED: Apply fees to TP sell
                    tp1_usdt = (tp1_sell_amounts * current_price) * fee_factor
                    balances = torch.where(tp1_trigger, balances + tp1_usdt, balances)
                    position_sizes = torch.where(tp1_trigger, position_sizes - tp1_sell_amounts, position_sizes)
                    tp1_hit = torch.where(tp1_trigger, True, tp1_hit)
                    trailing_active = torch.where(tp1_trigger, True, trailing_active)
                    peak_prices = torch.where(tp1_trigger, current_price, peak_prices)

                    # TP2
                    tp2_trigger = active_deals & ~tp2_hit & (profit_pcts >= tp_level2s)
                    tp2_sell_amounts = position_sizes * (tp_percent2s / 100.0)
                    # FIXED: Apply fees to TP sell
                    tp2_usdt = (tp2_sell_amounts * current_price) * fee_factor
                    balances = torch.where(tp2_trigger, balances + tp2_usdt, balances)
                    position_sizes = torch.where(tp2_trigger, position_sizes - tp2_sell_amounts, position_sizes)
                    tp2_hit = torch.where(tp2_trigger, True, tp2_hit)

                    # TP3
                    tp3_trigger = active_deals & ~tp3_hit & (profit_pcts >= tp_level3s)
                    tp3_sell_amounts = position_sizes * (tp_percent3s / 100.0)
                    # FIXED: Apply fees to TP sell
                    tp3_usdt = (tp3_sell_amounts * current_price) * fee_factor
                    balances = torch.where(tp3_trigger, balances + tp3_usdt, balances)
                    position_sizes = torch.where(tp3_trigger, position_sizes - tp3_sell_amounts, position_sizes)
                    tp3_hit = torch.where(tp3_trigger, True, tp3_hit)

                    # Update peak prices for trailing
                    peak_prices = torch.where(trailing_active & (current_price > peak_prices), current_price,
                                              peak_prices)

                    # Trailing stop
                    # FIXED: Limit trailing deviation to min(trailing_deviation, tp_level1)
                    effective_trailing_deviations = torch.minimum(trailing_deviations, tp_level1s)
                    trailing_thresholds = peak_prices * (1.0 - effective_trailing_deviations / 100.0)
                    trailing_trigger = trailing_active & (current_price <= trailing_thresholds)

                    # FIXED: Remove force_exit (not in CPU code)
                    # force_exit = active_deals & (current_rsi_4h > rsi_exit_thresholds)

                    # Execute full exits
                    exit_trigger = trailing_trigger  # FIXED: No | force_exit
                    # FIXED: Apply fees to full exit sell
                    exit_usdt = (position_sizes * current_price) * fee_factor
                    balances = torch.where(exit_trigger, balances + exit_usdt, balances)
                    position_sizes = torch.where(exit_trigger, 0.0, position_sizes)
                    active_deals = torch.where(exit_trigger, False, active_deals)
                    trailing_active = torch.where(exit_trigger, False, trailing_active)

                    # Close deals with zero position
                    zero_position = active_deals & (position_sizes < 0.0001)
                    active_deals = torch.where(zero_position, False, active_deals)

                    # FIXED: Update last_close_step on any close event
                    close_events = exit_trigger | zero_position
                    last_close_step = torch.where(close_events, torch.full_like(last_close_step, i), last_close_step)

                    # Record total portfolio values
                    portfolio_values = balances + position_sizes * current_price
                    balance_history[:, i] = portfolio_values

                    # Progress logging every 50k iterations
                    if i % 10000 == 0 and i > 0:
                        print(f"    GPU simulation progress: {i}/{data_length} ({i / data_length * 100:.1f}%)")

                except Exception as e:
                    print(f"    Error at iteration {i}: {e}")
                    # Skip this iteration and continue
                    continue

            # Final valuation for any remaining positions (no sell/fee, just market value)
            final_exit_usdt = position_sizes * self.prices[-1]  # FIXED: No fee here (matches CPU)
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

        print(f"    Running {len(param_batch)} simulations on CPU...")

        for i, params in enumerate(param_batch):
            try:
                # Create data frame from existing tensors (no recalculation needed)
                data_df = pd.DataFrame({
                    'open': self.prices.cpu().numpy(),
                    'high': self.prices.cpu().numpy(),
                    'low': self.prices.cpu().numpy(),
                    'close': self.prices.cpu().numpy(),
                    'vol': self.prices.cpu().numpy()  # Dummy volume
                }, index=pd.to_datetime(self.timestamps))

                # Create minimal backtester without recalculating indicators
                strategy = DCAStrategy(params, self.initial_balance)

                # Simple simulation loop
                for timestamp, row in data_df.iterrows():
                    # Use pre-calculated indicator values from tensors
                    ts_idx = list(data_df.index).index(timestamp)

                    current_indicators = {
                        'rsi_1h': self.rsi_1h_values[ts_idx].item(),
                        'rsi_4h': self.rsi_4h_values[ts_idx].item(),
                        'sma_fast_1h': self.sma_fast_1h_values[ts_idx].item(),
                        'sma_slow_1h': self.sma_slow_1h_values[ts_idx].item(),
                    }

                    # Check for new entry
                    if strategy.can_enter_new_deal(timestamp) and strategy.check_entry_conditions(row,
                                                                                                  current_indicators):
                        strategy.execute_base_order(row)

                    # Check active deal logic
                    if strategy.active_deal:
                        if strategy.check_safety_conditions(row, current_indicators):
                            strategy.execute_safety_order(row)

                        tp_triggers = strategy.check_take_profit_conditions(row)
                        for tp_level, sell_percent, _ in tp_triggers:
                            strategy.execute_take_profit(row, tp_level, sell_percent)

                        if strategy.check_trailing_stop(row):
                            strategy.execute_full_exit(row, 'trailing_stop')

                # Close any remaining position
                if strategy.active_deal:
                    last_row = data_df.iloc[-1]
                    strategy.execute_full_exit(last_row, 'end_of_data')

                # Calculate metrics
                final_balance = strategy.balance
                days = (data_df.index[-1] - data_df.index[0]).days
                years = days / 365.25 if days > 0 else 1
                apy = (pow(final_balance / self.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

                # Calculate drawdown from balance history
                if strategy.balance_history:
                    peak = strategy.balance_history[0][1]
                    max_dd = 0.0
                    for _, balance in strategy.balance_history:
                        if balance > peak:
                            peak = balance
                        drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
                        max_dd = max(max_dd, drawdown)
                else:
                    max_dd = 0.0

                results.append((apy, max_dd))

                if i % 4 == 0:
                    print(f"      CPU progress: {i + 1}/{len(param_batch)}")

            except Exception as e:
                print(f"      CPU simulation {i} failed: {e}")
                results.append((0.0, 100.0))

        print(f"    ✓ CPU fallback completed")
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

    def _suggest_params(self, trial):
        """Suggest parameters using Optuna trial"""
        tp_level1 = trial.suggest_categorical('tp_level1', [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

        # Fixed trailing deviation options - let the constraint be handled in the strategy
        trailing_deviation = trial.suggest_categorical('trailing_deviation',
                                                       [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

        return StrategyParams(
            # CONSTANT PARAMETERS
            base_percent=1.0,
            step_multiplier=1.5,
            volume_multiplier=1.2,
            max_safeties=8,
            rsi_entry_threshold=40.0,
            rsi_safety_threshold=30.0,
            rsi_exit_threshold=70.0,
            fees=0.075,

            # OPTIMIZABLE PARAMETERS
            initial_deviation=trial.suggest_categorical('initial_deviation', [3.0]),
            trailing_deviation=trailing_deviation,  # Will be constrained in strategy
            tp_level1=tp_level1,
            tp_percent1=trial.suggest_categorical('tp_percent1', [50.0]),
            tp_percent2=trial.suggest_categorical('tp_percent2', [30.0]),
            tp_percent3=trial.suggest_categorical('tp_percent3', [20.0])
        )

    def objective(self, trial):
        """Optuna objective function with discrete values"""
        params = self._suggest_params(trial)

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
                    'tp_level2': params.tp_level2,  # Show calculated value
                    'tp_level3': params.tp_level3,  # Show calculated value
                    'tp_percent1': params.tp_percent1,
                    'tp_percent2': params.tp_percent2,
                    'tp_percent3': params.tp_percent3,

                    # Include constants for reference
                    'base_percent': 1.0,
                    'step_multiplier': 1.5,
                    'volume_multiplier': 1.2,
                    'max_safeties': 8,
                    'rsi_entry_threshold': 40.0,
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
        """GPU-accelerated batch optimization with Optuna's Bayesian sampling"""
        if not self.use_gpu:
            return self.optimize(n_trials, 1)  # Fall back to CPU

        print(f"Starting GPU batch optimization with Optuna Bayesian sampling for {n_trials} trials...")
        print("Optimizing for: 60% APY + 40% Drawdown reduction")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Batch size: {self.gpu_simulator.max_batch_size}")
        print("=" * 80)

        # Set up Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')

        # Process in batches
        batch_size = self.gpu_simulator.max_batch_size
        all_results = []

        self.progress_bar = tqdm(
            total=n_trials,
            desc="Initializing GPU batches...",
            bar_format='{desc}\nProgress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} trials [{elapsed}<{remaining}]',
            ncols=80
        )

        trials_completed = 0
        try:
            while trials_completed < n_trials:
                current_batch_size = min(batch_size, n_trials - trials_completed)

                # Ask for a batch of trials
                optuna_trials = []
                batch_params = []
                for _ in range(current_batch_size):
                    trial = study.ask()
                    params = self._suggest_params(trial)
                    optuna_trials.append(trial)
                    batch_params.append(params)

                # Force much smaller batches for large datasets
                if len(self.gpu_simulator.prices) > 1000000:
                    batch_params = batch_params[:min(16, len(batch_params))]
                    optuna_trials = optuna_trials[:len(batch_params)]
                    print(f"  Large dataset - limiting to {len(batch_params)} simulations per batch")

                # Aggressive memory management
                if GPU_AVAILABLE:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                print(f"  Processing batch {(trials_completed // batch_size) + 1} with {len(batch_params)} simulations...")
                print(f"  GPU Memory before: {torch.cuda.memory_allocated(0) / 1024 ** 3:.1f}GB")

                # Add error protection (no timeout, as it doesn't interrupt properly)
                try:
                    batch_results = self.gpu_simulator.simulate_batch(batch_params)
                except Exception as e:
                    print(f"  ⚠️ Batch failed: {e} - falling back to CPU")
                    torch.cuda.empty_cache()
                    batch_results = self.gpu_simulator._simulate_batch_cpu(batch_params)

                print(f"  ✓ Batch completed successfully")

                # Tell Optuna the results and update best
                for j, (apy, max_drawdown) in enumerate(batch_results):
                    fitness = 0.6 * apy - 0.4 * max_drawdown
                    study.tell(optuna_trials[j], fitness)

                    if fitness > self.best_fitness:
                        self.best_fitness = fitness
                        self.best_apy = apy
                        self.best_drawdown = max_drawdown
                        self.best_params = {
                            # Only track optimizable parameters
                            'initial_deviation': batch_params[j].initial_deviation,
                            'trailing_deviation': batch_params[j].trailing_deviation,
                            'tp_level1': batch_params[j].tp_level1,
                            'tp_level2': batch_params[j].tp_level2,
                            'tp_level3': batch_params[j].tp_level3,
                            'tp_percent1': batch_params[j].tp_percent1,
                            'tp_percent2': batch_params[j].tp_percent2,
                            'tp_percent3': batch_params[j].tp_percent3,

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
                trials_completed += len(batch_results)

                # Update progress
                desc = f"Best: APY={self.best_apy:.1f}% DD={self.best_drawdown:.1f}% Fitness={self.best_fitness:.1f}"
                self.progress_bar.set_description(desc)
                self.progress_bar.update(len(batch_results))

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
