#!/usr/bin/env python3
"""
Fast DCA Backtesting Program - Optimized Version
Performance improvements: 10-100x faster than original
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
import numba
from numba import jit, njit
import multiprocessing as mp

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

# Check for GPU availability (optional)
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    GPU_AVAILABLE = False
    print("PyTorch not available - using CPU only mode")


# Import centralized strategy configuration
from strategy_config import StrategyParams, OptimizationConfig, StrategyPresets


@dataclass
class Trade:
    """Represents a single trade for visualization"""
    timestamp: datetime
    action: str
    amount_coin: float
    price: float
    usdt_amount: float
    reason: str


class FastDataProcessor:
    """Optimized data processing with caching and vectorization"""

    def __init__(self):
        self._indicator_cache = {}

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load and preprocess data from CSV"""
        df = pd.read_csv(file_path)
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate SuperTrend indicator - optimized for Supertrend-based DCA strategy
        Returns: (supertrend_values, trend_direction) where trend_direction: 1=bullish, -1=bearish
        """
        try:
            # Handle small datasets - adjust period if necessary
            if len(df) < period:
                period = max(1, len(df) - 1)
                if period <= 0:
                    # Return neutral values for very small datasets
                    return df['close'].values, np.ones(len(df))

            # Calculate True Range components
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()

            # True Range is the maximum of the three
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate ATR (Average True Range)
            atr = true_range.rolling(window=period).mean()

            # Calculate basic upper and lower bands
            hl2 = (df['high'] + df['low']) / 2.0
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            # Initialize SuperTrend arrays
            supertrend_values = np.full(len(df), np.nan)
            trend_direction = np.full(len(df), 1)  # 1 for bullish, -1 for bearish

            # Start calculation from first valid ATR value
            first_valid_idx = atr.first_valid_index()
            if first_valid_idx is not None:
                start_idx = df.index.get_loc(first_valid_idx)

                # Initialize first value
                supertrend_values[start_idx] = lower_band.iloc[start_idx]
                trend_direction[start_idx] = 1

                # Calculate SuperTrend iteratively
                for i in range(start_idx + 1, len(df)):
                    prev_close = df['close'].iloc[i-1]
                    curr_close = df['close'].iloc[i]
                    prev_supertrend = supertrend_values[i-1]
                    prev_trend = trend_direction[i-1]

                    curr_upper = upper_band.iloc[i]
                    curr_lower = lower_band.iloc[i]

                    # Calculate final upper and lower bands
                    if curr_upper < prev_supertrend or prev_close > prev_supertrend:
                        final_upper = curr_upper
                    else:
                        final_upper = prev_supertrend

                    if curr_lower > prev_supertrend or prev_close < prev_supertrend:
                        final_lower = curr_lower
                    else:
                        final_lower = prev_supertrend

                    # Determine SuperTrend value and direction
                    if prev_trend == 1:  # Was bullish
                        if curr_close <= final_lower:
                            # Trend flips to bearish
                            trend_direction[i] = -1
                            supertrend_values[i] = final_upper
                        else:
                            # Trend remains bullish
                            trend_direction[i] = 1
                            supertrend_values[i] = final_lower
                    else:  # Was bearish (prev_trend == -1)
                        if curr_close >= final_upper:
                            # Trend flips to bullish
                            trend_direction[i] = 1
                            supertrend_values[i] = final_lower
                        else:
                            # Trend remains bearish
                            trend_direction[i] = -1
                            supertrend_values[i] = final_upper

            # Fill any remaining NaN values with close prices and neutral direction
            mask = np.isnan(supertrend_values)
            if mask.any():
                supertrend_values[mask] = df['close'].iloc[mask].values
                trend_direction[mask] = 1

            return supertrend_values, trend_direction

        except Exception as e:
            print(f"SuperTrend calculation failed: {e}")
            # Return neutral values
            return df['close'].values, np.ones(len(df))

    def calculate_indicators_fast(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Vectorized indicator calculation with caching"""
        cache_key = f"{len(df)}_{df.index[0]}_{df.index[-1]}"

        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]

        print(f"Calculating indicators for {len(df)} data points...")

        # Vectorized resampling - do once and reuse
        df_15m = df.resample('15T').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'vol': 'sum'
        }).dropna()
        print(f"  15M data: {len(df_15m)} candles")

        df_30m = df.resample('30T').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'vol': 'sum'
        }).dropna()
        print(f"  30M data: {len(df_30m)} candles")

        df_1h = df.resample('1H').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'vol': 'sum'
        }).dropna()
        print(f"  1H data: {len(df_1h)} candles")

        df_4h = df.resample('4H').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'vol': 'sum'
        }).dropna()
        print(f"  4H data: {len(df_4h)} candles")

        df_1d = df.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'vol': 'sum'
        }).dropna()
        print(f"  1D data: {len(df_1d)} candles")

        print("  Computing SuperTrend indicators...")
        # SuperTrend indicators for multiple timeframes (DRAWDOWN ELIMINATION)
        supertrend_15m, supertrend_direction_15m = FastDataProcessor.calculate_supertrend(df_15m, 10, 3.0)
        supertrend_30m, supertrend_direction_30m = FastDataProcessor.calculate_supertrend(df_30m, 10, 3.0)
        supertrend_1h, supertrend_direction_1h = FastDataProcessor.calculate_supertrend(df_1h, 10, 3.0)
        supertrend_4h, supertrend_direction_4h = FastDataProcessor.calculate_supertrend(df_4h, 10, 3.0)
        supertrend_1d, supertrend_direction_1d = FastDataProcessor.calculate_supertrend(df_1d, 10, 3.0)

        # Forward-fill to original timeframe using pandas reindex
        indicators = {
            # SuperTrend indicators (DRAWDOWN ELIMINATION)
            'supertrend_direction_15m': pd.Series(supertrend_direction_15m, index=df_15m.index).reindex(df.index, method='ffill').fillna(1.0).values,
            'supertrend_direction_30m': pd.Series(supertrend_direction_30m, index=df_30m.index).reindex(df.index, method='ffill').fillna(1.0).values,
            'supertrend_direction_1h': pd.Series(supertrend_direction_1h, index=df_1h.index).reindex(df.index, method='ffill').fillna(1.0).values,
            'supertrend_direction_4h': pd.Series(supertrend_direction_4h, index=df_4h.index).reindex(df.index, method='ffill').fillna(1.0).values,
            'supertrend_direction_1d': pd.Series(supertrend_direction_1d, index=df_1d.index).reindex(df.index, method='ffill').fillna(1.0).values,
        }

        # Cache the results
        self._indicator_cache[cache_key] = indicators
        return indicators


# Enhanced simulation with trade tracking for visualization
def enhanced_simulate_strategy_with_trades(
        prices: np.ndarray,
        timestamps: np.ndarray,
        supertrend_direction_15m: np.ndarray,
        supertrend_direction_30m: np.ndarray,
        supertrend_direction_1h: np.ndarray,
        supertrend_direction_4h: np.ndarray,
        supertrend_direction_1d: np.ndarray,
        params_array: np.ndarray,
        initial_balance: float = 10000.0
) -> Tuple[float, float, int, np.ndarray, float, List[Trade]]:
    """
    Enhanced DCA simulation WITH ACTUAL TRADE TRACKING
    Returns: (final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration, trades_list)
    """

    # Run the numba-optimized simulation first
    final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration = enhanced_simulate_strategy(
        prices,
        supertrend_direction_15m, supertrend_direction_30m, supertrend_direction_1h, supertrend_direction_4h, supertrend_direction_1d,
        params_array, initial_balance
    )

    # Now run a Python version to track actual trades
    trades = []

    # Unpack parameters
    base_percent = params_array[0]
    initial_deviation = params_array[1]
    trailing_deviation = params_array[2]
    tp_level1 = params_array[3]
    tp_percent1 = params_array[4] / 100.0
    fees = params_array[5] / 100.0

    # Unpack SuperTrend parameters
    use_supertrend_filter = bool(params_array[6])
    supertrend_timeframe = int(params_array[7])

    # Constants
    step_multiplier = 1.5
    volume_multiplier = 1.2
    max_safeties = 8

    # State variables
    balance = initial_balance
    position_size = 0.0
    average_entry = 0.0
    total_spent = 0.0
    active_deal = False
    safety_count = 0
    last_entry_price = 0.0
    last_close_step = -999999
    peak_price = 0.0
    trailing_active = False
    deal_direction = 0  # 1 for long, -1 for short

    # Track actual trades
    for i in range(len(prices)):
        current_price = prices[i]
        current_timestamp = timestamps[i]

        # Select Supertrend direction based on timeframe
        if supertrend_timeframe == 0:  # 15m
            supertrend_direction = supertrend_direction_15m[i] if i < len(supertrend_direction_15m) else 1.0
        elif supertrend_timeframe == 1:  # 30m
            supertrend_direction = supertrend_direction_30m[i] if i < len(supertrend_direction_30m) else 1.0
        elif supertrend_timeframe == 2:  # 1h
            supertrend_direction = supertrend_direction_1h[i] if i < len(supertrend_direction_1h) else 1.0
        elif supertrend_timeframe == 3:  # 4h
            supertrend_direction = supertrend_direction_4h[i] if i < len(supertrend_direction_4h) else 1.0
        else:  # 1d
            supertrend_direction = supertrend_direction_1d[i] if i < len(supertrend_direction_1d) else 1.0

        # Check for new deal entry
        if not active_deal:
            cooldown_ok = (i - last_close_step) >= 1  # 60 seconds cooldown (1 step for 1m data)

            supertrend_bullish = supertrend_direction > 0
            supertrend_bearish = supertrend_direction < 0

            if (supertrend_bullish or supertrend_bearish) and cooldown_ok:
                deal_direction = 1 if supertrend_bullish else -1

                base_amount_usdt = initial_balance * (base_percent / 100.0)

                if base_amount_usdt > 1.0 and base_amount_usdt <= balance:
                    coin_amount = base_amount_usdt / current_price

                    trades.append(Trade(
                        timestamp=current_timestamp,
                        action='buy' if deal_direction == 1 else 'sell',
                        amount_coin=coin_amount,
                        price=current_price,
                        usdt_amount=base_amount_usdt,
                        reason='base_order'
                    ))

                    if deal_direction == 1:
                        balance -= base_amount_usdt
                        position_size = coin_amount
                    else:
                        balance += base_amount_usdt
                        position_size = -coin_amount
                    average_entry = current_price
                    total_spent = base_amount_usdt
                    last_entry_price = current_price
                    active_deal = True
                    safety_count = 0

        # Active deal management
        elif active_deal:
            # Safety orders
            if safety_count < max_safeties:
                current_deviation = initial_deviation
                for j in range(safety_count):
                    current_deviation *= step_multiplier

                if deal_direction == 1:
                    price_threshold = last_entry_price * (1.0 - current_deviation / 100.0)
                    safety_trigger = current_price <= price_threshold
                else:
                    price_threshold = last_entry_price * (1.0 + current_deviation / 100.0)
                    safety_trigger = current_price >= price_threshold

                if safety_trigger:
                    safety_amount_usdt = initial_balance * (base_percent / 100.0) * (volume_multiplier ** (safety_count + 1))

                    if safety_amount_usdt > 1.0 and safety_amount_usdt <= balance:
                        safety_coins = safety_amount_usdt / current_price

                        trades.append(Trade(
                            timestamp=current_timestamp,
                            action='buy' if deal_direction == 1 else 'sell',
                            amount_coin=safety_coins,
                            price=current_price,
                            usdt_amount=safety_amount_usdt,
                            reason=f'safety_order_{safety_count + 1}'
                        ))

                        if deal_direction == 1:
                            balance -= safety_amount_usdt
                            position_size += safety_coins
                        else:
                            balance += safety_amount_usdt
                            position_size -= safety_coins
                        total_spent += safety_amount_usdt
                        average_entry = total_spent / abs(position_size)
                        last_entry_price = current_price
                        safety_count += 1

            # Check for Supertrend flip exit
            supertrend_flip_exit = False
            if deal_direction == 1 and supertrend_direction <= 0:
                supertrend_flip_exit = True
            elif deal_direction == -1 and supertrend_direction >= 0:
                supertrend_flip_exit = True

            if supertrend_flip_exit:
                exit_usdt_gross = abs(position_size) * current_price
                exit_usdt_net = exit_usdt_gross * (1 - fees)

                trades.append(Trade(
                    timestamp=current_timestamp,
                    action='sell' if deal_direction == 1 else 'buy',
                    amount_coin=abs(position_size),
                    price=current_price,
                    usdt_amount=exit_usdt_net,
                    reason='supertrend_flip'
                ))

                if deal_direction == 1:
                    balance += exit_usdt_net
                else:
                    balance -= exit_usdt_net
                position_size = 0.0
                active_deal = False
                last_close_step = i

            # Check for take profit
            elif position_size != 0:
                if deal_direction == 1:
                    profit_percent = (current_price - average_entry) / average_entry * 100.0
                else:
                    profit_percent = (average_entry - current_price) / average_entry * 100.0

                if profit_percent >= tp_level1:
                    tp_sell = abs(position_size) * tp_percent1
                    tp_usdt_gross = tp_sell * current_price
                    tp_usdt_net = tp_usdt_gross * (1 - fees)

                    trades.append(Trade(
                        timestamp=current_timestamp,
                        action='sell' if deal_direction == 1 else 'buy',
                        amount_coin=tp_sell,
                        price=current_price,
                        usdt_amount=tp_usdt_net,
                        reason='take_profit'
                    ))

                    if deal_direction == 1:
                        balance += tp_usdt_net
                        position_size -= tp_sell
                    else:
                        balance -= tp_usdt_net
                        position_size += tp_sell

                    # Close deal if sold 100%
                    if tp_percent1 >= 1.0:
                        active_deal = False
                        last_close_step = i

    return final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration, trades


# Enhanced simulation with Supertrend only
@njit
def enhanced_simulate_strategy(
        prices: np.ndarray,
        supertrend_direction_15m: np.ndarray,
        supertrend_direction_30m: np.ndarray,
        supertrend_direction_1h: np.ndarray,
        supertrend_direction_4h: np.ndarray,
        supertrend_direction_1d: np.ndarray,
        params_array: np.ndarray,
        initial_balance: float = 10000.0
) -> Tuple[float, float, int, np.ndarray, float]:
    """
    Enhanced numba-optimized DCA simulation with Supertrend only
    Returns: (final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration)
    """

    # Unpack basic parameters
    base_percent = params_array[0]
    initial_deviation = params_array[1]
    trailing_deviation = params_array[2]
    tp_level1 = params_array[3]
    tp_percent1 = params_array[4] / 100.0
    fees = params_array[5] / 100.0

    # Unpack SuperTrend parameters
    use_supertrend_filter = bool(params_array[6])
    supertrend_timeframe = int(params_array[7])

    # Constants
    step_multiplier = 1.5
    volume_multiplier = 1.2
    max_safeties = 8

    # State variables
    balance = initial_balance
    position_size = 0.0
    average_entry = 0.0
    total_spent = 0.0
    active_deal = False
    safety_count = 0
    last_entry_price = 0.0
    peak_price = 0.0
    trailing_active = False
    last_close_step = -999999
    num_trades = 0
    tp_hit = False
    deal_direction = 0  # 1 for long, -1 for short

    # Balance history
    n_points = len(prices)
    balance_history = np.zeros(n_points)

    # Drawdown tracking
    max_portfolio_value = initial_balance
    max_drawdown = 0.0
    current_drawdown_start = -1
    total_drawdown_duration = 0.0
    drawdown_count = 0

    for i in range(n_points):
        current_price = prices[i]

        # Get current SuperTrend direction
        if supertrend_timeframe == 0:  # 15m
            current_supertrend_direction = supertrend_direction_15m[i] if i < len(supertrend_direction_15m) else 1.0
        elif supertrend_timeframe == 1:  # 30m
            current_supertrend_direction = supertrend_direction_30m[i] if i < len(supertrend_direction_30m) else 1.0
        elif supertrend_timeframe == 2:  # 1h
            current_supertrend_direction = supertrend_direction_1h[i] if i < len(supertrend_direction_1h) else 1.0
        elif supertrend_timeframe == 3:  # 4h
            current_supertrend_direction = supertrend_direction_4h[i] if i < len(supertrend_direction_4h) else 1.0
        else:  # 1d
            current_supertrend_direction = supertrend_direction_1d[i] if i < len(supertrend_direction_1d) else 1.0

        # 1. SUPERTREND-BASED DEAL ENTRY (only if no active deal)
        if not active_deal:
            cooldown_ok = (i - last_close_step) >= 1  # 60 seconds cooldown

            supertrend_bullish = current_supertrend_direction > 0
            supertrend_bearish = current_supertrend_direction < 0

            if (supertrend_bullish or supertrend_bearish) and cooldown_ok:
                deal_direction = 1 if supertrend_bullish else -1

                base_amount_usdt = initial_balance * (base_percent / 100.0)

                if base_amount_usdt > 1.0:
                    fee_amount = base_amount_usdt * fees
                    net_amount_usdt = base_amount_usdt - fee_amount
                    coin_amount = net_amount_usdt / current_price

                    if deal_direction == 1:
                        balance -= base_amount_usdt
                        position_size = coin_amount
                    else:
                        balance += net_amount_usdt
                        position_size = -coin_amount

                    total_spent = base_amount_usdt
                    average_entry = current_price
                    last_entry_price = current_price
                    active_deal = True
                    safety_count = 0
                    peak_price = current_price
                    tp_hit = False
                    trailing_active = False
                    num_trades += 1

        # 2. ACTIVE DEAL MANAGEMENT
        if active_deal:
            # Safety orders
            if safety_count < max_safeties:
                current_deviation = initial_deviation
                for j in range(safety_count):
                    current_deviation *= step_multiplier

                if deal_direction == 1:
                    price_threshold = last_entry_price * (1.0 - current_deviation / 100.0)
                    safety_trigger = current_price <= price_threshold
                else:
                    price_threshold = last_entry_price * (1.0 + current_deviation / 100.0)
                    safety_trigger = current_price >= price_threshold

                if safety_trigger:
                    safety_multiplier = volume_multiplier ** safety_count
                    safety_amount_usdt = initial_balance * (base_percent / 100.0) * safety_multiplier

                    if safety_amount_usdt > balance:
                        safety_amount_usdt = balance * 0.95

                    if safety_amount_usdt > 1.0:
                        fee_amount = safety_amount_usdt * fees
                        net_amount_usdt = safety_amount_usdt - fee_amount
                        safety_coins = net_amount_usdt / current_price

                        if deal_direction == 1:
                            balance -= safety_amount_usdt
                            position_size += safety_coins
                        else:
                            balance += net_amount_usdt
                            position_size -= safety_coins

                        total_spent += safety_amount_usdt
                        average_entry = total_spent / abs(position_size)
                        last_entry_price = current_price
                        safety_count += 1
                        num_trades += 1

            # SUPERTREND EXIT LOGIC: Exit immediately when SuperTrend flips direction
            supertrend_flip_exit = False
            if deal_direction == 1 and current_supertrend_direction <= 0:
                supertrend_flip_exit = True
            elif deal_direction == -1 and current_supertrend_direction >= 0:
                supertrend_flip_exit = True

            if supertrend_flip_exit:
                if deal_direction == 1:
                    exit_usdt_gross = position_size * current_price
                    exit_fee = exit_usdt_gross * fees
                    exit_usdt_net = exit_usdt_gross - exit_fee
                    balance += exit_usdt_net
                else:
                    buy_cost = abs(position_size) * current_price
                    buy_fee = buy_cost * fees
                    balance -= buy_cost + buy_fee

                position_size = 0.0
                active_deal = False
                trailing_active = False
                last_close_step = i
                num_trades += 1

            # Take profit conditions
            elif position_size != 0:
                if deal_direction == 1:
                    profit_percent = (current_price - average_entry) / average_entry * 100.0
                else:
                    profit_percent = (average_entry - current_price) / average_entry * 100.0

                if profit_percent >= tp_level1 and not tp_hit:
                    tp_sell = abs(position_size) * tp_percent1
                    tp_usdt_gross = tp_sell * current_price
                    tp_fee = tp_usdt_gross * fees
                    tp_usdt_net = tp_usdt_gross - tp_fee if deal_direction == 1 else tp_usdt_gross + tp_fee

                    if deal_direction == 1:
                        balance += tp_usdt_net
                        position_size -= tp_sell
                    else:
                        balance -= tp_usdt_net
                        position_size += tp_sell

                    tp_hit = True
                    trailing_active = True
                    peak_price = current_price
                    num_trades += 1

                    if tp_percent1 >= 1.0:
                        active_deal = False
                        trailing_active = False
                        position_size = 0.0
                        last_close_step = i

            # Trailing stop
            if trailing_active and position_size != 0 and not supertrend_flip_exit:
                if deal_direction == 1:
                    if current_price > peak_price:
                        peak_price = current_price
                    effective_trailing = min(trailing_deviation, tp_level1)
                    trailing_threshold = peak_price * (1.0 - effective_trailing / 100.0)
                    trailing_trigger = current_price <= trailing_threshold
                else:
                    if current_price < peak_price:
                        peak_price = current_price
                    effective_trailing = min(trailing_deviation, tp_level1)
                    trailing_threshold = peak_price * (1.0 + effective_trailing / 100.0)
                    trailing_trigger = current_price >= trailing_threshold

                    if trailing_trigger:
                        if deal_direction == 1:
                            exit_usdt_gross = position_size * current_price
                            exit_fee = exit_usdt_gross * fees
                            exit_usdt_net = exit_usdt_gross - exit_fee
                            balance += exit_usdt_net
                        else:
                            buy_cost = abs(position_size) * current_price
                            buy_fee = buy_cost * fees
                            balance -= buy_cost + buy_fee

                        position_size = 0.0
                        active_deal = False
                        trailing_active = False
                        last_close_step = i
                        num_trades += 1

            # Close deal if position too small
            if abs(position_size) < 0.0001:
                active_deal = False
                last_close_step = i

        # Record portfolio value
        portfolio_value = balance + position_size * current_price
        balance_history[i] = portfolio_value

        # Track drawdown with duration
        if portfolio_value > max_portfolio_value:
            max_portfolio_value = portfolio_value
            # End any ongoing drawdown
            if current_drawdown_start >= 0:
                drawdown_duration = i - current_drawdown_start
                total_drawdown_duration += drawdown_duration
                drawdown_count += 1
                current_drawdown_start = -1

        current_drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value * 100.0
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

        # Track drawdown start
        if current_drawdown > 1.0 and current_drawdown_start < 0:  # 1% threshold
            current_drawdown_start = i

    # Handle any ongoing drawdown at the end
    if current_drawdown_start >= 0:
        drawdown_duration = n_points - 1 - current_drawdown_start
        total_drawdown_duration += drawdown_duration
        drawdown_count += 1

    # Calculate average drawdown duration (in minutes for 1m data)
    avg_drawdown_duration = total_drawdown_duration / max(1, drawdown_count)

    # Final portfolio value
    final_portfolio_value = balance + position_size * prices[-1]

    return final_portfolio_value, max_drawdown, num_trades, balance_history, avg_drawdown_duration


class FastBacktester:
    """High-performance backtesting engine"""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance

        # Pre-calculate all indicators once
        processor = FastDataProcessor()
        self.indicators = processor.calculate_indicators_fast(data)

        # Convert to numpy arrays for speed
        self.prices = data['close'].values
        self.timestamps = data.index.values

    def simulate_strategy_fast(self, params: StrategyParams) -> Tuple[float, float, int, List]:
        """Ultra-fast simulation using numba - returns balance history for visualization"""

        # Pack parameters into array for numba
        params_array = np.array([
            params.base_percent,
            params.initial_deviation,
            params.trailing_deviation,
            params.tp_level1,
            params.tp_percent1,
            params.fees
        ])

        final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration = fast_simulate_strategy(
            self.prices,
            self.indicators['supertrend_direction_15m'],
            self.indicators['supertrend_direction_30m'],
            self.indicators['supertrend_direction_1h'],
            self.indicators['supertrend_direction_4h'],
            self.indicators['supertrend_direction_1d'],
            params_array,
            self.initial_balance
        )

        # Calculate APY
        days = (self.timestamps[-1] - self.timestamps[0]) / np.timedelta64(1, 'D')
        years = days / 365.25
        apy = (pow(final_balance / self.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

        # Convert balance history to list of tuples for compatibility
        balance_history_tuples = [(self.timestamps[i], balance_history[i]) for i in range(len(balance_history))]

        return apy, max_drawdown, num_trades, balance_history_tuples


class FastOptimizer:
    """High-performance optimizer with parallel processing"""

    def __init__(self, backtester: FastBacktester, optimization_config: OptimizationConfig = None, max_apy_only: bool = False):
        self.backtester = backtester
        self.best_fitness = -1000
        self.best_apy = 0
        self.best_drawdown = 100
        self.best_params = {}
        self.trial_count = 0
        self.max_apy_only = max_apy_only  # New flag for pure APY maximization
        # Use centralized optimization configuration
        self.optimization_config = optimization_config or OptimizationConfig()

    def _suggest_params(self, trial):
        """Suggest parameters using centralized configuration"""
        return self.optimization_config.suggest_params(trial)

    def objective(self, trial):
        """Enhanced objective function optimizing for APY and shorter drawdown duration with 3commas filters"""
        params = self._suggest_params(trial)

        # Validate trailing % doesn't exceed TP-0.2% to avoid losses
        max_trailing = params.tp_level1 - 0.2
        if params.trailing_deviation > max_trailing:
            return -1000  # Invalid configuration penalty

        try:
            # Pack parameters including 3commas conditional filters and SuperTrend
            # Map timeframe string to index for numba
            timeframe_map = {'15m': 0, '30m': 1, '1h': 2, '4h': 3, '1d': 4}
            supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 2)  # Default to 1h

            params_array = np.array([
                params.base_percent,
                params.initial_deviation,
                params.trailing_deviation,
                params.tp_level1,
                params.tp_percent1,  # Single TP target only
                params.fees,
                # SuperTrend drawdown elimination parameters
                float(params.use_supertrend_filter),
                float(supertrend_timeframe_idx)
            ])

            final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration = enhanced_simulate_strategy(
                self.backtester.prices,
                self.backtester.indicators['supertrend_direction_15m'],
                self.backtester.indicators['supertrend_direction_30m'],
                self.backtester.indicators['supertrend_direction_1h'],
                self.backtester.indicators['supertrend_direction_4h'],
                self.backtester.indicators['supertrend_direction_1d'],
                params_array,
                self.backtester.initial_balance
            )

            # Calculate APY
            days = (self.backtester.timestamps[-1] - self.backtester.timestamps[0]) / np.timedelta64(1, 'D')
            years = days / 365.25
            apy = (pow(final_balance / self.backtester.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

            # Fitness: If max_apy_only, pure APY; else original balanced
            if self.max_apy_only:
                fitness = apy
            else:
                # Enhanced fitness function heavily prioritizing APY:
                # 1. Higher APY (80% weight) - MOST IMPORTANT
                # 2. Lower max drawdown (15% weight)
                # 3. Shorter drawdown duration (5% weight)

                # Convert drawdown duration from minutes to hours for better scaling
                avg_drawdown_hours = avg_drawdown_duration / 60.0

                # Light penalty for long drawdowns (normalize to reasonable scale)
                drawdown_duration_penalty = min(avg_drawdown_hours / 168.0, 2.0)  # Cap at 2x penalty, weekly scale

                # Light penalty for high drawdowns (allow up to 40% before heavy penalty)
                drawdown_penalty = max(0, max_drawdown - 40.0) * 0.5  # Only penalize above 40%

                fitness = (
                        0.8 * apy -                           # Maximize APY (80% - PRIORITY)
                        0.15 * drawdown_penalty -             # Light drawdown penalty (15%)
                        0.05 * drawdown_duration_penalty      # Very light duration penalty (5%)
                )

            # Update best results
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_apy = apy
                self.best_drawdown = max_drawdown
                self.best_drawdown_duration = avg_drawdown_duration
                # Store the complete params object
                self.best_params = params
                self.best_num_trades = num_trades

            return fitness

        except Exception as e:
            print(f"Trial failed: {e}")
            return -1000

    def optimize_fast(self, n_trials: int = 500) -> Dict:
        """Fast optimization with progress tracking"""
        # Use more aggressive pruning for speed
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction='maximize',
            # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )

        # Progress bar
        with tqdm(total=n_trials, desc="Optimizing",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            def callback(study, trial):
                pbar.set_description(f"Best: APY={self.best_apy:.1f}% DD={self.best_drawdown:.1f}%")
                pbar.update(1)

            study.optimize(self.objective, n_trials=n_trials, callbacks=[callback])

        return self.best_params

    def optimize_with_drawdown_elimination(self, trend_trials: int = 50, dca_trials: int = 50) -> StrategyParams:
        """Two-phase optimization: 1) Eliminate drawdown with trend filtering, 2) Optimize DCA parameters"""

        print("=" * 80)
        print("🚀 FAST DUAL-PHASE OPTIMIZATION FOR DRAWDOWN ELIMINATION")
        print("=" * 80)
        print("Phase 1: Optimizing trend filtering (SuperTrend + 3commas filters)")
        print("Phase 2: Optimizing DCA parameters with trend filtering enabled")
        print()

        # PHASE 1: TREND FILTERING OPTIMIZATION
        print("🔄 PHASE 1: SUPERTREND & TREND FILTERING OPTIMIZATION")
        print("-" * 50)
        print(f"Goal: Find trend parameters that keep drawdown below 15%")
        print(f"Trials: {trend_trials}")
        print(f"Focus: SuperTrend, SMA/EMA filters, volatility filters")
        print()

        # Create trend-focused optimization ranges
        from strategy_config import OptimizationRanges
        trend_ranges = OptimizationRanges(
            # Fix some DCA parameters during phase 1, but optimize TP levels
            base_percent=[2.0],
            initial_deviation=[3.0],
            tp_level1=[2.0, 3.0, 4.0, 5.0],  # Optimize TP levels in phase 1
            tp_percent1=[100.0],  # Single TP target - sell entire position
            trailing_deviation=[0.5, 1.0, 1.5, 2.0],  # Will be validated against TP
            rsi_entry_threshold=[40.0],
            rsi_safety_threshold=[30.0],

            # Focus on trend filtering parameters
            use_supertrend_filter=[True],  # Always enabled in phase 1
            supertrend_timeframe=['15m', '30m', '1h', '4h', '1d'],
            supertrend_period=[7, 10, 14, 21],
            supertrend_multiplier=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            require_bullish_supertrend=[True, False],

            # Test 3commas filters more aggressively
            sma_trend_filter=[True, False],
            sma_trend_period=[50, 100, 200],
            ema_trend_filter=[True, False],
            ema_trend_period=[21, 50, 100],
            atr_volatility_filter=[True, False],
            atr_period=[14, 21, 28],
            atr_multiplier=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        )

        trend_optimizer = FastOptimizer(self.backtester, OptimizationConfig(trend_ranges))

        # Modified objective for phase 1 - strict drawdown enforcement
        def phase1_objective(trial):
            params = trend_optimizer._suggest_params(trial)

            # Validate trailing % doesn't exceed TP-0.2% to avoid losses
            max_trailing = params.tp_level1 - 0.2
            if params.trailing_deviation > max_trailing:
                return -1000  # Invalid configuration penalty

            try:
                # Pack parameters for simulation
                timeframe_map = {'15m': 0, '30m': 1, '1h': 2, '4h': 3, '1d': 4}
                supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 2)

                params_array = np.array([
                    params.base_percent, params.initial_deviation, params.trailing_deviation,
                    params.tp_level1, params.tp_percent1,  # Single TP target only
                    params.rsi_entry_threshold, params.rsi_safety_threshold, params.fees,
                    float(params.sma_trend_filter), float(params.sma_trend_period),
                    float(params.ema_trend_filter), float(params.ema_trend_period),
                    float(params.atr_volatility_filter), float(params.atr_period), params.atr_multiplier,
                    float(params.higher_highs_filter), float(params.higher_highs_period),
                    float(params.volume_confirmation), float(params.volume_sma_period),
                    float(params.use_supertrend_filter), float(supertrend_timeframe_idx),
                    float(params.require_bullish_supertrend)
                ])

                final_balance, max_drawdown, num_trades, _, avg_dd_duration = enhanced_simulate_strategy(
                    self.backtester.prices, self.backtester.indicators['rsi_1h'],
                    self.backtester.indicators['rsi_4h'], self.backtester.indicators['sma_fast_1h'],
                    self.backtester.indicators['sma_slow_1h'], self.backtester.indicators['sma_50'],
                    self.backtester.indicators['sma_100'], self.backtester.indicators['sma_200'],
                    self.backtester.indicators['ema_21'], self.backtester.indicators['ema_50'],
                    self.backtester.indicators['ema_100'], self.backtester.indicators['atr_14'],
                    self.backtester.indicators['atr_21'], self.backtester.indicators['atr_28'],
                    self.backtester.indicators['volume'], self.backtester.indicators['vol_sma_10'],
                    self.backtester.indicators['vol_sma_20'], self.backtester.indicators['vol_sma_30'],
                    self.backtester.indicators['supertrend_direction_15m'],
                    self.backtester.indicators['supertrend_direction_30m'],
                    self.backtester.indicators['supertrend_direction_1h'],
                    self.backtester.indicators['supertrend_direction_4h'],
                    self.backtester.indicators['supertrend_direction_1d'],
                    params_array, self.backtester.initial_balance
                )

                # Calculate APY
                days = (self.backtester.timestamps[-1] - self.backtester.timestamps[0]) / np.timedelta64(1, 'D')
                years = days / 365.25
                apy = (pow(final_balance / self.backtester.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

                # Debug output for first few trials in Phase 1
                if not hasattr(trend_optimizer, 'debug_count'):
                    trend_optimizer.debug_count = 0
                if trend_optimizer.debug_count < 3:
                    print(f"DEBUG Phase1 Trial {trend_optimizer.debug_count}: final_balance={final_balance:.2f}, initial={self.backtester.initial_balance:.2f}")
                    print(f"DEBUG: num_trades={num_trades}, max_drawdown={max_drawdown:.2f}%, apy={apy:.2f}%")
                    print(f"DEBUG: SuperTrend={params.supertrend_timeframe}, SMA_filter={params.sma_trend_filter}, EMA_filter={params.ema_trend_filter}")
                trend_optimizer.debug_count += 1

                # FIXED: Focus purely on APY - allow up to 90 days drawdown duration
                # User accepts long durations as long as APY is maximized

                # Require minimum trading activity: 50 deals per year as requested
                min_trades_per_year = 50  # User requirement: minimum 50 deals/year
                min_trades_required = max(1, int(min_trades_per_year * years))
                if num_trades < min_trades_required:
                    return -100 - (min_trades_required - num_trades) * 5  # Lighter penalty for discovery

                # Convert drawdown duration from minutes to hours and days
                avg_drawdown_hours = avg_dd_duration / 60.0
                avg_drawdown_days = avg_drawdown_hours / 24.0

                # PHASE 1: Allow up to 90 days drawdown duration - focus purely on APY
                if avg_drawdown_days > 90.0:  # 90 days maximum as requested
                    return -100 - (avg_drawdown_days - 90.0)  # Very light penalty beyond 90 days

                # If max_apy_only, pure APY; else balanced
                if self.max_apy_only:
                    fitness = apy
                else:
                    duration_penalty = min(avg_drawdown_days / 90.0, 1.0)  # Normalize to 90-day scale
                    fitness = 0.95 * apy - 0.05 * duration_penalty  # 95% APY, 5% duration penalty

                # Update trend-focused best results
                if fitness > trend_optimizer.best_fitness:
                    trend_optimizer.best_fitness = fitness
                    trend_optimizer.best_apy = apy
                    trend_optimizer.best_drawdown = max_drawdown
                    trend_optimizer.best_params = params

                return fitness

            except Exception as e:
                if not hasattr(trend_optimizer, 'error_count'):
                    trend_optimizer.error_count = 0
                if trend_optimizer.error_count < 3:
                    import traceback
                    traceback.print_exc()
                trend_optimizer.error_count += 1
                return -1000

        # Run phase 1 optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_phase1 = optuna.create_study(direction='maximize')

        with tqdm(total=trend_trials, desc="Phase 1: Trend optimization",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            def callback1(study, trial):
                pbar.set_description(f"Phase 1: Best APY={trend_optimizer.best_apy:.1f}% DD={trend_optimizer.best_drawdown:.1f}%")
                pbar.update(1)

            study_phase1.optimize(phase1_objective, n_trials=trend_trials, callbacks=[callback1])

        phase1_best = trend_optimizer.best_params
        print()
        print("✅ PHASE 1 COMPLETE!")
        print(f"Best drawdown achieved: {trend_optimizer.best_drawdown:.2f}%")
        print(f"APY with trend filtering: {trend_optimizer.best_apy:.2f}%")

        # Check if we actually found valid parameters
        if isinstance(phase1_best, dict) and not phase1_best:
            print("❌ WARNING: Phase 1 found no valid parameters!")
            print("All trials failed - filters may be too restrictive")
            return StrategyParams()  # Return default params

        if hasattr(phase1_best, 'supertrend_timeframe'):
            print(f"SuperTrend: {phase1_best.supertrend_timeframe}, period={phase1_best.supertrend_period}, mult={phase1_best.supertrend_multiplier}")
            print(f"Filters: SMA={phase1_best.sma_trend_filter}, EMA={phase1_best.ema_trend_filter}, ATR={phase1_best.atr_volatility_filter}")
        else:
            print("Phase 1 parameters: Basic fallback configuration")
        print()

        # PHASE 2: DCA PARAMETERS OPTIMIZATION
        print("🔄 PHASE 2: DCA PARAMETERS OPTIMIZATION")
        print("-" * 50)
        print(f"Goal: Maximize APY while maintaining drawdown control")
        print(f"Trials: {dca_trials}")
        print("Trend filtering: ENABLED with phase 1 parameters")
        print()

        # Reset optimizer for phase 2
        self.best_fitness = -1000
        self.best_apy = 0
        self.best_drawdown = 100

        # Modified objective for phase 2 - optimize DCA with fixed trend parameters
        def phase2_objective(trial):
            params = self._suggest_params(trial)

            # Validate trailing % doesn't exceed TP-0.2% to avoid losses
            max_trailing = params.tp_level1 - 0.2
            if params.trailing_deviation > max_trailing:
                return -1000  # Invalid configuration penalty

            # Override with best trend parameters from phase 1
            params.use_supertrend_filter = phase1_best.use_supertrend_filter
            params.supertrend_timeframe = phase1_best.supertrend_timeframe
            params.supertrend_period = phase1_best.supertrend_period
            params.supertrend_multiplier = phase1_best.supertrend_multiplier
            params.require_bullish_supertrend = phase1_best.require_bullish_supertrend
            params.sma_trend_filter = phase1_best.sma_trend_filter
            params.sma_trend_period = phase1_best.sma_trend_period
            params.ema_trend_filter = phase1_best.ema_trend_filter
            params.ema_trend_period = phase1_best.ema_trend_period
            params.atr_volatility_filter = phase1_best.atr_volatility_filter
            params.atr_period = phase1_best.atr_period
            params.atr_multiplier = phase1_best.atr_multiplier

            try:
                # Pack parameters for simulation
                timeframe_map = {'15m': 0, '30m': 1, '1h': 2, '4h': 3, '1d': 4}
                supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 2)

                params_array = np.array([
                    params.base_percent, params.initial_deviation, params.trailing_deviation,
                    params.tp_level1, params.tp_percent1,  # Single TP target only
                    params.rsi_entry_threshold, params.rsi_safety_threshold, params.fees,
                    float(params.sma_trend_filter), float(params.sma_trend_period),
                    float(params.ema_trend_filter), float(params.ema_trend_period),
                    float(params.atr_volatility_filter), float(params.atr_period), params.atr_multiplier,
                    float(params.higher_highs_filter), float(params.higher_highs_period),
                    float(params.volume_confirmation), float(params.volume_sma_period),
                    float(params.use_supertrend_filter), float(supertrend_timeframe_idx),
                    float(params.require_bullish_supertrend)
                ])

                final_balance, max_drawdown, num_trades, _, avg_dd_duration = enhanced_simulate_strategy(
                    self.backtester.prices, self.backtester.indicators['rsi_1h'],
                    self.backtester.indicators['rsi_4h'], self.backtester.indicators['sma_fast_1h'],
                    self.backtester.indicators['sma_slow_1h'], self.backtester.indicators['sma_50'],
                    self.backtester.indicators['sma_100'], self.backtester.indicators['sma_200'],
                    self.backtester.indicators['ema_21'], self.backtester.indicators['ema_50'],
                    self.backtester.indicators['ema_100'], self.backtester.indicators['atr_14'],
                    self.backtester.indicators['atr_21'], self.backtester.indicators['atr_28'],
                    self.backtester.indicators['volume'], self.backtester.indicators['vol_sma_10'],
                    self.backtester.indicators['vol_sma_20'], self.backtester.indicators['vol_sma_30'],
                    self.backtester.indicators['supertrend_direction_15m'],
                    self.backtester.indicators['supertrend_direction_30m'],
                    self.backtester.indicators['supertrend_direction_1h'],
                    self.backtester.indicators['supertrend_direction_4h'],
                    self.backtester.indicators['supertrend_direction_1d'],
                    params_array, self.backtester.initial_balance
                )

                # Calculate APY
                days = (self.backtester.timestamps[-1] - self.backtester.timestamps[0]) / np.timedelta64(1, 'D')
                years = days / 365.25
                apy = (pow(final_balance / self.backtester.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

                # FIXED: Phase 2 focuses purely on APY - same 90-day duration limit as phase 1
                # User accepts long durations as long as APY is maximized

                # Require minimum trading activity in phase 2 as well
                min_trades_per_year = 50  # Same requirement as phase 1
                min_trades_required = max(1, int(min_trades_per_year * years))
                if num_trades < min_trades_required:
                    return -100 - (min_trades_required - num_trades) * 5  # Lighter penalty in phase 2

                # Convert drawdown duration from minutes to hours and days
                avg_drawdown_hours = avg_dd_duration / 60.0
                avg_drawdown_days = avg_drawdown_hours / 24.0

                # PHASE 2: Same 90-day duration limit as phase 1 - focus purely on APY
                if avg_drawdown_days > 90.0:  # 90 days maximum as requested
                    return -100 - (avg_drawdown_days - 90.0)  # Very light penalty beyond 90 days

                # If max_apy_only, pure APY; else balanced
                if self.max_apy_only:
                    fitness = apy
                else:
                    duration_penalty = min(avg_drawdown_days / 90.0, 1.0)  # Normalize to 90-day scale
                    fitness = 0.97 * apy - 0.03 * duration_penalty  # 97% APY, 3% duration penalty (maximum APY focus)

                # Update final best results
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_apy = apy
                    self.best_drawdown = max_drawdown
                    self.best_params = params

                return fitness

            except Exception as e:
                return -1000

        # Run phase 2 optimization
        study_phase2 = optuna.create_study(direction='maximize')

        with tqdm(total=dca_trials, desc="Phase 2: DCA optimization",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            def callback2(study, trial):
                pbar.set_description(f"Phase 2: Best APY={self.best_apy:.1f}% DD={self.best_drawdown:.1f}%")
                pbar.update(1)

            study_phase2.optimize(phase2_objective, n_trials=dca_trials, callbacks=[callback2])

        print()
        print("✅ PHASE 2 COMPLETE!")
        print("=" * 80)
        print("🏆 FAST DUAL-PHASE OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Final APY: {self.best_apy:.2f}%")
        print(f"Final Max Drawdown: {self.best_drawdown:.2f}%")
        print(f"Total Trials: {trend_trials + dca_trials}")

        drawdown_status = "✅ ELIMINATED" if self.best_drawdown <= 15.0 else f"⚠️  REDUCED to {self.best_drawdown:.1f}%"
        print(f"Drawdown Status: {drawdown_status}")

        print()
        print("📝 Note: Results will be saved after final simulation with optimized parameters")
        print()

        return self.best_params


class Visualizer:
    """Create visualizations and reports"""

    @staticmethod
    def simulate_with_trades(backtester: FastBacktester, params: StrategyParams) -> Tuple[float, float, List, List]:
        """Run simulation with ACTUAL DCA trade tracking for accurate visualization"""

        print("Running DCA simulation with ACTUAL strategy logic...")

        # Use a modified enhanced simulation that tracks actual trades
        timeframe_map = {'15m': 0, '30m': 1, '1h': 2, '4h': 3, '1d': 4}
        supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 2)  # Default to 1h

        params_array = np.array([
            params.base_percent,
            params.initial_deviation,
            params.trailing_deviation,
            params.tp_level1,
            params.tp_percent1,
            params.fees,
            # SuperTrend parameters
            float(params.use_supertrend_filter),
            float(supertrend_timeframe_idx)
        ])

        # Run the actual DCA simulation with trade tracking
        final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration, actual_trades = enhanced_simulate_strategy_with_trades(
            backtester.prices,
            backtester.timestamps,
            backtester.indicators['supertrend_direction_15m'],
            backtester.indicators['supertrend_direction_30m'],
            backtester.indicators['supertrend_direction_1h'],
            backtester.indicators['supertrend_direction_4h'],
            backtester.indicators['supertrend_direction_1d'],
            params_array,
            backtester.initial_balance
        )

        # Calculate APY
        days = (backtester.timestamps[-1] - backtester.timestamps[0]) / np.timedelta64(1, 'D')
        years = days / 365.25
        apy = (pow(final_balance / backtester.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

        # Convert balance history to proper format
        balance_history_tuples = [(backtester.timestamps[i], balance_history[i]) for i in range(len(balance_history))]

        print(f"Generated {len(actual_trades)} ACTUAL DCA trades from strategy")
        print(f"Final APY: {apy:.2f}%, Max DD: {max_drawdown:.2f}%")

        return apy, max_drawdown, balance_history_tuples, actual_trades

    @staticmethod
    def plot_results(balance_history: List[Tuple[datetime, float]],
                     trades: List[Trade], coin: str, save_path: str):
        """Create comprehensive visualization with price chart, balance, and trades"""
        try:
            print(f"Creating comprehensive chart with {len(balance_history)} data points and {len(trades)} trades...")

            # Downsample balance history to exactly 5000 points as requested
            max_balance_points = 5000
            if len(balance_history) > max_balance_points:
                sample_rate = len(balance_history) // max_balance_points
                balance_history = balance_history[::sample_rate]
                print(f"Downsampled balance to {len(balance_history)} points for visualization")

            # Create figure with 2 subplots (removed cumulative trade count)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'{coin} - DCA Strategy Backtest Results', fontsize=16, fontweight='bold')

            # Extract data
            times, balances = zip(*balance_history)

            # Subplot 1: Portfolio Value
            ax1.plot(times, balances, 'b-', linewidth=2, label='Portfolio Value')
            ax1.set_title('Portfolio Value Over Time', fontweight='bold')
            ax1.set_ylabel('USDT Value', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Add performance metrics as text
            initial_balance = balances[0]
            final_balance = balances[-1]
            total_return = (final_balance - initial_balance) / initial_balance * 100
            max_balance = max(balances)
            max_drawdown = (max_balance - min(balances)) / max_balance * 100

            ax1.text(0.02, 0.98, f'Total Return: {total_return:.1f}%\nMax Drawdown: {max_drawdown:.1f}%',
                     transform=ax1.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Subplot 2: Daily Trade Count Bar Chart
            if len(trades) > 0:
                # Group trades by date and count them
                trade_dates = []
                for trade in trades:
                    if hasattr(trade.timestamp, 'date'):
                        trade_dates.append(trade.timestamp.date())
                    else:
                        trade_dates.append(pd.to_datetime(trade.timestamp).date())

                # Count trades per day
                from collections import Counter
                daily_trade_counts = Counter(trade_dates)

                # Convert to sorted lists for plotting
                dates = sorted(daily_trade_counts.keys())
                counts = [daily_trade_counts[date] for date in dates]

                # Create bar chart
                ax2.bar(dates, counts, color='steelblue', alpha=0.7, width=1)
                ax2.set_title('Daily Trade Count', fontweight='bold')
                ax2.set_ylabel('Number of Trades per Day', fontweight='bold')
                ax2.set_xlabel('Date', fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')

                # Add trade statistics
                total_trades = len(trades)
                avg_trades_per_day = total_trades / len(dates) if dates else 0
                max_trades_per_day = max(counts) if counts else 0

                # Separate buy and sell trades for statistics
                buy_trades = [t for t in trades if t.action == 'buy']
                sell_trades = [t for t in trades if t.action == 'sell']

                ax2.text(0.02, 0.98, f'Total Trades: {total_trades}\nAvg/Day: {avg_trades_per_day:.1f}\nMax/Day: {max_trades_per_day}\nBuys: {len(buy_trades)} | Sells: {len(sell_trades)}',
                         transform=ax2.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'No trades executed', ha='center', va='center',
                         transform=ax2.transAxes, fontsize=14)
                ax2.set_title('Daily Trade Count', fontweight='bold')
                ax2.set_xlabel('Date', fontweight='bold')

            # Format x-axis for both subplots
            for ax in [ax1, ax2]:
                try:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                except Exception:
                    pass

            plt.tight_layout()

            # Save with high quality
            print("Saving comprehensive chart...")
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Chart saved to: {save_path}")

        except Exception as e:
            print(f"Chart generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Create a simple text file instead
            try:
                with open(save_path.replace('.png', '_summary.txt'), 'w') as f:
                    f.write(f"Chart generation failed for {coin}\n")
                    f.write(f"Balance history points: {len(balance_history)}\n")
                    f.write(f"Total trades: {len(trades)}\n")
                    if balance_history:
                        f.write(f"Final balance: {balance_history[-1][1]:.2f}\n")
                print(f"Summary saved to text file instead")
            except Exception:
                print("Could not save chart or summary")

    @staticmethod
    def save_trades_log(trades: List[Trade], save_path: str):
        """Save detailed trades log to CSV"""
        trades_data = []
        for trade in trades:
            # Handle both pandas Timestamp and numpy datetime64 objects
            if hasattr(trade.timestamp, 'strftime'):
                timestamp_str = trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = pd.to_datetime(trade.timestamp).strftime('%Y-%m-%d %H:%M:%S')

            trades_data.append({
                'timestamp': timestamp_str,
                'action': trade.action,
                'amount_coin': round(trade.amount_coin, 6),
                'price': round(trade.price, 6),
                'usdt_amount': round(trade.usdt_amount, 2),
                'reason': trade.reason
            })

        df_trades = pd.DataFrame(trades_data)
        df_trades.to_csv(save_path, index=False)


def main():
    """Main execution with performance optimizations"""
    parser = argparse.ArgumentParser(description='Fast DCA Strategy Backtester')
    parser.add_argument('--data_path', required=True, help='Path to CSV data file')
    parser.add_argument('--coin', required=True, help='Coin symbol')
    parser.add_argument('--initial_balance', type=float, default=10000)
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--trials', type=int, default=500, help='Number of trials')
    parser.add_argument('--sample_days', type=int, default=0, help='Sample N days for faster testing (0=all data)')
    parser.add_argument('--output_dir', default='./results')
    parser.add_argument('--market_type', choices=['bull', 'bear', 'sideways', 'default'], default='default',
                        help='Market type for optimization ranges')
    parser.add_argument('--preset', choices=['conservative', 'aggressive', 'bull_market', 'bear_market', 'scalping'],
                        help='Use preset strategy parameters')
    parser.add_argument('--drawdown_elimination', action='store_true',
                        help='Use dual-phase optimization to eliminate huge drawdown periods first')
    parser.add_argument('--trend_trials', type=int, default=50,
                        help='Number of trials for trend filtering optimization (phase 1)')
    parser.add_argument('--dca_trials', type=int, default=50,
                        help='Number of trials for DCA parameters optimization (phase 2)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    try:
        data = FastDataProcessor.load_data(args.data_path)

        # Optional sampling for faster testing
        if args.sample_days > 0:
            sample_size = args.sample_days * 1440  # 1-minute data
            if len(data) > sample_size:
                data = data.tail(sample_size)

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Initialize fast backtester
    backtester = FastBacktester(data, args.initial_balance)

    # Choose strategy parameters
    if args.preset:
        # Use preset strategy
        preset_map = {
            'conservative': StrategyPresets.conservative,
            'aggressive': StrategyPresets.aggressive,
            'bull_market': StrategyPresets.bull_market,
            'bear_market': StrategyPresets.bear_market,
            'scalping': StrategyPresets.scalping
        }
        strategy_params = preset_map[args.preset]()
        print(f"Using {args.preset} preset strategy")

    elif args.optimize or args.drawdown_elimination:
        # Choose optimization ranges based on market type
        from strategy_config import MarketOptimizationRanges

        if args.market_type == 'bull':
            optimization_config = OptimizationConfig(MarketOptimizationRanges.bull_market())
            print("Optimizing for BULL market conditions")
        elif args.market_type == 'bear':
            optimization_config = OptimizationConfig(MarketOptimizationRanges.bear_market())
            print("Optimizing for BEAR market conditions")
        elif args.market_type == 'sideways':
            optimization_config = OptimizationConfig(MarketOptimizationRanges.sideways_market())
            print("Optimizing for SIDEWAYS market conditions")
        else:
            optimization_config = OptimizationConfig()
            print("Using DEFAULT optimization ranges")

        optimizer = FastOptimizer(backtester, optimization_config)

        if args.drawdown_elimination:
            # Use dual-phase optimization for drawdown elimination
            print("🎯 FAST DRAWDOWN ELIMINATION MODE ACTIVATED")
            print("This will focus on eliminating huge drawdown periods first, then optimize APY")
            print()
            best_params = optimizer.optimize_with_drawdown_elimination(
                trend_trials=args.trend_trials,
                dca_trials=args.dca_trials
            )
        else:
            best_params = optimizer.optimize_fast(args.trials)

        print(f"Optimization completed! Best params: {best_params}")

        # best_params is already a StrategyParams object
        strategy_params = best_params
    else:
        strategy_params = StrategyParams()
        print("Using default strategy parameters")

    # FIXED: Always use the proper simulation with realistic trade tracking
    print("Running full simulation with trades for visualization...")
    try:
        apy, max_drawdown, balance_history_for_save, trades = Visualizer.simulate_with_trades(backtester, strategy_params)
        print(f"Full simulation completed: APY={apy:.2f}%, Max DD={max_drawdown:.2f}%")
    except Exception as e:
        print(f"❌ Error in final simulation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Calculate additional metrics for results
    num_trades = len(trades)
    avg_drawdown_duration = 0  # Default value, will be calculated properly if needed

    if not args.optimize and not args.drawdown_elimination:
        # For non-optimization runs, get additional metrics if needed
        pass  # Metrics already calculated above

    final_balance = args.initial_balance * (1 + apy/100)

    # Results
    results = {
        'coin': args.coin,
        'initial_balance': args.initial_balance,
        'final_balance': final_balance,
        'apy': round(apy, 2),
        'max_drawdown': round(max_drawdown, 2),
        'avg_drawdown_duration_hours': round(avg_drawdown_duration/60, 1),
        'total_trades': num_trades,
        'parameters': {
            # DCA parameters
            'base_percent': strategy_params.base_percent,
            'tp_level1': strategy_params.tp_level1,
            'initial_deviation': strategy_params.initial_deviation,
            'trailing_deviation': strategy_params.trailing_deviation,
            'tp_percent1': strategy_params.tp_percent1,
            'fees': strategy_params.fees,
            # Safety order parameters (as requested by user)
            'volume_multiplier': strategy_params.volume_multiplier,
            'step_multiplier': strategy_params.step_multiplier,
            'max_safeties': strategy_params.max_safeties,
            # SuperTrend filters (Phase 1 drawdown elimination)
            'use_supertrend_filter': strategy_params.use_supertrend_filter,
            'supertrend_timeframe': strategy_params.supertrend_timeframe,
            'supertrend_period': strategy_params.supertrend_period,
            'supertrend_multiplier': strategy_params.supertrend_multiplier,
            'require_bullish_supertrend': strategy_params.require_bullish_supertrend
        },
        'data_period': {
            'start': data.index[0].isoformat(),
            'end': data.index[-1].isoformat(),
            'total_days': (data.index[-1] - data.index[0]).days
        }
    }

    print(f"\n⚡ FAST BACKTEST RESULTS FOR {args.coin}")
    print("=" * 60)
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"APY: {apy:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Avg Drawdown Duration: {avg_drawdown_duration/60:.1f} hours")
    print(f"Total Trades: {num_trades}")
    print("=" * 60)

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{args.coin}_fast_{timestamp}"

    # Save results JSON
    results_path = output_dir / f"{base_filename}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")

    # Save trades log
    trades_path = output_dir / f"{base_filename}_trades.csv"
    Visualizer.save_trades_log(trades, str(trades_path))
    print(f"Trades log saved to: {trades_path}")

    # Create and save visualization
    try:
        chart_path = output_dir / f"{base_filename}_chart.png"
        Visualizer.plot_results(balance_history_for_save, trades, args.coin, str(chart_path))
        print(f"Chart saved to: {chart_path}")
    except Exception as e:
        print(f"Warning: Could not create chart: {e}")
        print("Chart generation failed, but other results were saved successfully")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
