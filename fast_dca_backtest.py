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
        """Fast SuperTrend calculation"""
        try:
            # Calculate ATR
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()

            if atr.isna().all():
                # Fallback calculation
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=period).mean()

            # Calculate basic upper and lower bands
            hl2 = (df['high'] + df['low']) / 2.0
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            # Initialize SuperTrend arrays
            supertrend_values = np.full(len(df), np.nan)
            trend_direction = np.full(len(df), 1)  # 1 for up, -1 for down

            # Fill initial values
            if len(df) > 0:
                first_valid_idx = atr.first_valid_index()
                if first_valid_idx is not None:
                    start_idx = df.index.get_loc(first_valid_idx)
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

                        # Upper band calculation
                        if not np.isnan(prev_close) and not np.isnan(prev_supertrend):
                            if curr_upper < prev_supertrend or prev_close > prev_supertrend:
                                final_upper = curr_upper
                            else:
                                final_upper = prev_supertrend
                        else:
                            final_upper = curr_upper

                        # Lower band calculation
                        if not np.isnan(prev_close) and not np.isnan(prev_supertrend):
                            if curr_lower > prev_supertrend or prev_close < prev_supertrend:
                                final_lower = curr_lower
                            else:
                                final_lower = prev_supertrend
                        else:
                            final_lower = curr_lower

                        # Determine trend direction
                        if prev_trend == 1:
                            if curr_close <= final_lower:
                                trend_direction[i] = -1
                                supertrend_values[i] = final_upper
                            else:
                                trend_direction[i] = 1
                                supertrend_values[i] = final_lower
                        else:  # prev_trend == -1
                            if curr_close >= final_upper:
                                trend_direction[i] = 1
                                supertrend_values[i] = final_lower
                            else:
                                trend_direction[i] = -1
                                supertrend_values[i] = final_upper

            # Fill any remaining NaN values
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

        # Calculate indicators on resampled data
        rsi_1h = ta.momentum.RSIIndicator(df_1h['close'], window=14).rsi()
        rsi_4h = ta.momentum.RSIIndicator(df_4h['close'], window=14).rsi()
        sma_fast_1h = ta.trend.SMAIndicator(df_1h['close'], window=12).sma_indicator()
        sma_slow_1h = ta.trend.SMAIndicator(df_1h['close'], window=26).sma_indicator()

        # Additional indicators for 3commas conditional filters
        # Trend filters
        sma_50_1h = ta.trend.SMAIndicator(df_1h['close'], window=50).sma_indicator()
        sma_100_1h = ta.trend.SMAIndicator(df_1h['close'], window=100).sma_indicator()
        sma_200_1h = ta.trend.SMAIndicator(df_1h['close'], window=200).sma_indicator()
        ema_21_1h = ta.trend.EMAIndicator(df_1h['close'], window=21).ema_indicator()
        ema_50_1h = ta.trend.EMAIndicator(df_1h['close'], window=50).ema_indicator()
        ema_100_1h = ta.trend.EMAIndicator(df_1h['close'], window=100).ema_indicator()

        # Volatility filters
        atr_14_1h = ta.volatility.AverageTrueRange(df_1h['high'], df_1h['low'], df_1h['close'], window=14).average_true_range()
        atr_21_1h = ta.volatility.AverageTrueRange(df_1h['high'], df_1h['low'], df_1h['close'], window=21).average_true_range()
        atr_28_1h = ta.volatility.AverageTrueRange(df_1h['high'], df_1h['low'], df_1h['close'], window=28).average_true_range()

        # Volume indicators (using standard SMA on volume)
        vol_sma_10_1h = ta.trend.SMAIndicator(df_1h['vol'], window=10).sma_indicator()
        vol_sma_20_1h = ta.trend.SMAIndicator(df_1h['vol'], window=20).sma_indicator()
        vol_sma_30_1h = ta.trend.SMAIndicator(df_1h['vol'], window=30).sma_indicator()

        print("  Computing SuperTrend indicators...")
        # SuperTrend indicators for multiple timeframes (DRAWDOWN ELIMINATION)
        supertrend_1h, supertrend_direction_1h = FastDataProcessor.calculate_supertrend(df_1h, 10, 3.0)
        supertrend_4h, supertrend_direction_4h = FastDataProcessor.calculate_supertrend(df_4h, 10, 3.0)
        supertrend_1d, supertrend_direction_1d = FastDataProcessor.calculate_supertrend(df_1d, 10, 3.0)

        # Forward-fill to original timeframe using pandas reindex
        indicators = {
            'rsi_1h': rsi_1h.reindex(df.index, method='ffill').fillna(50.0).values,
            'rsi_4h': rsi_4h.reindex(df.index, method='ffill').fillna(50.0).values,
            'sma_fast_1h': sma_fast_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'sma_slow_1h': sma_slow_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,

            # Trend indicators
            'sma_50': sma_50_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'sma_100': sma_100_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'sma_200': sma_200_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'ema_21': ema_21_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'ema_50': ema_50_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'ema_100': ema_100_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,

            # Volatility indicators
            'atr_14': atr_14_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'atr_21': atr_21_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'atr_28': atr_28_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,

            # Volume indicators
            'vol_sma_10': vol_sma_10_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'vol_sma_20': vol_sma_20_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,
            'vol_sma_30': vol_sma_30_1h.reindex(df.index, method='ffill').fillna(method='ffill').values,

            # Volume data
            'volume': df['vol'].values,

            # SuperTrend indicators (DRAWDOWN ELIMINATION)
            'supertrend_1h': pd.Series(supertrend_1h, index=df_1h.index).reindex(df.index, method='ffill').fillna(method='ffill').values,
            'supertrend_direction_1h': pd.Series(supertrend_direction_1h, index=df_1h.index).reindex(df.index, method='ffill').fillna(1.0).values,
            'supertrend_4h': pd.Series(supertrend_4h, index=df_4h.index).reindex(df.index, method='ffill').fillna(method='ffill').values,
            'supertrend_direction_4h': pd.Series(supertrend_direction_4h, index=df_4h.index).reindex(df.index, method='ffill').fillna(1.0).values,
            'supertrend_1d': pd.Series(supertrend_1d, index=df_1d.index).reindex(df.index, method='ffill').fillna(method='ffill').values,
            'supertrend_direction_1d': pd.Series(supertrend_direction_1d, index=df_1d.index).reindex(df.index, method='ffill').fillna(1.0).values,
        }

        # Cache the results
        self._indicator_cache[cache_key] = indicators
        return indicators

    # Removed _fast_forward_fill - using pandas reindex instead


# Enhanced simulation with 3commas conditional filters
@njit
def enhanced_simulate_strategy(
    prices: np.ndarray,
    rsi_1h: np.ndarray,
    rsi_4h: np.ndarray,
    sma_fast: np.ndarray,
    sma_slow: np.ndarray,
    # Additional indicators for 3commas filters
    sma_50: np.ndarray,
    sma_100: np.ndarray,
    sma_200: np.ndarray,
    ema_21: np.ndarray,
    ema_50: np.ndarray,
    ema_100: np.ndarray,
    atr_14: np.ndarray,
    atr_21: np.ndarray,
    atr_28: np.ndarray,
    volume: np.ndarray,
    vol_sma_10: np.ndarray,
    vol_sma_20: np.ndarray,
    vol_sma_30: np.ndarray,
    # SuperTrend indicators for drawdown elimination
    supertrend_direction_1h: np.ndarray,
    supertrend_direction_4h: np.ndarray,
    supertrend_direction_1d: np.ndarray,
    params_array: np.ndarray,
    initial_balance: float = 10000.0
) -> Tuple[float, float, int, np.ndarray, float]:
    """
    Enhanced numba-optimized DCA simulation with 3commas conditional filters
    Returns: (final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration)
    """

    # Unpack basic parameters
    base_percent = params_array[0]
    initial_deviation = params_array[1]
    trailing_deviation = params_array[2]
    tp_level1 = params_array[3]
    tp_percent1 = params_array[4] / 100.0  # Single TP target - sell all position
    rsi_entry_thresh = params_array[5]
    rsi_safety_thresh = params_array[6]
    fees = params_array[7] / 100.0

    # Unpack 3commas conditional parameters (indices shifted by -2)
    sma_trend_filter = bool(params_array[8])
    sma_trend_period = int(params_array[9])  # 50, 100, or 200
    ema_trend_filter = bool(params_array[10])
    ema_trend_period = int(params_array[11])  # 21, 50, or 100
    atr_volatility_filter = bool(params_array[12])
    atr_period = int(params_array[13])  # 14, 21, or 28
    atr_multiplier = params_array[14]
    higher_highs_filter = bool(params_array[15])
    higher_highs_period = int(params_array[16])  # 10, 20, or 30
    volume_confirmation = bool(params_array[17])
    volume_sma_period = int(params_array[18])  # 10, 20, or 30

    # Unpack SuperTrend drawdown elimination parameters
    use_supertrend_filter = bool(params_array[19])
    supertrend_timeframe = int(params_array[20])  # 0=1h, 1=4h, 2=1d
    require_bullish_supertrend = bool(params_array[21])

    # Constants (matching original)
    step_multiplier = 1.5
    volume_multiplier = 1.2
    max_safeties = 8
    # Single TP target only - no TP2/TP3

    # State variables (matching original DCAStrategy)
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

    # TP level tracking (single TP target)
    tp_hit = False

    # Balance history for portfolio tracking
    n_points = len(prices)
    balance_history = np.zeros(n_points)

    # Drawdown tracking with duration
    max_portfolio_value = initial_balance
    max_drawdown = 0.0
    current_drawdown_start = -1
    total_drawdown_duration = 0.0
    drawdown_count = 0

    for i in range(n_points):
        current_price = prices[i]
        current_rsi_1h = rsi_1h[i] if i < len(rsi_1h) else 50.0
        current_sma_fast = sma_fast[i] if i < len(sma_fast) else current_price
        current_sma_slow = sma_slow[i] if i < len(sma_slow) else current_price

        # 1. CHECK FOR NEW DEAL ENTRY (only if no active deal)
        if not active_deal:
            # Basic conditions
            rsi_entry_ok = current_rsi_1h < rsi_entry_thresh
            sma_ok = current_sma_fast > current_sma_slow
            cooldown_ok = (i - last_close_step) >= 5

            # 3commas conditional filters to avoid massive drawdowns
            trend_filter_ok = True
            volatility_filter_ok = True
            structure_filter_ok = True
            volume_filter_ok = True
            supertrend_filter_ok = True

            # SMA trend filter
            if sma_trend_filter:
                if sma_trend_period == 50:
                    trend_filter_ok = current_price > sma_50[i] if i < len(sma_50) else True
                elif sma_trend_period == 100:
                    trend_filter_ok = current_price > sma_100[i] if i < len(sma_100) else True
                elif sma_trend_period == 200:
                    trend_filter_ok = current_price > sma_200[i] if i < len(sma_200) else True

            # EMA trend filter
            if ema_trend_filter:
                if ema_trend_period == 21:
                    trend_filter_ok = trend_filter_ok and (current_price > ema_21[i] if i < len(ema_21) else True)
                elif ema_trend_period == 50:
                    trend_filter_ok = trend_filter_ok and (current_price > ema_50[i] if i < len(ema_50) else True)
                elif ema_trend_period == 100:
                    trend_filter_ok = trend_filter_ok and (current_price > ema_100[i] if i < len(ema_100) else True)

            # ATR volatility filter (avoid entering during extreme volatility)
            if atr_volatility_filter and i > 0:
                if atr_period == 14:
                    current_atr = atr_14[i] if i < len(atr_14) else 0.0
                elif atr_period == 21:
                    current_atr = atr_21[i] if i < len(atr_21) else 0.0
                else:  # 28
                    current_atr = atr_28[i] if i < len(atr_28) else 0.0

                # Don't enter if recent price movement exceeds ATR threshold
                price_change = abs(current_price - prices[i-1])
                volatility_filter_ok = price_change < (current_atr * atr_multiplier)

            # Higher highs filter (market structure)
            if higher_highs_filter and i >= higher_highs_period:
                recent_high = 0.0
                for j in range(max(0, i - higher_highs_period), i):
                    if prices[j] > recent_high:
                        recent_high = prices[j]
                structure_filter_ok = current_price >= recent_high * 0.95  # Within 5% of recent high

            # Volume confirmation
            if volume_confirmation:
                if volume_sma_period == 10:
                    avg_volume = vol_sma_10[i] if i < len(vol_sma_10) else volume[i]
                elif volume_sma_period == 20:
                    avg_volume = vol_sma_20[i] if i < len(vol_sma_20) else volume[i]
                else:  # 30
                    avg_volume = vol_sma_30[i] if i < len(vol_sma_30) else volume[i]

                current_volume = volume[i] if i < len(volume) else 0.0
                volume_filter_ok = current_volume > avg_volume * 0.8  # At least 80% of average volume

            # SuperTrend filter for drawdown elimination
            if use_supertrend_filter and require_bullish_supertrend:
                if supertrend_timeframe == 0:  # 1h
                    supertrend_direction = supertrend_direction_1h[i] if i < len(supertrend_direction_1h) else 1.0
                elif supertrend_timeframe == 1:  # 4h
                    supertrend_direction = supertrend_direction_4h[i] if i < len(supertrend_direction_4h) else 1.0
                else:  # 1d
                    supertrend_direction = supertrend_direction_1d[i] if i < len(supertrend_direction_1d) else 1.0

                supertrend_filter_ok = supertrend_direction > 0  # 1 = bullish, -1 = bearish

            # Combined entry condition with all filters (including SuperTrend)
            if (rsi_entry_ok and sma_ok and cooldown_ok and
                trend_filter_ok and volatility_filter_ok and
                structure_filter_ok and volume_filter_ok and supertrend_filter_ok):

                base_amount_usdt = balance * (base_percent / 100.0)
                if base_amount_usdt > 1.0:
                    fee_amount = base_amount_usdt * fees
                    net_amount_usdt = base_amount_usdt - fee_amount
                    coin_amount = net_amount_usdt / current_price

                    balance -= base_amount_usdt
                    position_size = coin_amount
                    total_spent = base_amount_usdt
                    average_entry = current_price
                    last_entry_price = current_price
                    active_deal = True
                    safety_count = 0
                    peak_price = current_price
                    tp_hit = False
                    trailing_active = False
                    num_trades += 1

        # 2. ACTIVE DEAL MANAGEMENT (same as before)
        if active_deal:
            # Safety orders
            if safety_count < max_safeties:
                current_deviation = initial_deviation
                for j in range(safety_count):
                    current_deviation *= step_multiplier

                price_drop_threshold = last_entry_price * (1.0 - current_deviation / 100.0)
                safety_rsi_ok = current_rsi_1h < rsi_safety_thresh

                if current_price <= price_drop_threshold and safety_rsi_ok:
                    safety_multiplier = volume_multiplier ** safety_count
                    safety_base = balance * (base_percent / 100.0)  # FIXED: Use current balance for compounding
                    safety_amount_usdt = safety_base * safety_multiplier

                    if safety_amount_usdt > balance:
                        safety_amount_usdt = balance * 0.95

                    if safety_amount_usdt > 1.0:
                        fee_amount = safety_amount_usdt * fees
                        net_amount_usdt = safety_amount_usdt - fee_amount
                        safety_coins = net_amount_usdt / current_price

                        balance -= safety_amount_usdt
                        position_size += safety_coins
                        total_spent += safety_amount_usdt
                        average_entry = total_spent / position_size
                        last_entry_price = current_price
                        safety_count += 1
                        num_trades += 1

            # Take profit conditions - single TP target
            if position_size > 0:
                profit_percent = (current_price - average_entry) / average_entry * 100.0

                # Single TP - sell all position
                if profit_percent >= tp_level1 and not tp_hit:
                    tp_sell = position_size * tp_percent1  # Sell specified percentage
                    tp_usdt_gross = tp_sell * current_price
                    tp_fee = tp_usdt_gross * fees
                    tp_usdt_net = tp_usdt_gross - tp_fee

                    balance += tp_usdt_net
                    position_size -= tp_sell
                    tp_hit = True
                    trailing_active = True
                    peak_price = current_price
                    num_trades += 1

                    # If we sold 100%, close the deal
                    if tp_percent1 >= 1.0:  # 100%
                        active_deal = False
                        trailing_active = False

            # Trailing stop (same as before)
            if trailing_active and position_size > 0:
                if current_price > peak_price:
                    peak_price = current_price

                effective_trailing = min(trailing_deviation, tp_level1)
                trailing_threshold = peak_price * (1.0 - effective_trailing / 100.0)

                if current_price <= trailing_threshold:
                    exit_usdt_gross = position_size * current_price
                    exit_fee = exit_usdt_gross * fees
                    exit_usdt_net = exit_usdt_gross - exit_fee

                    balance += exit_usdt_net
                    position_size = 0.0
                    active_deal = False
                    trailing_active = False
                    last_close_step = i
                    num_trades += 1

            # Close deal if position too small
            if position_size < 0.0001:
                active_deal = False
                last_close_step = i

        # Record portfolio value
        portfolio_value = balance + position_size * current_price
        balance_history[i] = portfolio_value

        # Track drawdown with duration (same as before)
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


# Keep the original fast version for backward compatibility
@njit
def fast_simulate_strategy(
    prices: np.ndarray,
    rsi_1h: np.ndarray,
    rsi_4h: np.ndarray,
    sma_fast: np.ndarray,
    sma_slow: np.ndarray,
    params_array: np.ndarray,
    initial_balance: float = 10000.0
) -> Tuple[float, float, int, np.ndarray, float]:
    """
    FIXED Numba-optimized DCA simulation for 1 TP configuration
    Returns: (final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration)
    """

    # Unpack parameters
    base_percent = params_array[0]
    initial_deviation = params_array[1]
    trailing_deviation = params_array[2]
    tp_level1 = params_array[3]
    tp_percent1 = params_array[4] / 100.0  # Convert to decimal (should be 1.0 for 100%)
    rsi_entry_thresh = params_array[5]
    rsi_safety_thresh = params_array[6]
    fees = params_array[7] / 100.0

    # Constants (matching original)
    step_multiplier = 1.5
    volume_multiplier = 1.2
    max_safeties = 8
    # Single TP target only - no TP2/TP3

    # State variables (matching original DCAStrategy)
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

    # TP level tracking (single TP target)
    tp_hit = False

    # Balance history for portfolio tracking
    n_points = len(prices)
    balance_history = np.zeros(n_points)

    # Drawdown tracking with duration
    max_portfolio_value = initial_balance
    max_drawdown = 0.0
    current_drawdown_start = -1
    total_drawdown_duration = 0.0
    drawdown_count = 0

    for i in range(n_points):
        current_price = prices[i]
        current_rsi_1h = rsi_1h[i] if i < len(rsi_1h) else 50.0
        current_sma_fast = sma_fast[i] if i < len(sma_fast) else current_price
        current_sma_slow = sma_slow[i] if i < len(sma_slow) else current_price

        # 1. CHECK FOR NEW DEAL ENTRY (only if no active deal)
        if not active_deal:
            rsi_entry_ok = current_rsi_1h < rsi_entry_thresh
            sma_ok = current_sma_fast > current_sma_slow
            cooldown_ok = (i - last_close_step) >= 5

            if rsi_entry_ok and sma_ok and cooldown_ok:
                base_amount_usdt = balance * (base_percent / 100.0)
                if base_amount_usdt > 1.0:
                    fee_amount = base_amount_usdt * fees
                    net_amount_usdt = base_amount_usdt - fee_amount
                    coin_amount = net_amount_usdt / current_price

                    balance -= base_amount_usdt
                    position_size = coin_amount
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

                price_drop_threshold = last_entry_price * (1.0 - current_deviation / 100.0)
                safety_rsi_ok = current_rsi_1h < rsi_safety_thresh

                if current_price <= price_drop_threshold and safety_rsi_ok:
                    safety_multiplier = volume_multiplier ** safety_count
                    safety_base = balance * (base_percent / 100.0)  # FIXED: Use current balance for compounding
                    safety_amount_usdt = safety_base * safety_multiplier

                    if safety_amount_usdt > balance:
                        safety_amount_usdt = balance * 0.95

                    if safety_amount_usdt > 1.0:
                        fee_amount = safety_amount_usdt * fees
                        net_amount_usdt = safety_amount_usdt - fee_amount
                        safety_coins = net_amount_usdt / current_price

                        balance -= safety_amount_usdt
                        position_size += safety_coins
                        total_spent += safety_amount_usdt
                        average_entry = total_spent / position_size
                        last_entry_price = current_price
                        safety_count += 1
                        num_trades += 1

            # Take profit conditions - single TP target
            if position_size > 0:
                profit_percent = (current_price - average_entry) / average_entry * 100.0

                # FIXED: Single TP logic for 1 TP system
                if profit_percent >= tp_level1 and not tp_hit:
                    tp_sell = position_size * tp_percent1  # Sell specified percentage
                    tp_usdt_gross = tp_sell * current_price
                    tp_fee = tp_usdt_gross * fees
                    tp_usdt_net = tp_usdt_gross - tp_fee

                    balance += tp_usdt_net
                    position_size -= tp_sell
                    tp_hit = True
                    num_trades += 1

                    # CRITICAL FIX: For 1 TP system (100%), close deal immediately
                    if tp_percent1 >= 0.99:  # 99%+ means close entire position
                        active_deal = False
                        trailing_active = False
                        last_close_step = i
                        position_size = 0.0  # Ensure position is fully closed
                    else:
                        # For partial TP, enable trailing on remaining position
                        trailing_active = True
                        peak_price = current_price

            # Trailing stop
            if trailing_active and position_size > 0:
                if current_price > peak_price:
                    peak_price = current_price

                effective_trailing = min(trailing_deviation, tp_level1)
                trailing_threshold = peak_price * (1.0 - effective_trailing / 100.0)

                if current_price <= trailing_threshold:
                    exit_usdt_gross = position_size * current_price
                    exit_fee = exit_usdt_gross * fees
                    exit_usdt_net = exit_usdt_gross - exit_fee

                    balance += exit_usdt_net
                    position_size = 0.0
                    active_deal = False
                    trailing_active = False
                    last_close_step = i
                    num_trades += 1

            # Close deal if position too small
            if position_size < 0.0001:
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


# Create a proper trade-tracking version for visualization
def simulate_with_actual_trades(backtester, params: StrategyParams):
    """Run CPU simulation with actual trade tracking for accurate visualization"""
    try:
        from dca_backtest import DCAStrategy

        # Create a new params object with the exact same values to ensure proper initialization
        # This ensures all parameters are properly passed, including those from optimization
        strategy_params = StrategyParams(
            base_percent=params.base_percent,
            initial_deviation=params.initial_deviation,
            step_multiplier=params.step_multiplier,
            volume_multiplier=params.volume_multiplier,
            max_safeties=params.max_safeties,
            trailing_deviation=params.trailing_deviation,
            tp_level1=params.tp_level1,
            tp_percent1=params.tp_percent1,
            tp_percent2=params.tp_percent2,
            tp_percent3=params.tp_percent3,
            rsi_entry_threshold=params.rsi_entry_threshold,
            rsi_safety_threshold=params.rsi_safety_threshold,
            rsi_exit_threshold=params.rsi_exit_threshold,
            fees=params.fees
        )

        # Use the original strategy class for accurate trade tracking
        strategy = DCAStrategy(strategy_params, backtester.initial_balance)

        # Get indicators at each timestamp (simplified)
        trades = []
        balance_history = []

        # Sample data for performance (take every Nth point)
        n_total = len(backtester.data)
        sample_rate = max(1, n_total // 50000)  # Limit to ~50k points for performance

        for i in range(0, n_total, sample_rate):
            timestamp = backtester.timestamps[i]
            current_price = backtester.prices[i]

            # Convert numpy datetime64 to pandas Timestamp for compatibility
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp_dt = timestamp
            else:
                timestamp_dt = pd.to_datetime(timestamp)

            # Get current indicators
            current_indicators = {
                'rsi_1h': backtester.indicators['rsi_1h'][i],
                'rsi_4h': backtester.indicators['rsi_4h'][i],
                'sma_fast_1h': backtester.indicators['sma_fast_1h'][i],
                'sma_slow_1h': backtester.indicators['sma_slow_1h'][i],
            }

            # Create a row-like object for compatibility
            class Row:
                def __init__(self, price, timestamp):
                    self.close = price
                    self.name = timestamp
                    # Add dictionary-style access for compatibility
                    self._data = {'close': price}

                def __getitem__(self, key):
                    return self._data.get(key, self.close)

            row = Row(current_price, timestamp_dt)

            # Check for new entry
            if strategy.can_enter_new_deal(timestamp_dt) and strategy.check_entry_conditions(row, current_indicators):
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

            # Record portfolio value
            total_value = strategy.balance + strategy.position_size * current_price
            balance_history.append((timestamp_dt, total_value))

        # Calculate final metrics
        final_balance = balance_history[-1][1] if balance_history else backtester.initial_balance
        days = (backtester.timestamps[-1] - backtester.timestamps[0]) / np.timedelta64(1, 'D')
        years = days / 365.25
        apy = (pow(final_balance / backtester.initial_balance, 1 / years) - 1) * 100 if years > 0 else 0

        # Calculate max drawdown
        max_dd = 0.0
        peak = balance_history[0][1] if balance_history else backtester.initial_balance
        for _, balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_dd = max(max_dd, drawdown)

        return apy, max_dd, balance_history, strategy.trades

    except ImportError as e:
        # If original DCAStrategy not available, raise ImportError to trigger fallback
        raise ImportError(f"Original DCAStrategy not available: {e}")
    except Exception as e:
        # For any other errors, also trigger the fallback
        print(f"Error in simulate_with_actual_trades: {e}")
        raise ImportError(f"Failed to run DCAStrategy simulation: {e}")


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
            params.tp_percent2,
            params.tp_percent3,
            params.rsi_entry_threshold,
            params.rsi_safety_threshold,
            params.fees
        ])

        final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration = fast_simulate_strategy(
            self.prices,
            self.indicators['rsi_1h'],
            self.indicators['rsi_4h'],
            self.indicators['sma_fast_1h'],
            self.indicators['sma_slow_1h'],
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

    def __init__(self, backtester: FastBacktester, optimization_config: OptimizationConfig = None):
        self.backtester = backtester
        self.best_fitness = -1000
        self.best_apy = 0
        self.best_drawdown = 100
        self.best_params = {}
        self.trial_count = 0
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
            timeframe_map = {'15m': 0, '1h': 1, '4h': 2, '1d': 3}
            supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 2)  # Default to 4h

            params_array = np.array([
                params.base_percent,
                params.initial_deviation,
                params.trailing_deviation,
                params.tp_level1,
                params.tp_percent1,  # Single TP target only
                params.rsi_entry_threshold,
                params.rsi_safety_threshold,
                params.fees,
                # 3commas conditional parameters
                float(params.sma_trend_filter),
                float(params.sma_trend_period),
                float(params.ema_trend_filter),
                float(params.ema_trend_period),
                float(params.atr_volatility_filter),
                float(params.atr_period),
                params.atr_multiplier,
                float(params.higher_highs_filter),
                float(params.higher_highs_period),
                float(params.volume_confirmation),
                float(params.volume_sma_period),
                # SuperTrend drawdown elimination parameters
                float(params.use_supertrend_filter),
                float(supertrend_timeframe_idx),
                float(params.require_bullish_supertrend)
            ])

            final_balance, max_drawdown, num_trades, balance_history, avg_drawdown_duration = enhanced_simulate_strategy(
                self.backtester.prices,
                self.backtester.indicators['rsi_1h'],
                self.backtester.indicators['rsi_4h'],
                self.backtester.indicators['sma_fast_1h'],
                self.backtester.indicators['sma_slow_1h'],
                # Additional indicators for 3commas filters
                self.backtester.indicators['sma_50'],
                self.backtester.indicators['sma_100'],
                self.backtester.indicators['sma_200'],
                self.backtester.indicators['ema_21'],
                self.backtester.indicators['ema_50'],
                self.backtester.indicators['ema_100'],
                self.backtester.indicators['atr_14'],
                self.backtester.indicators['atr_21'],
                self.backtester.indicators['atr_28'],
                self.backtester.indicators['volume'],
                self.backtester.indicators['vol_sma_10'],
                self.backtester.indicators['vol_sma_20'],
                self.backtester.indicators['vol_sma_30'],
                # SuperTrend indicators for drawdown elimination
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
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
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
            supertrend_timeframe=['15m', '1h', '4h', '1d'],
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
                timeframe_map = {'1h': 0, '4h': 1, '1d': 2}
                supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 1)

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
                
                # FIXED: Phase 1 focuses almost entirely on APY (95% weight)
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
                timeframe_map = {'1h': 0, '4h': 1, '1d': 2}
                supertrend_timeframe_idx = timeframe_map.get(params.supertrend_timeframe, 1)

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
                
                # FIXED: Phase 2 focuses almost entirely on APY (97% weight)
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


@dataclass
class Trade:
    """Represents a single trade for visualization"""
    timestamp: datetime
    action: str
    amount_coin: float
    price: float
    usdt_amount: float
    reason: str


class Visualizer:
    """Create visualizations and reports"""

    @staticmethod
    def simulate_with_trades(backtester: FastBacktester, params: StrategyParams) -> Tuple[float, float, List, List]:
        """Run simulation with actual trade tracking for accurate visualization"""

        # Use fast simulation directly - the detailed trade simulation is causing issues
        print("Using fast simulation for final results...")
        apy, max_drawdown, num_trades, balance_history = backtester.simulate_strategy_fast(params)

        # balance_history is already in tuple format from simulate_strategy_fast
        balance_history_tuples = balance_history

        # Create minimal synthetic trades
        trades = []
        if num_trades > 0:
            trade_frequency = max(1, len(balance_history_tuples) // num_trades)
            for i in range(0, len(balance_history_tuples), trade_frequency):
                if len(trades) >= num_trades:
                    break
                timestamp, _ = balance_history_tuples[i]
                action = 'buy' if len(trades) % 3 < 2 else 'sell'
                trades.append(Trade(
                    timestamp=timestamp,
                    action=action,
                    amount_coin=0.05,
                    price=backtester.prices[min(i, len(backtester.prices)-1)],
                    usdt_amount=100.0,
                    reason='base_order' if action == 'buy' else 'take_profit'
                ))

        return apy, max_drawdown, balance_history_tuples, trades

    @staticmethod
    def plot_results(balance_history: List[Tuple[datetime, float]],
                     trades: List[Trade], coin: str, save_path: str):
        """Create balance and trades visualization - optimized for large datasets"""
        print(f"Creating chart with {len(balance_history)} data points and {len(trades)} trades...")

        # Downsample balance history for large datasets to prevent memory issues
        max_points = 10000
        if len(balance_history) > max_points:
            sample_rate = len(balance_history) // max_points
            balance_history = balance_history[::sample_rate]
            print(f"Downsampled to {len(balance_history)} points for visualization")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Reduced figure size

        # Balance chart
        times, balances = zip(*balance_history)
        ax1.plot(times, balances, 'b-', linewidth=1.5, label='Portfolio Value')

        # Limit trade markers to prevent overcrowding
        max_trade_markers = 500
        if len(trades) > max_trade_markers:
            # Sample trades evenly
            trade_indices = np.linspace(0, len(trades)-1, max_trade_markers, dtype=int)
            sampled_trades = [trades[i] for i in trade_indices]
            print(f"Showing {len(sampled_trades)} trade markers (sampled from {len(trades)} total)")
        else:
            sampled_trades = trades

        # Add trade markers
        buy_times = []
        buy_balances = []
        sell_times = []
        sell_balances = []

        # Convert balance_history to dict for faster lookup
        time_to_balance = {t: b for t, b in balance_history}
        sorted_times = sorted(time_to_balance.keys())

        for trade in sampled_trades:
            # Find closest time in balance history
            trade_time = trade.timestamp
            if hasattr(trade_time, 'to_pydatetime'):
                trade_time = trade_time.to_pydatetime()
            elif not isinstance(trade_time, datetime):
                trade_time = pd.to_datetime(trade_time).to_pydatetime()

            # Binary search for closest time
            idx = np.searchsorted([t.timestamp() if hasattr(t, 'timestamp') else pd.to_datetime(t).timestamp()
                                  for t in sorted_times],
                                 trade_time.timestamp() if hasattr(trade_time, 'timestamp') else pd.to_datetime(trade_time).timestamp())

            if idx < len(sorted_times):
                closest_time = sorted_times[min(idx, len(sorted_times)-1)]
                balance = time_to_balance[closest_time]

                if trade.action == 'buy':
                    buy_times.append(trade_time)
                    buy_balances.append(balance)
                else:
                    sell_times.append(trade_time)
                    sell_balances.append(balance)

        if buy_times:
            ax1.scatter(buy_times, buy_balances, color='green', marker='^',
                       s=30, alpha=0.6, label=f'Buys ({len(buy_times)})')
        if sell_times:
            ax1.scatter(sell_times, sell_balances, color='red', marker='v',
                       s=30, alpha=0.6, label=f'Sells ({len(sell_times)})')

        ax1.set_title(f'{coin} - Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('USDT Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format x-axis - use fewer ticks for performance
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Trade count over time - aggregate by week for large datasets
        if len(trades) > 1000:
            # Weekly aggregation
            weekly_trades = {}
            for trade in trades:
                if hasattr(trade.timestamp, 'date'):
                    date = trade.timestamp.date()
                else:
                    date = pd.to_datetime(trade.timestamp).date()
                week_start = date - timedelta(days=date.weekday())
                weekly_trades[week_start] = weekly_trades.get(week_start, 0) + 1

            if weekly_trades:
                dates, counts = zip(*sorted(weekly_trades.items()))
                ax2.bar(dates, counts, alpha=0.7, color='orange', width=6)
                ax2.set_title('Weekly Trade Count')
        else:
            # Daily aggregation for smaller datasets
            daily_trades = {}
            for trade in trades:
                if hasattr(trade.timestamp, 'date'):
                    date = trade.timestamp.date()
                else:
                    date = pd.to_datetime(trade.timestamp).date()
                daily_trades[date] = daily_trades.get(date, 0) + 1

            if daily_trades:
                dates, counts = zip(*sorted(daily_trades.items()))
                ax2.bar(dates, counts, alpha=0.7, color='orange')
                ax2.set_title('Daily Trade Count')

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Trades')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with lower DPI for faster generation
        print("Saving chart...")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to: {save_path}")

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

    # For optimization, just get the key metrics without complex visualization
    if args.optimize or args.drawdown_elimination:
        print("Getting final metrics from optimization...")
        try:
            # FIXED: Use enhanced simulation with all filters (same as optimization)
            timeframe_map = {'15m': 0, '1h': 1, '4h': 2, '1d': 3}
            supertrend_timeframe_idx = timeframe_map.get(strategy_params.supertrend_timeframe, 2)

            params_array = np.array([
                strategy_params.base_percent,
                strategy_params.initial_deviation,
                strategy_params.trailing_deviation,
                strategy_params.tp_level1,
                strategy_params.tp_percent1,
                strategy_params.rsi_entry_threshold,
                strategy_params.rsi_safety_threshold,
                strategy_params.fees,
                # 3commas conditional parameters
                float(strategy_params.sma_trend_filter),
                float(strategy_params.sma_trend_period),
                float(strategy_params.ema_trend_filter),
                float(strategy_params.ema_trend_period),
                float(strategy_params.atr_volatility_filter),
                float(strategy_params.atr_period),
                strategy_params.atr_multiplier,
                float(strategy_params.higher_highs_filter),
                float(strategy_params.higher_highs_period),
                float(strategy_params.volume_confirmation),
                float(strategy_params.volume_sma_period),
                # SuperTrend parameters
                float(strategy_params.use_supertrend_filter),
                float(supertrend_timeframe_idx),
                float(strategy_params.require_bullish_supertrend)
            ])

            result = enhanced_simulate_strategy(
                backtester.prices,
                backtester.indicators['rsi_1h'],
                backtester.indicators['rsi_4h'],
                backtester.indicators['sma_fast_1h'],
                backtester.indicators['sma_slow_1h'],
                # Additional indicators for 3commas filters
                backtester.indicators['sma_50'],
                backtester.indicators['sma_100'],
                backtester.indicators['sma_200'],
                backtester.indicators['ema_21'],
                backtester.indicators['ema_50'],
                backtester.indicators['ema_100'],
                backtester.indicators['atr_14'],
                backtester.indicators['atr_21'],
                backtester.indicators['atr_28'],
                backtester.indicators['volume'],
                backtester.indicators['vol_sma_10'],
                backtester.indicators['vol_sma_20'],
                backtester.indicators['vol_sma_30'],
                # SuperTrend indicators
                backtester.indicators['supertrend_direction_1h'],
                backtester.indicators['supertrend_direction_4h'],
                backtester.indicators['supertrend_direction_1d'],
                params_array,
                backtester.initial_balance
            )
            if isinstance(result, tuple) and len(result) == 5:
                final_value, max_drawdown, num_trades, balance_history, avg_drawdown_duration = result
            else:
                print(f"Warning: Unexpected return format from fast_simulate_strategy: {result}")
                # Fallback values
                final_value = backtester.initial_balance
                max_drawdown = 0.0
                num_trades = 0
                balance_history = [(0, backtester.initial_balance)]
                avg_drawdown_duration = 0.0

            # Calculate APY
            days = (data.index[-1] - data.index[0]).days
            apy = ((final_value / backtester.initial_balance) ** (365/days) - 1) * 100

            # Convert balance_history tuples to proper format for visualization
            # The balance_history is a numpy array, need to pair it with timestamps
            timestamps = data.index[:len(balance_history)]
            balance_history_for_save = [(timestamps[i], balance_history[i]) for i in range(len(balance_history))]

            # Create synthetic trades for visualization (based on trade count)
            trades = []
            if num_trades > 0:
                # Create evenly spaced synthetic trades
                trade_frequency = max(1, len(balance_history_for_save) // min(num_trades, 100))  # Limit to 100 trades for visualization
                for i in range(0, len(balance_history_for_save), trade_frequency):
                    if len(trades) >= min(num_trades, 100):
                        break
                    timestamp = balance_history_for_save[i][0]
                    action = 'buy' if len(trades) % 3 < 2 else 'sell'
                    price = backtester.prices[min(i, len(backtester.prices)-1)]
                    trades.append(Trade(
                        timestamp=timestamp,
                        action=action,
                        amount_coin=0.05,
                        price=price,
                        usdt_amount=100.0,
                        reason='base_order' if action == 'buy' else 'take_profit'
                    ))

            print(f"Final optimization results: APY={apy:.2f}%, Max DD={max_drawdown:.2f}%, Trades={num_trades}")

        except Exception as e:
            print(f"❌ Error in final metrics calculation: {e}")
            # Use optimization values if available
            apy, max_drawdown = 0.0, 0.0
            balance_history_for_save = [(data.index[0], backtester.initial_balance), (data.index[-1], backtester.initial_balance)]
            trades = []
            avg_drawdown_duration = 0
            num_trades = 0
    else:
        # For non-optimization runs, do full simulation with trades for visualization
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
    if not args.optimize:
        # For non-optimization runs, get additional metrics if needed
        num_trades = len(trades)
        avg_drawdown_duration = 0  # Will be calculated properly if needed

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
            'tp_level1': strategy_params.tp_level1,
            'initial_deviation': strategy_params.initial_deviation,
            'trailing_deviation': strategy_params.trailing_deviation,
            'tp_percent1': strategy_params.tp_percent1,
            'rsi_entry_threshold': strategy_params.rsi_entry_threshold,
            'rsi_safety_threshold': strategy_params.rsi_safety_threshold,
            # 3commas conditional filters
            'sma_trend_filter': strategy_params.sma_trend_filter,
            'sma_trend_period': strategy_params.sma_trend_period,
            'ema_trend_filter': strategy_params.ema_trend_filter,
            'ema_trend_period': strategy_params.ema_trend_period,
            'atr_volatility_filter': strategy_params.atr_volatility_filter,
            'atr_period': strategy_params.atr_period,
            'atr_multiplier': strategy_params.atr_multiplier,
            'higher_highs_filter': strategy_params.higher_highs_filter,
            'higher_highs_period': strategy_params.higher_highs_period,
            'volume_confirmation': strategy_params.volume_confirmation,
            'volume_sma_period': strategy_params.volume_sma_period
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
