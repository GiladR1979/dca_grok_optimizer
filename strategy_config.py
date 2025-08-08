#!/usr/bin/env python3
"""
Centralized Strategy Configuration
All customizable optimization variables in one place
"""

from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
import optuna


@dataclass
class OptimizationRanges:
    """Define optimization ranges for all customizable parameters"""

    # === POSITION SIZING ===
    base_percent: List[float] = None           # Base order as % of balance
    volume_multiplier: List[float] = None      # Safety order volume scaling

    # === ENTRY CONDITIONS ===
    initial_deviation: List[float] = None      # First safety order deviation %
    step_multiplier: List[float] = None        # Safety order step scaling
    max_safeties: List[int] = None            # Maximum safety orders

    # === TAKE PROFIT LEVELS === (Single TP target only)
    tp_level1: List[float] = None             # TP level %
    tp_percent1: List[float] = None           # % to sell at TP (typically 100%)

    # === TRAILING STOP ===
    trailing_deviation: List[float] = None     # Trailing stop %

    # === RSI CONDITIONS ===
    rsi_entry_threshold: List[float] = None    # RSI entry threshold
    rsi_safety_threshold: List[float] = None   # RSI safety threshold
    rsi_exit_threshold: List[float] = None     # RSI exit threshold

    # === 3COMMAS CONDITIONAL TRADE START CONDITIONS ===
    # Market trend filters (available on 3commas)
    sma_trend_filter: List[bool] = None        # Require price > SMA for trend confirmation
    sma_trend_period: List[int] = None         # SMA period for trend filter (50, 100, 200)
    ema_trend_filter: List[bool] = None        # Require price > EMA for trend confirmation
    ema_trend_period: List[int] = None         # EMA period for trend filter (21, 50, 100)

    # Volatility filters (available on 3commas)
    atr_volatility_filter: List[bool] = None   # Enable ATR volatility filter
    atr_period: List[int] = None               # ATR calculation period
    atr_multiplier: List[float] = None         # ATR threshold multiplier

    # Market structure filters (available on 3commas)
    higher_highs_filter: List[bool] = None     # Require recent higher highs
    higher_highs_period: List[int] = None      # Period to check for higher highs

    # Volume confirmation (available on 3commas)
    volume_confirmation: List[bool] = None     # Require volume confirmation
    volume_sma_period: List[int] = None        # Volume SMA period for comparison

    # === SUPERTREND DRAWDOWN ELIMINATION (NEW) ===
    use_supertrend_filter: List[bool] = None   # Enable SuperTrend trend filtering
    supertrend_timeframe: List[str] = None     # Timeframe for SuperTrend ('15m', '30m', '1h', '4h', '1d')
    supertrend_period: List[int] = None        # SuperTrend ATR period
    supertrend_multiplier: List[float] = None  # SuperTrend multiplier
    require_bullish_supertrend: List[bool] = None  # Only enter when SuperTrend is bullish
    max_acceptable_drawdown: float = 15.0      # Maximum acceptable drawdown % (fixed)

    # === FIXED PARAMETERS ===
    fees: float = 0.075                       # Trading fees %

    def __post_init__(self):
        """Set default ranges if not provided"""
        if self.base_percent is None:
            # FIXED: Limited to ensure total position never exceeds 100%
            # With max_safeties=8 and volume_multiplier=1.5, max base is 1.33%
            self.base_percent = [1.33] # DO NOT CHANGE

        if self.volume_multiplier is None:
            self.volume_multiplier = [1.5] # DO NOT CHANGE

        if self.initial_deviation is None:
            self.initial_deviation = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        if self.step_multiplier is None:
            self.step_multiplier = [1, 1.1, 1.2, 1.3, 1.4, 1.5]

        if self.max_safeties is None:
            self.max_safeties = [8]

        if self.tp_level1 is None:
            self.tp_level1 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

        if self.tp_percent1 is None:
            self.tp_percent1 = [100]  # Single TP target - sell entire position

        if self.trailing_deviation is None:
            self.trailing_deviation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 19.8]

        if self.rsi_entry_threshold is None:
            self.rsi_entry_threshold = [50.0]  # Fixed, not used

        if self.rsi_safety_threshold is None:
            self.rsi_safety_threshold = [50.0]  # Fixed, not used

        if self.rsi_exit_threshold is None:
            self.rsi_exit_threshold = [50.0]  # Fixed, not used

        # 3commas conditional trade start conditions - DISABLED
        if self.sma_trend_filter is None:
            self.sma_trend_filter = [False]

        if self.sma_trend_period is None:
            self.sma_trend_period = [50]

        if self.ema_trend_filter is None:
            self.ema_trend_filter = [False]

        if self.ema_trend_period is None:
            self.ema_trend_period = [21]

        if self.atr_volatility_filter is None:
            self.atr_volatility_filter = [False]

        if self.atr_period is None:
            self.atr_period = [14]

        if self.atr_multiplier is None:
            self.atr_multiplier = [2.0]

        if self.higher_highs_filter is None:
            self.higher_highs_filter = [False]

        if self.higher_highs_period is None:
            self.higher_highs_period = [10]

        if self.volume_confirmation is None:
            self.volume_confirmation = [False]

        if self.volume_sma_period is None:
            self.volume_sma_period = [10]

        # SuperTrend drawdown elimination settings
        if self.use_supertrend_filter is None:
            self.use_supertrend_filter = [True]  # Always enabled

        if self.supertrend_timeframe is None:
            self.supertrend_timeframe = ['15m', '30m', '1h', '4h', '1d']

        if self.supertrend_period is None:
            self.supertrend_period = [7, 10, 14, 21]  # ATR periods to test

        if self.supertrend_multiplier is None:
            self.supertrend_multiplier = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Multipliers to test

        if self.require_bullish_supertrend is None:
            self.require_bullish_supertrend = [False]  # Allow both directions


@dataclass
class StrategyParams:
    """Strategy parameters for optimization - centralized version"""

    # === POSITION SIZING (SEPARATE FOR LONG/SHORT) ===
    base_percent_long: float = 1.33                 # Base order as % of balance for longs
    base_percent_short: float = 1.33                # Base order as % of balance for shorts
    volume_multiplier_long: float = 1.5            # Safety order volume scaling for longs
    volume_multiplier_short: float = 1.5           # Safety order volume scaling for shorts

    # === ENTRY CONDITIONS (SEPARATE FOR LONG/SHORT) ===
    initial_deviation_long: float = 2.0            # First safety order deviation % for longs
    initial_deviation_short: float = 2.0           # First safety order deviation % for shorts
    step_multiplier_long: float = 1.5              # Safety order step scaling for longs
    step_multiplier_short: float = 1.5             # Safety order step scaling for shorts
    max_safeties_long: int = 8                     # Maximum safety orders for longs
    max_safeties_short: int = 8                    # Maximum safety orders for shorts

    # === TAKE PROFIT LEVELS === (Single TP target only, SEPARATE FOR LONG/SHORT)
    tp_level1_long: float = 3.0                    # TP level % for longs
    tp_level1_short: float = 3.0                   # TP level % for shorts
    tp_percent1_long: float = 100.0                # % to sell at TP for longs (100% = close entire position)
    tp_percent1_short: float = 100.0               # % to sell at TP for shorts

    # === TRAILING STOP (SEPARATE FOR LONG/SHORT) ===
    trailing_deviation_long: float = 3.0           # Trailing stop % for longs
    trailing_deviation_short: float = 3.0          # Trailing stop % for shorts

    # === RSI CONDITIONS ===
    rsi_entry_threshold: float = 50.0        # Not used
    rsi_safety_threshold: float = 50.0       # Not used
    rsi_exit_threshold: float = 50.0         # Not used

    # === 3COMMAS CONDITIONAL TRADE START CONDITIONS ===
    # Market trend filters (available on 3commas)
    sma_trend_filter: bool = False           # Disabled
    sma_trend_period: int = 200              # Not used
    ema_trend_filter: bool = False           # Disabled
    ema_trend_period: int = 50               # Not used

    # Volatility filters (available on 3commas)
    atr_volatility_filter: bool = False      # Disabled
    atr_period: int = 14                     # Not used
    atr_multiplier: float = 1.5              # Not used

    # Market structure filters (available on 3commas)
    higher_highs_filter: bool = False        # Disabled
    higher_highs_period: int = 20            # Not used

    # Volume confirmation (available on 3commas)
    volume_confirmation: bool = False        # Disabled
    volume_sma_period: int = 20              # Not used

    # === SUPERTREND DRAWDOWN ELIMINATION (SHARED) ===
    use_supertrend_filter: bool = True       # Always enabled
    supertrend_timeframe: str = '4h'         # Timeframe for SuperTrend analysis ('15m', '30m', '1h', '4h', '1d')
    supertrend_period: int = 10              # SuperTrend ATR period (optimizable 7-21)
    supertrend_multiplier: float = 3.0       # SuperTrend multiplier (optimizable 1.0-5.0)
    require_bullish_supertrend: bool = False # Allow both directions
    max_acceptable_drawdown: float = 15.0    # Maximum acceptable drawdown % (stop optimization if exceeded)

    # === FIXED PARAMETERS ===
    fees: float = 0.075                      # Trading fees %

    def __post_init__(self):
        """Ensure trailing deviation doesn't exceed TP1 for both directions"""
        if self.trailing_deviation_long > self.tp_level1_long:
            self.trailing_deviation_long = self.tp_level1_long
        if self.trailing_deviation_short > self.tp_level1_short:
            self.trailing_deviation_short = self.tp_level1_short

    # Removed tp_level2 and tp_level3 properties - using single TP target only


class StrategyPresets:
    """Pre-defined strategy configurations for different market conditions"""

    @staticmethod
    def conservative() -> StrategyParams:
        """Conservative strategy for risk management"""
        return StrategyParams(
            base_percent_long=1.0,
            base_percent_short=1.0,
            initial_deviation_long=3.0,
            initial_deviation_short=3.0,
            tp_level1_long=3.0,
            tp_level1_short=3.0,
            tp_percent1_long=100.0,  # Single TP - sell all
            tp_percent1_short=100.0,
            trailing_deviation_long=2.0,
            trailing_deviation_short=2.0
        )

    @staticmethod
    def aggressive() -> StrategyParams:
        """Aggressive strategy for bull markets"""
        return StrategyParams(
            base_percent_long=1.33,
            base_percent_short=1.33,
            initial_deviation_long=2.0,
            initial_deviation_short=2.0,
            step_multiplier_long=1.5,
            step_multiplier_short=1.5,
            volume_multiplier_long=1.5,
            volume_multiplier_short=1.5,
            max_safeties_long=8,
            max_safeties_short=8,
            tp_level1_long=5.0,
            tp_level1_short=5.0,
            tp_percent1_long=100.0,  # Single TP - sell all
            tp_percent1_short=100.0,
            trailing_deviation_long=3.0,
            trailing_deviation_short=3.0
        )

    @staticmethod
    def bull_market() -> StrategyParams:
        """Optimized for trending bull markets"""
        return StrategyParams(
            base_percent_long=1.33,
            base_percent_short=1.33,
            initial_deviation_long=1.5,
            initial_deviation_short=1.5,
            tp_level1_long=8.0,      # Let profits run longer
            tp_level1_short=8.0,
            tp_percent1_long=100.0,  # Single TP - sell all
            tp_percent1_short=100.0,
            trailing_deviation_long=4.0,
            trailing_deviation_short=4.0
        )

    @staticmethod
    def bear_market() -> StrategyParams:
        """Optimized for bear/sideways markets"""
        return StrategyParams(
            base_percent_long=1.33,
            base_percent_short=1.33,
            initial_deviation_long=2.5,
            initial_deviation_short=2.5,
            tp_level1_long=2.0,      # Take profits quickly
            tp_level1_short=2.0,
            tp_percent1_long=100.0,  # Single TP - sell all
            tp_percent1_short=100.0,
            trailing_deviation_long=1.5,
            trailing_deviation_short=1.5
        )

    @staticmethod
    def scalping() -> StrategyParams:
        """High-frequency trading strategy"""
        return StrategyParams(
            base_percent_long=1.33,
            base_percent_short=1.33,
            initial_deviation_long=1.0,
            initial_deviation_short=1.0,
            tp_level1_long=1.0,      # Quick small profits
            tp_level1_short=1.0,
            tp_percent1_long=100.0,  # Single TP - sell all
            tp_percent1_short=100.0,
            trailing_deviation_long=0.5,
            trailing_deviation_short=0.5
        )


class OptimizationConfig:
    """Centralized optimization configuration"""

    def __init__(self, ranges: OptimizationRanges = None):
        self.ranges = ranges or OptimizationRanges()

    def suggest_params(self, trial: optuna.Trial) -> StrategyParams:
        """Suggest parameters using Optuna trial with centralized ranges"""

        # Suggest separate values for long and short DCA params
        tp_level1_long = trial.suggest_categorical('tp_level1_long', self.ranges.tp_level1)
        trailing_deviation_long = trial.suggest_categorical('trailing_deviation_long', self.ranges.trailing_deviation)
        effective_trailing_long = min(trailing_deviation_long, tp_level1_long)

        tp_level1_short = trial.suggest_categorical('tp_level1_short', self.ranges.tp_level1)
        trailing_deviation_short = trial.suggest_categorical('trailing_deviation_short', self.ranges.trailing_deviation)
        effective_trailing_short = min(trailing_deviation_short, tp_level1_short)

        return StrategyParams(
            # Position sizing (separate for long/short)
            base_percent_long=trial.suggest_categorical('base_percent_long', self.ranges.base_percent),
            base_percent_short=trial.suggest_categorical('base_percent_short', self.ranges.base_percent),
            volume_multiplier_long=trial.suggest_categorical('volume_multiplier_long', self.ranges.volume_multiplier),
            volume_multiplier_short=trial.suggest_categorical('volume_multiplier_short', self.ranges.volume_multiplier),

            # Entry conditions (separate for long/short)
            initial_deviation_long=trial.suggest_categorical('initial_deviation_long', self.ranges.initial_deviation),
            initial_deviation_short=trial.suggest_categorical('initial_deviation_short', self.ranges.initial_deviation),
            step_multiplier_long=trial.suggest_categorical('step_multiplier_long', self.ranges.step_multiplier),
            step_multiplier_short=trial.suggest_categorical('step_multiplier_short', self.ranges.step_multiplier),
            max_safeties_long=trial.suggest_categorical('max_safeties_long', self.ranges.max_safeties),
            max_safeties_short=trial.suggest_categorical('max_safeties_short', self.ranges.max_safeties),

            # Take profits (single TP target, separate for long/short)
            tp_level1_long=tp_level1_long,
            tp_level1_short=tp_level1_short,
            tp_percent1_long=trial.suggest_categorical('tp_percent1_long', self.ranges.tp_percent1),
            tp_percent1_short=trial.suggest_categorical('tp_percent1_short', self.ranges.tp_percent1),

            # Trailing stop (separate for long/short)
            trailing_deviation_long=effective_trailing_long,
            trailing_deviation_short=effective_trailing_short,

            # RSI conditions (fixed, not used)
            rsi_entry_threshold=50.0,
            rsi_safety_threshold=50.0,
            rsi_exit_threshold=50.0,

            # 3commas conditional trade start conditions (disabled)
            sma_trend_filter=False,
            sma_trend_period=200,
            ema_trend_filter=False,
            ema_trend_period=50,
            atr_volatility_filter=False,
            atr_period=14,
            atr_multiplier=1.5,
            higher_highs_filter=False,
            higher_highs_period=20,
            volume_confirmation=False,
            volume_sma_period=20,

            # SuperTrend drawdown elimination (shared)
            use_supertrend_filter=True,
            supertrend_timeframe=trial.suggest_categorical('supertrend_timeframe', self.ranges.supertrend_timeframe),
            supertrend_period=trial.suggest_categorical('supertrend_period', self.ranges.supertrend_period),
            supertrend_multiplier=trial.suggest_categorical('supertrend_multiplier', self.ranges.supertrend_multiplier),
            require_bullish_supertrend=False,
            max_acceptable_drawdown=self.ranges.max_acceptable_drawdown,

            # Fixed
            fees=self.ranges.fees
        )

    def get_custom_ranges(self, **kwargs) -> 'OptimizationConfig':
        """Create custom optimization config with overridden ranges"""
        custom_ranges = OptimizationRanges(**kwargs)
        return OptimizationConfig(custom_ranges)


# === MARKET-SPECIFIC OPTIMIZATION RANGES ===

class MarketOptimizationRanges:
    """Pre-defined optimization ranges for different market conditions"""

    @staticmethod
    def bull_market() -> OptimizationRanges:
        """Ranges optimized for bull markets"""
        return OptimizationRanges(
            base_percent=[1.33],
            volume_multiplier=[1.5],
            max_safeties=[8],
            tp_level1=[3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],
            tp_percent1=[100.0],
            trailing_deviation=[2.0, 3.0, 4.0, 5.0]
        )

    @staticmethod
    def bear_market() -> OptimizationRanges:
        """Ranges optimized for bear markets"""
        return OptimizationRanges(
            base_percent=[1.33],
            volume_multiplier=[1.5],
            max_safeties=[8],
            tp_level1=[1.0, 1.5, 2.0, 2.5, 3.0],
            tp_percent1=[100.0],
            trailing_deviation=[0.5, 1.0, 1.5, 2.0]
        )

    @staticmethod
    def sideways_market() -> OptimizationRanges:
        """Ranges optimized for sideways markets"""
        return OptimizationRanges(
            base_percent=[1.33],
            volume_multiplier=[1.5],
            max_safeties=[8],
            tp_level1=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            tp_percent1=[100.0],
            trailing_deviation=[1.0, 1.5, 2.0, 2.5, 3.0]
        )


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    print("=== Strategy Configuration Examples ===\n")

    # Default parameters
    default = StrategyParams()
    print(f"Default Strategy:")
    print(f"  Base % (Long): {default.base_percent_long}%")
    print(f"  Base % (Short): {default.base_percent_short}%")
    print(f"  TP1 (Long): {default.tp_level1_long}% (sell {default.tp_percent1_long}%)")
    print(f"  TP1 (Short): {default.tp_level1_short}% (sell {default.tp_percent1_short}%)")
    print()

    # Bull market preset
    bull = StrategyPresets.bull_market()
    print(f"Bull Market Strategy:")
    print(f"  Base % (Long): {bull.base_percent_long}%")
    print(f"  Base % (Short): {bull.base_percent_short}%")
    print(f"  TP1 (Long): {bull.tp_level1_long}% (sell {bull.tp_percent1_long}%)")
    print(f"  TP1 (Short): {bull.tp_level1_short}% (sell {bull.tp_percent1_short}%)")
    print()

    # Optimization ranges
    ranges = OptimizationRanges()
    print(f"Default Optimization Ranges:")
    print(f"  Base %: {ranges.base_percent}")
    print(f"  TP1: {ranges.tp_level1}")
    print(f"  Initial Dev: {ranges.initial_deviation}")
    print()

    # Bull market optimization ranges
    bull_ranges = MarketOptimizationRanges.bull_market()
    print(f"Bull Market Ranges:")
    print(f"  Base %: {bull_ranges.base_percent}")
    print(f"  TP1: {bull_ranges.tp_level1}")
    print(f"  TP1 Sell %: {bull_ranges.tp_percent1}")
