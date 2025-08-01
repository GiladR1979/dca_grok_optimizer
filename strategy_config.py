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

    # === TAKE PROFIT LEVELS ===
    tp_level1: List[float] = None             # First TP level %
    tp_percent1: List[float] = None           # % to sell at TP1
    tp_percent2: List[float] = None           # % to sell at TP2
    tp_percent3: List[float] = None           # % to sell at TP3

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
    supertrend_timeframe: List[str] = None     # Timeframe for SuperTrend ('1h', '4h', '1d')
    supertrend_period: List[int] = None        # SuperTrend ATR period
    supertrend_multiplier: List[float] = None  # SuperTrend multiplier
    require_bullish_supertrend: List[bool] = None  # Only enter when SuperTrend is bullish
    max_acceptable_drawdown: float = 15.0      # Maximum acceptable drawdown % (fixed)

    # === FIXED PARAMETERS ===
    fees: float = 0.075                       # Trading fees %

    def __post_init__(self):
        """Set default ranges if not provided"""
        if self.base_percent is None:
            self.base_percent = [1.0, 1.5, 2.0, 2.5, 3.0]

        if self.volume_multiplier is None:
            self.volume_multiplier = [1.5]

        if self.initial_deviation is None:
            self.initial_deviation = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        if self.step_multiplier is None:
            self.step_multiplier = [1.2, 1.3, 1.4, 1.5]

        if self.max_safeties is None:
            self.max_safeties = [8]

        if self.tp_level1 is None:
            self.tp_level1 = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]

        if self.tp_percent1 is None:
            self.tp_percent1 = [50]

        if self.tp_percent2 is None:
            self.tp_percent2 = [30]

        if self.tp_percent3 is None:
            self.tp_percent3 = [20]

        if self.trailing_deviation is None:
            self.trailing_deviation = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        if self.rsi_entry_threshold is None:
            self.rsi_entry_threshold = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]

        if self.rsi_safety_threshold is None:
            self.rsi_safety_threshold = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]

        if self.rsi_exit_threshold is None:
            self.rsi_exit_threshold = [60.0, 65.0, 70.0, 75.0, 80.0]

        # 3commas conditional trade start conditions - LESS RESTRICTIVE for better APY
        if self.sma_trend_filter is None:
            self.sma_trend_filter = [False, False, False, True]  # 75% chance disabled

        if self.sma_trend_period is None:
            self.sma_trend_period = [50, 100]  # Removed 200 (too restrictive)

        if self.ema_trend_filter is None:
            self.ema_trend_filter = [False, False, False, True]  # 75% chance disabled

        if self.ema_trend_period is None:
            self.ema_trend_period = [21, 50]  # Removed 100 (too restrictive)

        if self.atr_volatility_filter is None:
            self.atr_volatility_filter = [False, False, True]  # 67% chance disabled

        if self.atr_period is None:
            self.atr_period = [14, 21]  # Removed 28 (too restrictive)

        if self.atr_multiplier is None:
            self.atr_multiplier = [2.0, 2.5, 3.0, 4.0, 5.0]  # Higher multipliers = less restrictive

        if self.higher_highs_filter is None:
            self.higher_highs_filter = [False, False, True]  # 67% chance disabled

        if self.higher_highs_period is None:
            self.higher_highs_period = [10, 20]  # Removed 30 (too restrictive)

        if self.volume_confirmation is None:
            self.volume_confirmation = [False, False, True]  # 67% chance disabled

        if self.volume_sma_period is None:
            self.volume_sma_period = [10, 20]  # Removed 30 (too restrictive)
        
        # SuperTrend drawdown elimination settings
        if self.use_supertrend_filter is None:
            self.use_supertrend_filter = [True, False]  # 50% chance enabled for testing
        
        if self.supertrend_timeframe is None:
            self.supertrend_timeframe = ['1h', '4h', '1d']
        
        if self.supertrend_period is None:
            self.supertrend_period = [7, 10, 14, 21]  # ATR periods to test
        
        if self.supertrend_multiplier is None:
            self.supertrend_multiplier = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Multipliers to test
        
        if self.require_bullish_supertrend is None:
            self.require_bullish_supertrend = [True, False]  # Test both modes


@dataclass
class StrategyParams:
    """Strategy parameters for optimization - centralized version"""

    # === POSITION SIZING ===
    base_percent: float = 1.0                 # Base order as % of balance
    volume_multiplier: float = 1.2           # Safety order volume scaling

    # === ENTRY CONDITIONS ===
    initial_deviation: float = 3.0           # First safety order deviation %
    step_multiplier: float = 1.5             # Safety order step scaling
    max_safeties: int = 8                    # Maximum safety orders

    # === TAKE PROFIT LEVELS ===
    tp_level1: float = 3.0                   # First TP level %
    tp_percent1: float = 50.0                # % to sell at TP1
    tp_percent2: float = 30.0                # % to sell at TP2
    tp_percent3: float = 20.0                # % to sell at TP3

    # === TRAILING STOP ===
    trailing_deviation: float = 3.0          # Trailing stop %

    # === RSI CONDITIONS ===
    rsi_entry_threshold: float = 40.0        # RSI entry threshold
    rsi_safety_threshold: float = 30.0       # RSI safety threshold
    rsi_exit_threshold: float = 70.0         # RSI exit threshold

    # === 3COMMAS CONDITIONAL TRADE START CONDITIONS ===
    # Market trend filters (available on 3commas)
    sma_trend_filter: bool = False           # Require price > SMA for trend confirmation
    sma_trend_period: int = 200              # SMA period for trend filter (50, 100, 200)
    ema_trend_filter: bool = False           # Require price > EMA for trend confirmation  
    ema_trend_period: int = 50               # EMA period for trend filter (21, 50, 100)
    
    # Volatility filters (available on 3commas)
    atr_volatility_filter: bool = False      # Enable ATR volatility filter
    atr_period: int = 14                     # ATR calculation period
    atr_multiplier: float = 1.5              # ATR threshold multiplier
    
    # Market structure filters (available on 3commas)
    higher_highs_filter: bool = False        # Require recent higher highs
    higher_highs_period: int = 20            # Period to check for higher highs
    
    # Volume confirmation (available on 3commas)
    volume_confirmation: bool = False        # Require volume confirmation
    volume_sma_period: int = 20              # Volume SMA period for comparison
    
    # === SUPERTREND DRAWDOWN ELIMINATION (NEW) ===
    use_supertrend_filter: bool = False      # Enable SuperTrend trend filtering for drawdown elimination
    supertrend_timeframe: str = '4h'         # Timeframe for SuperTrend analysis ('1h', '4h', '1d')
    supertrend_period: int = 10              # SuperTrend ATR period (optimizable 7-21)
    supertrend_multiplier: float = 3.0       # SuperTrend multiplier (optimizable 1.0-5.0)
    require_bullish_supertrend: bool = True  # Only enter when SuperTrend is bullish
    max_acceptable_drawdown: float = 15.0    # Maximum acceptable drawdown % (stop optimization if exceeded)

    # === FIXED PARAMETERS ===
    fees: float = 0.075                      # Trading fees %

    def __post_init__(self):
        """Ensure trailing deviation doesn't exceed TP1"""
        if self.trailing_deviation > self.tp_level1:
            self.trailing_deviation = self.tp_level1

    @property
    def tp_level2(self) -> float:
        return self.tp_level1 * 2

    @property
    def tp_level3(self) -> float:
        return self.tp_level1 * 3


class StrategyPresets:
    """Pre-defined strategy configurations for different market conditions"""

    @staticmethod
    def conservative() -> StrategyParams:
        """Conservative strategy for risk management"""
        return StrategyParams(
            base_percent=1.0,
            initial_deviation=3.0,
            tp_level1=3.0,
            tp_percent1=50.0,
            rsi_entry_threshold=35.0,
            trailing_deviation=2.0
        )

    @staticmethod
    def aggressive() -> StrategyParams:
        """Aggressive strategy for bull markets"""
        return StrategyParams(
            base_percent=5.0,
            initial_deviation=2.0,
            tp_level1=5.0,
            tp_percent1=25.0,
            tp_percent2=25.0,
            tp_percent3=25.0,  # Keep 25% running
            rsi_entry_threshold=55.0,
            trailing_deviation=3.0
        )

    @staticmethod
    def bull_market() -> StrategyParams:
        """Optimized for trending bull markets"""
        return StrategyParams(
            base_percent=3.0,
            initial_deviation=1.5,
            tp_level1=8.0,      # Let profits run longer
            tp_percent1=20.0,   # Sell less on TPs
            tp_percent2=20.0,
            tp_percent3=30.0,   # Keep 30% forever
            rsi_entry_threshold=60.0,
            rsi_safety_threshold=50.0,
            trailing_deviation=4.0
        )

    @staticmethod
    def bear_market() -> StrategyParams:
        """Optimized for bear/sideways markets"""
        return StrategyParams(
            base_percent=2.0,
            initial_deviation=2.5,
            tp_level1=2.0,      # Take profits quickly
            tp_percent1=60.0,   # Sell more on TPs
            tp_percent2=35.0,
            tp_percent3=5.0,    # Keep very little
            rsi_entry_threshold=40.0,
            rsi_safety_threshold=25.0,
            trailing_deviation=1.5
        )

    @staticmethod
    def scalping() -> StrategyParams:
        """High-frequency trading strategy"""
        return StrategyParams(
            base_percent=1.5,
            initial_deviation=1.0,
            tp_level1=1.0,      # Quick small profits
            tp_percent1=70.0,
            tp_percent2=25.0,
            tp_percent3=5.0,
            rsi_entry_threshold=50.0,
            rsi_safety_threshold=40.0,
            trailing_deviation=0.5
        )


class OptimizationConfig:
    """Centralized optimization configuration"""

    def __init__(self, ranges: OptimizationRanges = None):
        self.ranges = ranges or OptimizationRanges()

    def suggest_params(self, trial: optuna.Trial) -> StrategyParams:
        """Suggest parameters using Optuna trial with centralized ranges"""

        # Calculate dependent variables first
        tp_level1 = trial.suggest_categorical('tp_level1', self.ranges.tp_level1)
        trailing_deviation = trial.suggest_categorical('trailing_deviation', self.ranges.trailing_deviation)

        # Ensure trailing doesn't exceed TP1
        effective_trailing = min(trailing_deviation, tp_level1)

        return StrategyParams(
            # Position sizing
            base_percent=trial.suggest_categorical('base_percent', self.ranges.base_percent),
            volume_multiplier=trial.suggest_categorical('volume_multiplier', self.ranges.volume_multiplier),

            # Entry conditions
            initial_deviation=trial.suggest_categorical('initial_deviation', self.ranges.initial_deviation),
            step_multiplier=trial.suggest_categorical('step_multiplier', self.ranges.step_multiplier),
            max_safeties=trial.suggest_categorical('max_safeties', self.ranges.max_safeties),

            # Take profits
            tp_level1=tp_level1,
            tp_percent1=trial.suggest_categorical('tp_percent1', self.ranges.tp_percent1),
            tp_percent2=trial.suggest_categorical('tp_percent2', self.ranges.tp_percent2),
            tp_percent3=trial.suggest_categorical('tp_percent3', self.ranges.tp_percent3),

            # Trailing stop
            trailing_deviation=effective_trailing,

            # RSI conditions
            rsi_entry_threshold=trial.suggest_categorical('rsi_entry_threshold', self.ranges.rsi_entry_threshold),
            rsi_safety_threshold=trial.suggest_categorical('rsi_safety_threshold', self.ranges.rsi_safety_threshold),
            rsi_exit_threshold=trial.suggest_categorical('rsi_exit_threshold', self.ranges.rsi_exit_threshold),

            # 3commas conditional trade start conditions
            sma_trend_filter=trial.suggest_categorical('sma_trend_filter', self.ranges.sma_trend_filter),
            sma_trend_period=trial.suggest_categorical('sma_trend_period', self.ranges.sma_trend_period),
            ema_trend_filter=trial.suggest_categorical('ema_trend_filter', self.ranges.ema_trend_filter),
            ema_trend_period=trial.suggest_categorical('ema_trend_period', self.ranges.ema_trend_period),
            atr_volatility_filter=trial.suggest_categorical('atr_volatility_filter', self.ranges.atr_volatility_filter),
            atr_period=trial.suggest_categorical('atr_period', self.ranges.atr_period),
            atr_multiplier=trial.suggest_categorical('atr_multiplier', self.ranges.atr_multiplier),
            higher_highs_filter=trial.suggest_categorical('higher_highs_filter', self.ranges.higher_highs_filter),
            higher_highs_period=trial.suggest_categorical('higher_highs_period', self.ranges.higher_highs_period),
            volume_confirmation=trial.suggest_categorical('volume_confirmation', self.ranges.volume_confirmation),
            volume_sma_period=trial.suggest_categorical('volume_sma_period', self.ranges.volume_sma_period),
            
            # SuperTrend drawdown elimination
            use_supertrend_filter=trial.suggest_categorical('use_supertrend_filter', self.ranges.use_supertrend_filter),
            supertrend_timeframe=trial.suggest_categorical('supertrend_timeframe', self.ranges.supertrend_timeframe),
            supertrend_period=trial.suggest_categorical('supertrend_period', self.ranges.supertrend_period),
            supertrend_multiplier=trial.suggest_categorical('supertrend_multiplier', self.ranges.supertrend_multiplier),
            require_bullish_supertrend=trial.suggest_categorical('require_bullish_supertrend', self.ranges.require_bullish_supertrend),
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
            base_percent=[2.0, 3.0, 4.0, 5.0, 7.5, 10.0],           # Larger positions
            tp_level1=[3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0],        # Higher TPs
            tp_percent1=[15.0, 20.0, 25.0, 30.0, 35.0],             # Sell less
            tp_percent2=[15.0, 20.0, 25.0, 30.0],
            tp_percent3=[20.0, 25.0, 30.0, 40.0],                   # Keep more
            rsi_entry_threshold=[45.0, 50.0, 55.0, 60.0, 65.0],     # More entries
            trailing_deviation=[2.0, 3.0, 4.0, 5.0]                 # Wider stops
        )

    @staticmethod
    def bear_market() -> OptimizationRanges:
        """Ranges optimized for bear markets"""
        return OptimizationRanges(
            base_percent=[0.5, 1.0, 1.5, 2.0, 2.5],                 # Smaller positions
            tp_level1=[1.0, 1.5, 2.0, 2.5, 3.0],                    # Lower TPs
            tp_percent1=[40.0, 50.0, 60.0, 70.0],                   # Sell more
            tp_percent2=[25.0, 30.0, 35.0, 40.0],
            tp_percent3=[10.0, 15.0, 20.0],                         # Keep less
            rsi_entry_threshold=[25.0, 30.0, 35.0, 40.0, 45.0],     # Conservative entries
            trailing_deviation=[0.5, 1.0, 1.5, 2.0]                 # Tighter stops
        )

    @staticmethod
    def sideways_market() -> OptimizationRanges:
        """Ranges optimized for sideways markets"""
        return OptimizationRanges(
            base_percent=[1.0, 1.5, 2.0, 2.5, 3.0],
            tp_level1=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            tp_percent1=[30.0, 40.0, 50.0, 60.0],
            rsi_entry_threshold=[35.0, 40.0, 45.0, 50.0, 55.0],
            trailing_deviation=[1.0, 1.5, 2.0, 2.5, 3.0]
        )


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    print("=== Strategy Configuration Examples ===\n")

    # Default parameters
    default = StrategyParams()
    print(f"Default Strategy:")
    print(f"  Base %: {default.base_percent}%")
    print(f"  TP1: {default.tp_level1}% (sell {default.tp_percent1}%)")
    print(f"  RSI Entry: < {default.rsi_entry_threshold}")
    print()

    # Bull market preset
    bull = StrategyPresets.bull_market()
    print(f"Bull Market Strategy:")
    print(f"  Base %: {bull.base_percent}%")
    print(f"  TP1: {bull.tp_level1}% (sell {bull.tp_percent1}%)")
    print(f"  RSI Entry: < {bull.rsi_entry_threshold}")
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
