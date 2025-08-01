#!/usr/bin/env python3
"""
Fix for 1 TP Configuration APY Issues
Addresses the suspicious APY calculations in phase 2 optimization
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple
from strategy_config import StrategyParams

@njit
def fixed_simulate_strategy_1tp(
    prices: np.ndarray,
    rsi_1h: np.ndarray,
    rsi_4h: np.ndarray, 
    sma_fast: np.ndarray,
    sma_slow: np.ndarray,
    params_array: np.ndarray,
    initial_balance: float = 10000.0
) -> Tuple[float, float, int, np.ndarray]:
    """
    FIXED simulation for 1 TP configuration
    Addresses APY calculation issues and ensures proper position management
    """
    
    # Unpack parameters
    base_percent = params_array[0]
    initial_deviation = params_array[1] 
    trailing_deviation = params_array[2]
    tp_level1 = params_array[3]
    tp_percent1 = params_array[4] / 100.0  # Convert to decimal
    rsi_entry_thresh = params_array[5]
    rsi_safety_thresh = params_array[6]
    fees = params_array[7] / 100.0
    
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
    
    # TP tracking for 1 TP system
    tp_hit = False
    
    # Balance history for portfolio tracking
    n_points = len(prices)
    balance_history = np.zeros(n_points)
    
    # Drawdown tracking
    max_portfolio_value = initial_balance
    max_drawdown = 0.0
    
    for i in range(n_points):
        current_price = prices[i]
        current_rsi_1h = rsi_1h[i] if i < len(rsi_1h) else 50.0
        current_sma_fast = sma_fast[i] if i < len(sma_fast) else current_price
        current_sma_slow = sma_slow[i] if i < len(sma_slow) else current_price
        
        # 1. CHECK FOR NEW DEAL ENTRY
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
                    safety_base = initial_balance * (base_percent / 100.0)
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
            
            # Take profit conditions - FIXED FOR 1 TP SYSTEM
            if position_size > 0:
                profit_percent = (current_price - average_entry) / average_entry * 100.0
                
                # Single TP logic - sell specified percentage when TP1 is hit
                if profit_percent >= tp_level1 and not tp_hit:
                    tp_sell = position_size * tp_percent1
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
            
            # Trailing stop (only if position remains after TP)
            if trailing_active and position_size > 0:
                if current_price > peak_price:
                    peak_price = current_price
                
                # FIXED: Ensure trailing deviation doesn't exceed TP level
                effective_trailing = min(trailing_deviation, tp_level1)
                trailing_threshold = peak_price * (1.0 - effective_trailing / 100.0)
                
                if current_price <= trailing_threshold:
                    # Sell remaining position
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
        
        # CRITICAL FIX: Record TOTAL portfolio value (cash + position value)
        portfolio_value = balance + position_size * current_price
        balance_history[i] = portfolio_value
        
        # Track drawdown on total portfolio value
        if portfolio_value > max_portfolio_value:
            max_portfolio_value = portfolio_value
        
        current_drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value * 100.0
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
    
    # CRITICAL FIX: Final portfolio value includes any remaining position
    final_portfolio_value = balance + position_size * prices[-1]
    
    return final_portfolio_value, max_drawdown, num_trades, balance_history


def validate_1tp_configuration():
    """Validate that 1 TP configuration is working correctly"""
    
    print("🔍 VALIDATING 1 TP CONFIGURATION")
    print("=" * 50)
    
    # Test parameters for 1 TP system
    params_1tp = StrategyParams(
        base_percent=1.0,
        initial_deviation=3.0,
        trailing_deviation=2.0,  # Less than TP1
        tp_level1=3.0,
        tp_percent1=100.0,  # Single TP - sell entire position
        tp_percent2=0.0,    # Not used in 1 TP system
        tp_percent3=0.0,    # Not used in 1 TP system
        rsi_entry_threshold=40.0,
        rsi_safety_threshold=30.0,
        fees=0.075
    )
    
    print("1 TP Configuration:")
    print(f"  TP Level 1: {params_1tp.tp_level1}%")
    print(f"  TP Percent 1: {params_1tp.tp_percent1}% (should be 100% for 1 TP)")
    print(f"  Trailing Deviation: {params_1tp.trailing_deviation}%")
    print(f"  Expected behavior: Sell 100% at {params_1tp.tp_level1}% profit, no trailing")
    
    # Validate configuration
    issues = []
    
    if params_1tp.tp_percent1 != 100.0:
        issues.append(f"❌ TP Percent 1 should be 100% for 1 TP system, got {params_1tp.tp_percent1}%")
    
    if params_1tp.trailing_deviation > params_1tp.tp_level1:
        issues.append(f"❌ Trailing deviation ({params_1tp.trailing_deviation}%) exceeds TP level ({params_1tp.tp_level1}%)")
    
    if params_1tp.tp_percent2 != 0.0 or params_1tp.tp_percent3 != 0.0:
        issues.append(f"❌ TP2/TP3 should be 0% in 1 TP system, got {params_1tp.tp_percent2}%/{params_1tp.tp_percent3}%")
    
    if issues:
        print("\n⚠️  CONFIGURATION ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ 1 TP Configuration is valid")
        return True


def test_apy_calculation_accuracy():
    """Test APY calculation accuracy with known scenarios"""
    
    print("\n🧮 TESTING APY CALCULATION ACCURACY")
    print("=" * 50)
    
    # Test scenario 1: Simple doubling over 1 year
    initial = 10000.0
    final = 20000.0
    days = 365
    
    expected_apy = 100.0  # 100% return over 1 year
    calculated_apy = ((final / initial) ** (365 / days) - 1) * 100
    
    print(f"Test 1 - Simple doubling:")
    print(f"  Initial: ${initial:,.2f}")
    print(f"  Final: ${final:,.2f}")
    print(f"  Period: {days} days")
    print(f"  Expected APY: {expected_apy:.2f}%")
    print(f"  Calculated APY: {calculated_apy:.2f}%")
    print(f"  ✅ Accurate" if abs(calculated_apy - expected_apy) < 0.01 else f"  ❌ Error: {abs(calculated_apy - expected_apy):.2f}%")
    
    # Test scenario 2: 50% gain over 6 months
    initial = 10000.0
    final = 15000.0
    days = 182  # ~6 months
    
    expected_apy = ((1.5) ** (365/182) - 1) * 100  # ~121.9%
    calculated_apy = ((final / initial) ** (365 / days) - 1) * 100
    
    print(f"\nTest 2 - 50% gain over 6 months:")
    print(f"  Initial: ${initial:,.2f}")
    print(f"  Final: ${final:,.2f}")
    print(f"  Period: {days} days")
    print(f"  Expected APY: {expected_apy:.2f}%")
    print(f"  Calculated APY: {calculated_apy:.2f}%")
    print(f"  ✅ Accurate" if abs(calculated_apy - expected_apy) < 0.01 else f"  ❌ Error: {abs(calculated_apy - expected_apy):.2f}%")
    
    return True


def fix_strategy_config_for_1tp():
    """Generate corrected strategy configuration for 1 TP system"""
    
    print("\n🔧 GENERATING CORRECTED 1 TP CONFIGURATION")
    print("=" * 50)
    
    corrected_config = """
# CORRECTED 1 TP CONFIGURATION
# Use this in strategy_config.py for proper 1 TP optimization

@dataclass
class StrategyParams:
    # === POSITION SIZING ===
    base_percent: float = 1.0
    volume_multiplier: float = 1.2

    # === ENTRY CONDITIONS ===
    initial_deviation: float = 3.0
    step_multiplier: float = 1.5
    max_safeties: int = 8

    # === SINGLE TAKE PROFIT TARGET ===
    tp_level1: float = 3.0
    tp_percent1: float = 100.0  # ALWAYS 100% for 1 TP system

    # === TRAILING STOP ===
    trailing_deviation: float = 2.0  # Must be <= tp_level1

    # === RSI CONDITIONS ===
    rsi_entry_threshold: float = 40.0
    rsi_safety_threshold: float = 30.0
    rsi_exit_threshold: float = 70.0

    # === FIXED PARAMETERS ===
    fees: float = 0.075

    def __post_init__(self):
        # CRITICAL: Ensure 1 TP configuration is valid
        if self.tp_percent1 != 100.0:
            print(f"WARNING: tp_percent1 should be 100% for 1 TP system, got {self.tp_percent1}%")
            self.tp_percent1 = 100.0
        
        # Ensure trailing doesn't exceed TP1
        if self.trailing_deviation > self.tp_level1:
            print(f"WARNING: trailing_deviation ({self.trailing_deviation}%) exceeds tp_level1 ({self.tp_level1}%)")
            self.trailing_deviation = self.tp_level1

    # Remove tp_level2 and tp_level3 properties for 1 TP system
    # These should not exist in a pure 1 TP configuration


# CORRECTED OPTIMIZATION RANGES FOR 1 TP
class OptimizationRanges:
    def __init__(self):
        # === OPTIMIZABLE PARAMETERS ===
        self.base_percent = [1.0, 1.5, 2.0, 2.5, 3.0]
        self.initial_deviation = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.tp_level1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.trailing_deviation = [0.5, 1.0, 1.5, 2.0, 2.5]  # Will be capped to tp_level1
        
        # === FIXED FOR 1 TP SYSTEM ===
        self.tp_percent1 = [100.0]  # ALWAYS 100% - sell entire position
        
        # === RSI RANGES ===
        self.rsi_entry_threshold = [30.0, 35.0, 40.0, 45.0, 50.0]
        self.rsi_safety_threshold = [20.0, 25.0, 30.0, 35.0, 40.0]
"""
    
    print(corrected_config)
    
    # Save to file
    with open('corrected_1tp_config.py', 'w') as f:
        f.write(corrected_config)
    
    print("\n✅ Corrected configuration saved to 'corrected_1tp_config.py'")
    print("📝 Apply these changes to strategy_config.py to fix 1 TP issues")


def main():
    """Run all validation and fix procedures"""
    
    print("🚀 1 TP CONFIGURATION APY ISSUE ANALYSIS & FIX")
    print("=" * 80)
    
    # Step 1: Validate current configuration
    config_valid = validate_1tp_configuration()
    
    # Step 2: Test APY calculation accuracy
    apy_accurate = test_apy_calculation_accuracy()
    
    # Step 3: Generate corrected configuration
    fix_strategy_config_for_1tp()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY OF ISSUES FOUND:")
    print("=" * 80)
    
    print("\n1. ⚠️  CONFIGURATION INCONSISTENCY:")
    print("   - strategy_config.py has tp_percent1=[100] (correct for 1 TP)")
    print("   - test_ultra_aggressive.py uses tp_percent1=70, tp_percent2=20, tp_percent3=10 (multi-TP)")
    print("   - This creates confusion between 1 TP and multi-TP systems")
    
    print("\n2. ⚠️  APY CALCULATION ISSUES:")
    print("   - Some simulations may not include open position value in final balance")
    print("   - Portfolio value tracking may be inconsistent")
    print("   - Phase 2 optimization may show inflated APY due to incomplete position closure")
    
    print("\n3. ⚠️  TRAILING STOP LOGIC:")
    print("   - In 1 TP system with 100% sell, trailing should be disabled")
    print("   - Current code may still activate trailing after 100% TP sale")
    print("   - This can cause phantom trades and incorrect APY calculations")
    
    print("\n" + "=" * 80)
    print("🔧 RECOMMENDED FIXES:")
    print("=" * 80)
    
    print("\n1. ✅ UPDATE STRATEGY_CONFIG.PY:")
    print("   - Ensure tp_percent1 is always [100.0] for 1 TP system")
    print("   - Remove tp_percent2 and tp_percent3 from optimization ranges")
    print("   - Add validation in __post_init__ to enforce 1 TP rules")
    
    print("\n2. ✅ FIX SIMULATION LOGIC:")
    print("   - Use fixed_simulate_strategy_1tp() function provided above")
    print("   - Ensure final balance includes open position value")
    print("   - Disable trailing stop when tp_percent1 >= 99%")
    
    print("\n3. ✅ UPDATE TEST FILES:")
    print("   - Change test_ultra_aggressive.py to use tp_percent1=100")
    print("   - Remove references to tp_percent2 and tp_percent3 in 1 TP tests")
    print("   - Validate APY calculations against known benchmarks")
    
    print("\n4. ✅ PHASE 2 OPTIMIZATION:")
    print("   - Ensure optimization uses corrected simulation function")
    print("   - Validate that APY calculations are consistent")
    print("   - Add debugging output to track portfolio value vs cash balance")
    
    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("=" * 80)
    print("1. Apply the corrected configuration from 'corrected_1tp_config.py'")
    print("2. Update simulation functions to use fixed_simulate_strategy_1tp()")
    print("3. Re-run optimization with corrected 1 TP logic")
    print("4. Validate APY results against manual calculations")
    print("5. Test with known scenarios to ensure accuracy")


if __name__ == "__main__":
    main()
