#!/usr/bin/env python3
"""
Check if default fast_dca_backtest parameters have valid position size
"""

from strategy_config import StrategyParams, StrategyPresets

def calculate_total_position_size(base_percent, volume_multiplier, max_safeties):
    """Calculate total position size including all safety orders"""
    total = base_percent
    for i in range(max_safeties):
        total += base_percent * (volume_multiplier ** i)
    return total

def check_all_defaults():
    """Check position sizes for all default configurations"""
    
    print("🔍 CHECKING DEFAULT POSITION SIZES")
    print("=" * 80)
    
    # Check default parameters
    default = StrategyParams()
    default_total = calculate_total_position_size(
        default.base_percent,
        default.volume_multiplier,
        default.max_safeties
    )
    
    print(f"Default StrategyParams:")
    print(f"  Base: {default.base_percent}%")
    print(f"  Volume Multiplier: {default.volume_multiplier}")
    print(f"  Max Safeties: {default.max_safeties}")
    print(f"  TOTAL POSITION: {default_total:.2f}%")
    print(f"  Status: {'✅ VALID' if default_total <= 100 else '❌ INVALID'}")
    
    print("\n" + "-" * 80)
    print("Checking all presets:")
    print("-" * 80)
    
    presets = {
        "Conservative": StrategyPresets.conservative(),
        "Aggressive": StrategyPresets.aggressive(),
        "Bull Market": StrategyPresets.bull_market(),
        "Bear Market": StrategyPresets.bear_market(),
        "Scalping": StrategyPresets.scalping()
    }
    
    for name, preset in presets.items():
        total = calculate_total_position_size(
            preset.base_percent,
            preset.volume_multiplier,
            preset.max_safeties
        )
        
        print(f"\n{name} Preset:")
        print(f"  Base: {preset.base_percent}%")
        print(f"  Volume Multiplier: {preset.volume_multiplier}")
        print(f"  Max Safeties: {preset.max_safeties}")
        print(f"  TOTAL POSITION: {total:.2f}%")
        print(f"  Status: {'✅ VALID' if total <= 100 else '❌ INVALID'}")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("\nIf you run fast_dca_backtest.py without any parameters:")
    print(f"- It will use the default StrategyParams")
    print(f"- Total position size will be: {default_total:.2f}%")
    print(f"- This is {'✅ VALID (within 100% limit)' if default_total <= 100 else '❌ INVALID (exceeds 100% limit)'}")
    
    if default_total <= 100:
        print("\n✅ YES, the order sizes will be OK if you run fast_dca_backtest with defaults!")
    else:
        print("\n❌ NO, the default order sizes exceed 100% of account!")
        print("You should use a preset like --preset conservative or --preset bear_market")

if __name__ == "__main__":
    check_all_defaults()
