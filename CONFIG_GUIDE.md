# DCA Strategy Configuration Guide

All customizable optimization variables are now centralized in `strategy_config.py`. This guide shows you how to use and customize them.

## 🎯 Quick Start

### Use Preset Strategies
```bash
# Conservative strategy (low risk)
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --preset conservative

# Bull market strategy (higher returns)
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --preset bull_market

# Aggressive strategy (high risk/reward)
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --preset aggressive
```

### Market-Specific Optimization
```bash
# Optimize for bull market conditions
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --optimize --market_type bull

# Optimize for bear market conditions  
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --optimize --market_type bear

# Optimize for sideways market
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --optimize --market_type sideways
```

## 📊 Available Presets

| Preset | Base % | TP1 % | TP1 Sell % | RSI Entry | Best For |
|--------|--------|-------|------------|-----------|----------|
| `conservative` | 1.0% | 3.0% | 50% | <35 | Risk management |
| `aggressive` | 5.0% | 5.0% | 25% | <55 | Bull markets |
| `bull_market` | 3.0% | 8.0% | 20% | <60 | Trending up |
| `bear_market` | 2.0% | 2.0% | 60% | <40 | Trending down |
| `scalping` | 1.5% | 1.0% | 70% | <50 | Quick profits |

## 🔧 Customizable Parameters

### Position Sizing
- **`base_percent`**: Base order size (% of balance)
  - Range: `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]`
  - Higher = More aggressive, more profit/loss per trade

- **`volume_multiplier`**: Safety order scaling
  - Range: `[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]`
  - Higher = Larger safety orders, more capital usage

### Entry Conditions
- **`initial_deviation`**: First safety order trigger (%)
  - Range: `[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`
  - Lower = More safety orders, better averaging

- **`step_multiplier`**: Safety order step scaling
  - Range: `[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]`
  - Higher = Wider spacing between safety orders

- **`max_safeties`**: Maximum safety orders
  - Range: `[4, 6, 8, 10, 12]`
  - More = Better averaging but more capital needed

### Take Profit Levels
- **`tp_level1`**: First take profit trigger (%)
  - Range: `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]`
  - Higher = Let profits run longer

- **`tp_percent1/2/3`**: % to sell at each TP level
  - TP1: `[15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0]`
  - TP2: `[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]`
  - TP3: `[10.0, 15.0, 20.0, 25.0, 30.0, 35.0]`
  - Lower = Keep more position running

### RSI Conditions
- **`rsi_entry_threshold`**: RSI level for new entries
  - Range: `[30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0]`
  - Higher = More entries (more aggressive)

- **`rsi_safety_threshold`**: RSI for safety orders
  - Range: `[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]`
  - Higher = More safety orders

### Trailing Stop
- **`trailing_deviation`**: Trailing stop distance (%)
  - Range: `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`
  - Lower = Tighter stops, less drawdown

## 🏗️ Custom Configuration

### Edit Ranges in Code
```python
from strategy_config import OptimizationRanges, OptimizationConfig

# Create custom ranges
custom_ranges = OptimizationRanges(
    base_percent=[1.0, 2.0, 3.0, 5.0],     # Only these values
    tp_level1=[5.0, 8.0, 10.0, 15.0],      # Higher TPs for bull market
    rsi_entry_threshold=[50.0, 60.0, 70.0] # More aggressive entries
)

# Use in optimization
config = OptimizationConfig(custom_ranges)
optimizer = FastOptimizer(backtester, config)
```

### Create Custom Preset
```python
from strategy_config import StrategyParams

def my_custom_strategy() -> StrategyParams:
    return StrategyParams(
        base_percent=4.0,           # 4% base orders
        initial_deviation=1.5,      # Tight safety orders
        tp_level1=6.0,             # 6% take profit
        tp_percent1=30.0,          # Sell 30% at TP1
        rsi_entry_threshold=55.0,   # Buy when RSI < 55
        trailing_deviation=2.0      # 2% trailing stop
    )
```

## 🎨 Market-Specific Ranges

### Bull Market Ranges
- **Larger positions**: `[2.0, 3.0, 4.0, 5.0, 7.5, 10.0]%`
- **Higher TPs**: `[3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]%`
- **Keep more position**: Sell `[15-35]%` at TPs
- **More entries**: RSI `[45-65]`

### Bear Market Ranges  
- **Smaller positions**: `[0.5, 1.0, 1.5, 2.0, 2.5]%`
- **Lower TPs**: `[1.0, 1.5, 2.0, 2.5, 3.0]%`
- **Sell more**: `[40-70]%` at TPs
- **Conservative entries**: RSI `[25-45]`

## 📈 Performance Examples

| Configuration | APY | Max DD | Trades | Best For |
|---------------|-----|--------|--------|----------|
| Default | 2.3% | 2.1% | 320 | Stable/Safe |
| Bull Market Preset | 19.3% | 0.8% | 8 | Bull runs |
| Bull Optimized | 49.0% | 2.0% | - | Aggressive bull |
| Conservative | 3.5% | 1.8% | 109 | Risk averse |

## 🚀 Best Practices

1. **Start with presets** - Use `--preset bull_market` for trending markets
2. **Test different periods** - Use `--sample_days 30` for quick tests
3. **Optimize by market** - Use `--market_type bull/bear/sideways`
4. **Monitor drawdown** - Higher returns often mean higher risk
5. **Backtest thoroughly** - Test on multiple time periods

## 🔍 Finding Optimal Parameters

```bash
# Quick optimization for current market conditions
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT \
  --optimize --trials 100 --market_type bull

# Full optimization with more trials
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT \
  --optimize --trials 500 --market_type bull

# Test preset vs optimized
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --preset bull_market
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --optimize --market_type bull
```

---

**Key Insight**: The default 2% APY was correct for conservative parameters. By using bull market presets or optimization, we can achieve 19-49% APY with appropriate risk levels.