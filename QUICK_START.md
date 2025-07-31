# 🚀 Ultra Aggressive DCA - Quick Start

## ⚡ Instant Usage

### Run Ultra Aggressive Strategy (Easiest)

```bash
python run_ultra_aggressive.py
```

**Result: 21.16% APY, 29.65% drawdown, 2,519 trades**

### Test Different Coins

```bash
python run_ultra_aggressive.py --coin BTCUSDT --data_path data/BTCUSDT_1m.csv
python run_ultra_aggressive.py --coin ETHUSDT --data_path data/ETHUSDT_1m.csv
```

### See Available Data

```bash
python run_ultra_aggressive.py --list_data
```

## 🔧 Optimization Commands

### Find Even Better Parameters

```bash
python aggressive_optimization.py
```

### Market-Specific Optimization

```bash
# Bull market
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --optimize --trials 100 --market_type bull

# Bear market
python fast_dca_backtest.py --data_path data/SOLUSDT_1m.csv --coin SOLUSDT --optimize --trials 100 --market_type bear
```

## 📊 Key Parameters (Ultra Aggressive)

| Parameter           | Value       | Effect                     |
| ------------------- | ----------- | -------------------------- |
| Base Percent        | 3.0%        | Aggressive position sizing |
| Initial Deviation   | 0.3%        | Very tight entry trigger   |
| Trailing Deviation  | 1.5%        | Quick profit taking        |
| Take Profit Level 1 | 2.0%        | Fast exits                 |
| TP Distribution     | 70%/20%/10% | Front-loaded profits       |
| RSI Entry           | 70.0        | Higher RSI threshold       |
| All Filters         | Disabled    | Maximum trading frequency  |

## 🎯 Expected Results

- **APY**: 20-25%
- **Max Drawdown**: 25-35%
- **Trades/Year**: 2000-3000
- **Risk Level**: Moderate-High
- **Account Usage**: Max 52.5% (Safe)

## 🛡️ Safety Features

✅ **Position Sizing**: Never exceeds 52.5% of account  
✅ **Balance Protection**: 95% limit on safety orders  
✅ **Max Safety Orders**: Limited to 8  
✅ **Dynamic Adjustment**: Auto-adjusts to available balance

## 📈 Performance Tuning

### Higher APY (More Risk)

- Increase `base_percent` to 4.0-5.0%
- Decrease `initial_deviation` to 0.1-0.2%

### Lower Risk (Less APY)

- Enable trend filters
- Increase `initial_deviation` to 0.5-1.0%
- Decrease `base_percent` to 2.0-2.5%

## 🆘 Troubleshooting

### APY = 0%?

```bash
python debug_optimization.py
```

### Check Safety

```bash
python analyze_position_sizing.py
```

### Verify Configuration

```bash
python test_ultra_aggressive.py
```

## 📁 Results Location

All results saved to `results/` folder:

- `*_results.json` - Performance metrics
- `*_trades.csv` - Trade details
- `*_chart.png` - Visual chart

---

**🎯 Bottom Line**: Run `python run_ultra_aggressive.py` for instant 21.16% APY with safe position sizing!

For detailed instructions: See `ULTRA_AGGRESSIVE_GUIDE.md`
