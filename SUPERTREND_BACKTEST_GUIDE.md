# 🚀 Supertrend DCA Backtesting Guide

This guide shows you how to backtest coins using the new **Supertrend-based DCA strategy** that opens both LONG and SHORT deals based on trend direction.

## 📋 Quick Start

### 1. **Simple Backtest (Default Parameters)**

```bash
python run_supertrend_backtest.py --coin BTCUSDT --data_path data/BTCUSDT_1m.csv
```

### 2. **Optimized Backtest (Recommended)**

```bash
python run_supertrend_backtest.py --coin BTCUSDT --data_path data/BTCUSDT_1m.csv --optimize --trials 200
```

### 3. **Custom Balance**

```bash
python run_supertrend_backtest.py --coin ETHUSDT --data_path data/ETHUSDT_1m.csv --optimize --initial_balance 50000
```

## 🎯 How the Supertrend Strategy Works

### **Entry Logic:**

- **LONG deals**: Opens when Supertrend is bullish (green)
- **SHORT deals**: Opens when Supertrend is bearish (red)
- **Consecutive deals**: Opens new deals continuously while trend direction is maintained
- **No other filters**: Only Supertrend direction matters for entries

### **Exit Logic:**

- **Immediate exit**: When Supertrend flips direction
- **LONG → SHORT flip**: Exits all long positions immediately
- **SHORT → LONG flip**: Exits all short positions immediately
- **Overrides TP/Trailing**: Supertrend exit takes priority

### **Safety Orders:**

- **LONG deals**: Buy more when price drops (traditional DCA)
- **SHORT deals**: Sell more when price rises (inverse DCA)
- **Same logic**: Uses existing deviation percentages

## 📊 Data Requirements

Your CSV file should have these columns:

```
ts,open,high,low,close,vol
2024-01-01 00:00:00,43000.0,43100.0,42900.0,43050.0,1234.56
2024-01-01 00:01:00,43050.0,43150.0,42950.0,43100.0,2345.67
...
```

## 🔧 Advanced Usage

### **Using the Original Fast Backtester**

```bash
python fast_dca_backtest.py --data_path data/BTCUSDT_1m.csv --coin BTCUSDT --optimize --trials 500
```

### **Optimization with Drawdown Elimination**

```bash
python fast_dca_backtest.py --data_path data/BTCUSDT_1m.csv --coin BTCUSDT --drawdown_elimination --trend_trials 100 --dca_trials 100
```

### **Market-Specific Optimization**

```bash
# Bull market optimization
python fast_dca_backtest.py --data_path data/BTCUSDT_1m.csv --coin BTCUSDT --optimize --market_type bull

# Bear market optimization
python fast_dca_backtest.py --data_path data/BTCUSDT_1m.csv --coin BTCUSDT --optimize --market_type bear
```

## ⚙️ Key Parameters Optimized

### **Supertrend Parameters:**

- **Timeframe**: 15m, 30m, 1h, 4h (as requested)
- **Period**: 7-21 (ATR calculation period)
- **Multiplier**: 1.5-5.0 (sensitivity)

### **Take Profit:**

- **Range**: 0.3% to 20% (as requested)
- **Trailing**: Up to TP-0.2% to prevent losses

### **Position Sizing (Never Changed):**

- **Base order**: Kept exactly as configured
- **Safety orders**: All multipliers preserved
- **Max safeties**: Unchanged

## 📈 Expected Results

### **What to Expect:**

- **Higher trade frequency**: More deals due to trend following
- **Both directions**: Profits from both bull and bear trends
- **Immediate exits**: Reduced drawdown duration
- **Optimized TP levels**: Better profit capture

### **Performance Metrics:**

- **APY**: Annual percentage yield
- **Max Drawdown**: Worst portfolio decline
- **Total Trades**: Number of completed deals
- **Avg Drawdown Duration**: Time spent in drawdown

## 📁 Output Files

After running, you'll get:

```
results/
├── BTCUSDT_supertrend_chart.png      # Portfolio chart with trades
├── BTCUSDT_supertrend_trades.csv     # Detailed trade log
└── BTCUSDT_fast_YYYYMMDD_HHMMSS_results.json  # Full results
```

## 🚨 Important Notes

### **Data Quality:**

- Use 1-minute OHLCV data for best results
- Ensure no gaps in data
- Minimum 30 days of data recommended

### **Optimization:**

- **Always use --optimize** for best results
- More trials = better optimization (but slower)
- 100-500 trials recommended

### **Position Sizing:**

- Base order sizes are **never changed** (as requested)
- Safety order multipliers are **preserved**
- Only entry/exit logic is modified

## 🔄 Migration from Old Strategy

If you were using the old RSI/SMA strategy:

### **Old Command:**

```bash
python dca_backtest.py --coin BTCUSDT --data_path data/BTCUSDT_1m.csv
```

### **New Command:**

```bash
python run_supertrend_backtest.py --coin BTCUSDT --data_path data/BTCUSDT_1m.csv --optimize
```

### **Key Differences:**

- ✅ **Dual direction**: Now trades both LONG and SHORT
- ✅ **Trend following**: Uses Supertrend instead of RSI/SMA
- ✅ **Immediate exits**: Exits on trend flip
- ✅ **Better optimization**: Includes Supertrend timeframes

## 🎯 Example Workflow

1. **Prepare your data** (1-minute OHLCV CSV)
2. **Run optimization** to find best parameters
3. **Review results** in the generated chart and logs
4. **Use optimized parameters** for live trading

```bash
# Step 1: Optimize
python run_supertrend_backtest.py --coin BTCUSDT --data_path data/BTCUSDT_1m.csv --optimize --trials 200

# Step 2: Review results in results/ folder

# Step 3: Use the optimized parameters shown in output
```

## 🆘 Troubleshooting

### **Common Issues:**

**"No module named 'fast_dca_backtest'"**

- Make sure you're in the correct directory
- Check that all files are present

**"Error loading data"**

- Verify CSV format matches requirements
- Check file path is correct
- Ensure data has no gaps

**"All trials failed"**

- Data might be too short (need minimum 30 days)
- Check data quality
- Try reducing number of trials

### **Getting Help:**

- Check the console output for detailed error messages
- Verify your data format matches the requirements
- Start with a smaller dataset for testing

---

**Ready to backtest? Start with the optimized command:**

```bash
python run_supertrend_backtest.py --coin YOUR_COIN --data_path path/to/your/data.csv --optimize
```
