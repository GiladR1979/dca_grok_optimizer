# 🚀 Multi-Coin DCA Optimization Guide

## 📋 Available Coins

You have **11 coins** available for optimization:

| Coin      | File Size | Notes                   |
| --------- | --------- | ----------------------- |
| 1INCHUSDT | 82.4 MB   | DeFi token              |
| BNBUSDT   | 148.1 MB  | Binance native token    |
| BTCUSDT   | 240.5 MB  | Bitcoin (largest file)  |
| EOSUSDT   | 76.8 MB   | EOS blockchain token    |
| ETHBTC    | 244.3 MB  | ETH/BTC pair            |
| ETHUSDT   | 234.3 MB  | Ethereum                |
| PEPEUSDT  | 75.1 MB   | Meme coin               |
| SOLUSDT   | 131.9 MB  | Solana (already tested) |
| SUIUSDT   | 55.7 MB   | Sui blockchain          |
| WIFUSDT   | 35.3 MB   | Meme coin (smallest)    |
| XLMUSDT   | 148.1 MB  | Stellar Lumens          |

## ⚡ Quick Usage

### Run All Coins (5000 trials each)

```bash
python optimize_all_coins.py --trials 5000
```

**Estimated time: ~5.5 hours (30 min timeout per coin)**

### Test Run (Fewer trials)

```bash
python optimize_all_coins.py --trials 100
```

**Estimated time: ~1-2 hours**

### Specific Coins Only

```bash
python optimize_all_coins.py --coins BTCUSDT ETHUSDT SOLUSDT --trials 5000
```

### Skip Large Files (Faster testing)

```bash
python optimize_all_coins.py --skip BTCUSDT ETHBTC ETHUSDT --trials 1000
```

### Small Coins Only (Quick test)

```bash
python optimize_all_coins.py --coins WIFUSDT SUIUSDT PEPEUSDT --trials 2000
```

## 🎯 Recommended Strategies

### 1. Quick Discovery (30 minutes)

```bash
# Test small/fast coins first
python optimize_all_coins.py --coins WIFUSDT SUIUSDT PEPEUSDT --trials 1000
```

### 2. Major Coins Focus (2-3 hours)

```bash
# Focus on major cryptocurrencies
python optimize_all_coins.py --coins BTCUSDT ETHUSDT SOLUSDT BNBUSDT --trials 3000
```

### 3. Full Portfolio Scan (5-6 hours)

```bash
# Optimize all coins with high trial count
python optimize_all_coins.py --trials 5000 --timeout 45
```

### 4. Meme Coin Strategy

```bash
# High volatility coins
python optimize_all_coins.py --coins PEPEUSDT WIFUSDT --trials 10000
```

## 📊 Expected Results

### Performance Tiers (Estimated)

- **Tier 1 (High APY)**: SOLUSDT, PEPEUSDT, WIFUSDT
- **Tier 2 (Stable)**: BTCUSDT, ETHUSDT, BNBUSDT
- **Tier 3 (Conservative)**: XLMUSDT, EOSUSDT, SUIUSDT

### Risk Levels

- **High Risk/High Reward**: Meme coins (PEPE, WIF)
- **Medium Risk**: Layer 1 tokens (SOL, SUI)
- **Lower Risk**: Established coins (BTC, ETH, BNB)

## 🔧 Advanced Options

### Custom Timeout (For large files)

```bash
python optimize_all_coins.py --timeout 60 --trials 5000
```

### Resume Failed Optimizations

```bash
# Skip already completed coins
python optimize_all_coins.py --skip SOLUSDT --trials 5000
```

### Parallel Processing (Future feature)

```bash
# Currently single-threaded, parallel support planned
python optimize_all_coins.py --parallel 2 --trials 5000
```

## 📈 Output Files

After completion, you'll get:

- **`all_coins_optimization_TIMESTAMP.json`** - Detailed results
- **`all_coins_summary_TIMESTAMP.csv`** - Summary table
- **Individual coin results** - Standard backtest outputs

## 🎯 Optimization Tips

### For Speed

1. Start with `--trials 100` to test the script
2. Use `--coins` to focus on specific coins
3. Skip large files initially: `--skip BTCUSDT ETHBTC ETHUSDT`

### For Accuracy

1. Use `--trials 5000` or higher for final runs
2. Increase `--timeout` for large files
3. Run multiple times and compare results

### For Discovery

1. Test meme coins first (high volatility = more opportunities)
2. Compare similar coins (e.g., PEPE vs WIF)
3. Look for coins with high trade frequency

## ⚠️ Important Notes

### Time Estimates

- **100 trials**: ~2-5 minutes per coin
- **1000 trials**: ~5-15 minutes per coin
- **5000 trials**: ~15-30 minutes per coin
- **Large files** (BTC, ETH): Add 50-100% more time

### Resource Usage

- **GPU**: Will be used automatically if available
- **Memory**: Large files may use 4-8GB RAM
- **Disk**: Results files will accumulate in `results/`

### Best Practices

1. **Start small**: Test with 2-3 coins first
2. **Monitor progress**: Check intermediate results
3. **Save results**: All outputs are automatically saved
4. **Compare fairly**: Use same trial count for comparison

## 🆘 Troubleshooting

### If a coin times out:

```bash
# Increase timeout for that specific coin
python optimize_all_coins.py --coins BTCUSDT --timeout 60 --trials 5000
```

### If optimization fails:

```bash
# Check individual coin manually
python fast_dca_backtest.py --data_path data/BTCUSDT_1m.csv --coin BTCUSDT --optimize --trials 100
```

### If results are poor:

1. Check if data quality is good
2. Try different trial counts
3. Compare with SOLUSDT baseline (21.16% APY)

## 🏆 Success Metrics

### Excellent Results

- **APY > 20%**: Outstanding performance
- **APY/Drawdown > 0.7**: Great risk-adjusted returns
- **Trades > 1000**: Good opportunity frequency

### Good Results

- **APY > 15%**: Solid performance
- **APY/Drawdown > 0.5**: Acceptable risk
- **Trades > 500**: Reasonable activity

### Investigate Further

- **APY < 10%**: May need parameter tuning
- **APY = 0%**: Check for configuration issues
- **Very high drawdown**: Consider risk management

---

**🎯 Bottom Line**: Start with `python optimize_all_coins.py --trials 1000` to get a feel for all coins, then focus on the best performers with higher trial counts!
