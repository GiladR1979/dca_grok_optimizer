
from strategy_config import StrategyParams

def get_ultra_aggressive_params():
    return StrategyParams(
        base_percent=3.0,
        initial_deviation=0.3,
        trailing_deviation=1.5,
        tp_level1=2.0,
        tp_percent1=100.0,
        rsi_entry_threshold=70.0,
        rsi_safety_threshold=75.0,
        fees=0.075
    )
