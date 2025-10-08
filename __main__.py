from src.models.train_models import main as train_models
from src.backtesting.backtest import main as backtest
from src.analysis.feature_importance_sweep import run_sweep as feature_importance_sweep  

if __name__ == "__main__":
    # train_models()
    # feature_importance_sweep()
    backtest()