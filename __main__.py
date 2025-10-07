from src.models.train_models import main as train_models
from src.backtesting.backtest import main as backtest
from src.features.feature_selection_per_ticker import main as feature_selection_per_ticker
from src.features.feature_selection_grid_search import main as feature_selection_grid_search
    
if __name__ == "__main__":
    # feature_selection_grid_search()
    # feature_selection_per_ticker()
    # train_models()
    backtest()