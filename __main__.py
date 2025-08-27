# __main__.py
# Arquivo principal para executar o projeto
from src.models.train_models import main as train_models
from src.backtesting.backtest import main as backtest
from src.reports.generate_report import main as generate_report
    
if __name__ == "__main__":
    train_models()
    backtest()
    generate_report()
