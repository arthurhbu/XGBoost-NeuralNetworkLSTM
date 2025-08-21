# __main__.py
# Arquivo principal para executar o projeto
from src.backtesting.backtest import main
from src.reports.generate_report import main as generate_report
    
if __name__ == "__main__":
    # main()
    generate_report()
