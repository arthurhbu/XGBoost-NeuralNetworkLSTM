import yaml
from pathlib import Path
import os
import pandas as pd
import xgboost as xgb

from ..models.train_models import split_data, create_target_variable

def run_backtest(test_df, predictions, initial_capital, transaction_cost_pct):

    print("\n Começando simulação de investimento")

    print(f"Capital inicial: {initial_capital}")
    print(f"Custo de transação: {transaction_cost_pct}")

    cash = initial_capital
    stocks_held = 0
    portfolio_history = []

    for i in range(len(test_df)-1):

        current_date = test_df.index[i]
        prediction_for_next_day = predictions[i]

        execution_price = test_df['Open'].iloc[i + 1]

        if prediction_for_next_day == 1 and cash > 0:
            stocks_to_buy = cash / execution_price
            cost = stocks_to_buy * execution_price * transaction_cost_pct
            stocks_held += stocks_to_buy
            cash -= (stocks_to_buy * execution_price) + cost

        elif prediction_for_next_day == 0 and stocks_held > 0:
            sale_value = stocks_held * execution_price
            cost = sale_value * transaction_cost_pct
            cash += sale_value - cost
            stocks_held = 0

        current_portfolio_value = cash + stocks_held * test_df['Close'].iloc[i]
        portfolio_history.append({
            'Date': current_date,
            'Cash': cash,
            'Stocks': stocks_held,
            'Portfolio Value': current_portfolio_value
        })

    return pd.Series(portfolio_history, index=test_df.index[:-1])

def run_buy_and_hold(test_df, initial_capital):

    first_day_price = test_df['Open'].iloc[0]
    stocks_bought = initial_capital / first_day_price

    portfolio_history = stocks_bought * test_df['Close']

    return portfolio_history

def calculate_metrics(portfolio_history, label):

    total_return = (portfolio_history.iloc[-1] / portfolio_history.iloc[0] - 1) * 100

    print(f"\n--- Métricas para: {label} ---")
    print(f"Capital Inicial: R$ {portfolio_history.iloc[0]:,.2f}")
    print(f"Capital Final: R$ {portfolio_history.iloc[-1]:,.2f}")
    print(f"Retorno Total: {total_return:.2f}%")

def main():
    
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path = Path(__file__).resolve().parents[2] / config['data']['features_data_path']
    model_path = Path(__file__).resolve().parents[2] / config['model_training']['model_output_path']
    
    backtest_config = config["backtesting"]

    for feature_data_file in os.listdir(features_path):
        df = pd.read_csv(features_path / feature_data_file, index_col='Date', parse_dates=True)

        model = xgb.XGBClassifier()
        model.load_model(model_path)

        df = create_target_variable(df, config['model_training' ]['target_column'])
        _, _, _, _, x_test, y_test = split_data(
            df,
            config['model_training']['validation_start_date'],
            config['model_training']['test_start_date'],
            config['model_training']['target_column']
        )

        test_df = df[df.index >= config['model_training']['test_start_date']]

        predictions = model.predict(x_test)
        
        model_portfolio = run_backtest(
            test_df,
            predictions,
            backtest_config['initial_capital'],
            backtest_config['transaction_cost_pct']
        )

        buy_and_hold_portfolio = run_buy_and_hold(test_df, backtest_config['initial_capital'])

        calculate_metrics(model_portfolio, "Modelo de Predição")
        calculate_metrics(buy_and_hold_portfolio, "Buy and Hold")

        results_df = pd.DataFrame({
            'Model_strategy': model_portfolio,
            'Buy_and_Hold': buy_and_hold_portfolio
        })

        results_path = backtest_config['results_path'] / feature_data_file
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path)


if __name__ == "__main__":
    main()