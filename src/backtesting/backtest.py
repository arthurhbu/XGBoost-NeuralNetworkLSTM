import yaml
from pathlib import Path
import os
import pandas as pd
import xgboost as xgb
import json
from datetime import datetime

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

        current_portfolio_value = cash + stocks_held * test_df['Open'].iloc[i + 1]
        portfolio_history.append(current_portfolio_value)

    return pd.Series(portfolio_history, index=test_df.index[:-1])

def run_buy_and_hold(test_df, initial_capital):

    first_day_price = test_df['Close'].iloc[0]
    stocks_bought = initial_capital / first_day_price

    portfolio_history = stocks_bought * test_df['Close']

    return portfolio_history

def calculate_metrics(portfolio_history, label, ticker):

    result = {
        'ticker': ticker,
        'label': label,
        'capital_inicial': portfolio_history.iloc[0],
        'capital_final': portfolio_history.iloc[-1],
        'retorno_total': (portfolio_history.iloc[-1] / portfolio_history.iloc[0] - 1) * 100
    }

    return result

def write_txt_report(results_data, backtest_config, output_path):
    """Escreve um relatório em formato TXT legível"""
    
    txt_path = output_path.replace('.csv', '.txt')
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DE BACKTESTING - MODELO XGBOOST vs BUY AND HOLD\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data da simulação: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Capital inicial: R$ {backtest_config['initial_capital']:,.2f}\n")
        f.write(f"Custo de transação: {backtest_config['transaction_cost_pct']*100:.3f}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RESULTADOS POR BOLSA\n")
        f.write("-" * 80 + "\n\n")
        
        # Agrupar resultados por ticker
        ticker_results = {}
        for result in results_data:
            ticker = result['ticker']
            if ticker not in ticker_results:
                ticker_results[ticker] = []
            ticker_results[ticker].append(result)
        
        for ticker in sorted(ticker_results.keys()):
            f.write(f"BOLSA: {ticker}\n")
            f.write("-" * 40 + "\n")
            
            for result in ticker_results[ticker]:
                f.write(f"Estratégia: {result['label']}\n")
                f.write(f"  Capital Inicial: R$ {result['capital_inicial']:,.2f}\n")
                f.write(f"  Capital Final: R$ {result['capital_final']:,.2f}\n")
                f.write(f"  Retorno Total: {result['retorno_total']:+.2f}%\n")
                f.write("\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESUMO GERAL\n")
        f.write("=" * 80 + "\n\n")
        
        # Calcular estatísticas gerais
        model_results = [r for r in results_data if r['label'] == 'Modelo de Predição']
        buyhold_results = [r for r in results_data if r['label'] == 'Buy and Hold']
        
        if model_results:
            avg_model_return = sum(r['retorno_total'] for r in model_results) / len(model_results)
            f.write(f"Retorno médio do modelo XGBoost: {avg_model_return:+.2f}%\n")
        
        if buyhold_results:
            avg_buyhold_return = sum(r['retorno_total'] for r in buyhold_results) / len(buyhold_results)
            f.write(f"Retorno médio Buy and Hold: {avg_buyhold_return:+.2f}%\n")
        
        f.write(f"Total de bolsas analisadas: {len(set(r['ticker'] for r in results_data))}\n")
        f.write(f"Total de simulações realizadas: {len(results_data)}\n")

def write_json_report(results_data, backtest_config, output_path):
    """Escreve um relatório em formato JSON estruturado"""
    
    json_path = output_path.replace('.csv', '.json')
    
    report_data = {
        'metadata': {
            'simulation_date': datetime.now().isoformat(),
            'initial_capital': backtest_config['initial_capital'],
            'transaction_cost_pct': backtest_config['transaction_cost_pct'],
            'total_tickers': len(set(r['ticker'] for r in results_data)),
            'total_simulations': len(results_data)
        },
        'results_by_ticker': {},
        'summary': {
            'model_performance': {},
            'buyhold_performance': {}
        }
    }
    
    # Organizar resultados por ticker
    ticker_results = {}
    for result in results_data:
        ticker = result['ticker']
        if ticker not in ticker_results:
            ticker_results[ticker] = []
        ticker_results[ticker].append(result)
    
    # Adicionar resultados organizados por ticker
    for ticker, results in ticker_results.items():
        report_data['results_by_ticker'][ticker] = {
            'model_prediction': next((r for r in results if r['label'] == 'Modelo de Predição'), None),
            'buy_and_hold': next((r for r in results if r['label'] == 'Buy and Hold'), None)
        }
    
    # Calcular estatísticas do modelo
    model_results = [r for r in results_data if r['label'] == 'Modelo de Predição']
    if model_results:
        report_data['summary']['model_performance'] = {
            'average_return': sum(r['retorno_total'] for r in model_results) / len(model_results),
            'best_performer': max(model_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(model_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in model_results)
        }
    
    # Calcular estatísticas do buy and hold
    buyhold_results = [r for r in results_data if r['label'] == 'Buy and Hold']
    if buyhold_results:
        report_data['summary']['buyhold_performance'] = {
            'average_return': sum(r['retorno_total'] for r in buyhold_results) / len(buyhold_results),
            'best_performer': max(buyhold_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(buyhold_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in buyhold_results)
        }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

def main():
    
    results_simulated = []

    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    features_path = Path(__file__).resolve().parents[2] / config['data']['features_data_path']
    model_path = Path(__file__).resolve().parents[2] / config['model_training']['model_output_path']
    
    backtest_config = config["backtesting"]

    for feature_data_file in os.listdir(features_path):

        model_name = str(feature_data_file.replace(".csv", ".json"))
        ticker = feature_data_file.replace(".csv", "")

        print(f"Processando arquivo: {feature_data_file}")
        df = pd.read_csv(features_path / feature_data_file, index_col='Date', parse_dates=True)

        model = xgb.XGBClassifier()
        model.load_model(model_path / model_name)

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

        model_metrics = calculate_metrics(model_portfolio, "Modelo de Predição", ticker)
        buy_and_hold_metrics = calculate_metrics(buy_and_hold_portfolio, "Buy and Hold", ticker)
        results_simulated.append(model_metrics)
        results_simulated.append(buy_and_hold_metrics)

        results_df = pd.DataFrame({
            'Model_strategy': model_portfolio,
            'Buy_and_Hold': buy_and_hold_portfolio
        })

        result_file_name = str(feature_data_file.replace(".csv", "_results.csv"))
        results_path = f"{backtest_config['results_path']}/{result_file_name}"
        os.makedirs(backtest_config['results_path'], exist_ok=True)
        results_df.to_csv(results_path)

    # Criar DataFrame com todos os resultados
    results_df = pd.DataFrame(results_simulated)
    final_results_path = f"{backtest_config['results_path']}/results_simulated.csv"
    results_df.to_csv(final_results_path)
    
    # Criar relatórios em TXT e JSON
    print("\nCriando relatórios em TXT e JSON...")
    write_txt_report(results_simulated, backtest_config, final_results_path)
    write_json_report(results_simulated, backtest_config, final_results_path)
    
    print(f"Relatórios criados com sucesso em: {backtest_config['results_path']}")
    print("- results_simulated.csv (formato CSV)")
    print("- results_simulated.txt (formato TXT legível)")
    print("- results_simulated.json (formato JSON estruturado)")


if __name__ == "__main__":
    main()