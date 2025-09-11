import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import xgboost as xgb
import json
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from ..models.train_models import create_target_variable, split_data
from datetime import datetime, timezone

def calculate_ml_metrics(y_true, y_pred):
    """Calcula e formata um dicionário com as métricas de ML."""
    
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=['Baixa/Estável', 'Alta'], 
        output_dict=True,
        zero_division=0
    )
    
    if 'Alta' in report_dict:
        alta_metrics = report_dict['Alta']
    else:
        # Se não há predições de alta, usar valores padrão
        alta_metrics = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}
    
    metrics = {
        "Acurácia": f"{report_dict['accuracy']:.2%}",
        "Precisão (Classe Alta)": f"{alta_metrics['precision']:.2%}",
        "Recall (Classe Alta)": f"{alta_metrics['recall']:.2%}",
        "F1-Score (Classe Alta)": f"{alta_metrics['f1-score']:.2f}"
    }
    
    cm = confusion_matrix(y_true, y_pred)
    metrics["Matriz de Confusão"] = cm.tolist()
    
    pred_counts = pd.Series(y_pred).value_counts()
    metrics["Distribuição Predições"] = f"Alta: {pred_counts.get(1, 0)}, Baixa: {pred_counts.get(0, 0)}"
    
    return metrics

def calculate_financial_metrics(portfolio_series, risk_free_rate=0):
    """Calcula um conjunto de métricas de performance financeira."""
    
    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
    
    days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
    annualized_return = (1 + total_return) ** (365.25 / days) - 1
    
    daily_returns = portfolio_series.pct_change().dropna()
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
    
    running_max = portfolio_series.cummax()
    drawdown = (portfolio_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    metrics = {
        "Retorno Total": f"{total_return:.2%}",
        "Retorno Anualizado": f"{annualized_return:.2%}",
        "Volatilidade Anualizada": f"{annualized_volatility:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Drawdown Máximo": f"{max_drawdown:.2%}"
    }
    
    return metrics

def process_ticker_model(ticker, config, features_path, model_path):
    """Processa um ticker específico e retorna suas métricas."""
    
    ticker_name = ticker.replace('.SA', '')
    
    # Carrega o modelo específico do ticker
    model_file = model_path / f"{ticker}.json"
    if not model_file.exists():
        print(f"Modelo para {ticker} não encontrado: {model_file}")
        return None
    
    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    
    # Carrega os dados de features do ticker
    features_file = features_path / f"{ticker}.csv"
    if not features_file.exists():
        print(f"Features para {ticker} não encontradas: {features_file}")
        return None
    
    df_features = pd.read_csv(features_file, index_col='Date', parse_dates=True)
    
    # Cria variável alvo e divide os dados
    df_target = create_target_variable(df_features, config['model_training']['target_column'])
    _, _, _, _, X_test, y_test = split_data(
        df_target,
        config['model_training']['train_final_date'],
        config['model_training']['validation_start_date'],
        config['model_training']['validation_end_date'],
        config['model_training']['test_start_date'],
        config['model_training']['test_end_date'],
        config['model_training']['target_column']
    )
    
    # Obter probabilidades
    probabilities = model.predict_proba(X_test)[:, 1]
    
    optimal_threshold = 0.5  # Pode ser ajustado conforme necessário
    predictions = (probabilities >= optimal_threshold).astype(int)
    
    # Calcula métricas
    ml_metrics = calculate_ml_metrics(y_test, predictions)
    
    return {
        'ticker': ticker,
        'ml_metrics': ml_metrics,
        'test_size': len(y_test),
        'predictions': predictions.tolist(),
        'y_test': y_test.tolist()
    }

def generate_consolidated_report(results_df, ticker_results, config):
    """Gera relatório consolidado com todos os tickers."""
    
    print("\n" + "="*60)
    print(" RELATÓRIO CONSOLIDADO DE PERFORMANCE - TODOS OS TICKERS")
    print("="*60)
    
    # Métricas consolidadas de ML
    print("\n--- 1. MÉTRICAS DE MACHINE LEARNING CONSOLIDADAS ---")
    for ticker_result in ticker_results:
        if ticker_result:
            ticker = ticker_result['ticker']
            ml_metrics = ticker_result['ml_metrics']
            print(f"\n{ticker}:")
            for metric, value in ml_metrics.items():
                if metric != "Matriz de Confusão":
                    print(f"  - {metric}: {value}")
    
    # Métricas financeiras consolidadas
    print("\n--- 2. MÉTRICAS FINANCEIRAS CONSOLIDADAS ---")
    
    # Agrupa por estratégia
    model_results = results_df[results_df['label'] == 'Modelo de Predição']
    bnh_results = results_df[results_df['label'] == 'Buy and Hold']
    
    print(f"\nEstratégia do Modelo (Consolidada):")
    total_model_return = (model_results['capital_final'].sum() / model_results['capital_inicial'].sum()) - 1
    print(f"  - Retorno Total Consolidado: {total_model_return:.2%}")
    print(f"  - Capital Final Total: R$ {model_results['capital_final'].sum():,.2f}")
    
    print(f"\nEstratégia Buy and Hold (Consolidada):")
    total_bnh_return = (bnh_results['capital_final'].sum() / bnh_results['capital_inicial'].sum()) - 1
    print(f"  - Retorno Total Consolidado: {total_bnh_return:.2%}")
    print(f"  - Capital Final Total: R$ {bnh_results['capital_final'].sum():,.2f}")
    
    return ticker_results, model_results, bnh_results

def save_report_files(ticker_results, model_results, bnh_results, results_df, config):
    """Salva relatórios em múltiplos formatos."""
    
    reports_dir = Path(__file__).resolve().parents[2] / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Timestamps mais claros e com fuso horário
    now_local = datetime.now().astimezone()
    now_utc = now_local.astimezone(timezone.utc)
    # Seguro para nome de arquivo (sem dois-pontos), inclui offset do fuso (ex.: -0300)
    timestamp = now_local.strftime("%Y-%m-%d_%H-%M-%S_%z")
    # Legível para humanos (ex.: 2025-09-01 23:46:47 BRT-0300)
    human_timestamp = now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    # ISO UTC para interoperabilidade
    iso_utc = now_utc.isoformat()
    
    # Prepara dados para o relatório
    report_data = {
        "timestamp": timestamp,
        "generated_at_local": human_timestamp,
        "generated_at_utc": iso_utc,
        "ticker_results": ticker_results,
        "financial_summary": {
            "total_tickers": len(results_df['ticker'].unique()),
            "model_total_return": (model_results['capital_final'].sum() / model_results['capital_inicial'].sum()) - 1,
            "bnh_total_return": (bnh_results['capital_final'].sum() / bnh_results['capital_inicial'].sum()) - 1,
            "model_final_capital": model_results['capital_final'].sum(),
            "bnh_final_capital": bnh_results['capital_final'].sum(),
            "initial_capital_per_ticker": config['backtesting']['initial_capital']
        },
        "individual_results": results_df.to_dict('records')
    }
    
    txt_path = reports_dir / f"comprehensive_report_{timestamp}.txt"
    json_path = reports_dir / f"comprehensive_report_{timestamp}.json"
    csv_path = reports_dir / f"comprehensive_report_{timestamp}.csv"
    
    # Gera relatório TXT
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO COMPREENSIVO DE PERFORMANCE - TODOS OS TICKERS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Gerado em: {human_timestamp} (UTC: {iso_utc})\n\n")
        
        f.write("1. MÉTRICAS DE MACHINE LEARNING POR TICKER\n")
        f.write("-" * 50 + "\n")
        for ticker_result in ticker_results:
            if ticker_result:
                ticker = ticker_result['ticker']
                ml_metrics = ticker_result['ml_metrics']
                f.write(f"\n{ticker}:\n")
                for metric, value in ml_metrics.items():
                    if metric != "Matriz de Confusão":
                        f.write(f"  {metric}: {value}\n")
                f.write(f"  Tamanho do teste: {ticker_result['test_size']}\n")
        
        f.write("\n2. RESUMO FINANCEIRO CONSOLIDADO\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de tickers: {report_data['financial_summary']['total_tickers']}\n")
        f.write(f"Retorno total (Modelo): {report_data['financial_summary']['model_total_return']:.2%}\n")
        f.write(f"Retorno total (Buy&Hold): {report_data['financial_summary']['bnh_total_return']:.2%}\n")
        f.write(f"Capital final (Modelo): R$ {report_data['financial_summary']['model_final_capital']:,.2f}\n")
        f.write(f"Capital final (Buy&Hold): R$ {report_data['financial_summary']['bnh_final_capital']:,.2f}\n")
        
        f.write("\n3. RESULTADOS INDIVIDUAIS POR TICKER\n")
        f.write("-" * 50 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['ticker']} - {row['label']}: R$ {row['capital_inicial']:,.2f} → R$ {row['capital_final']:,.2f} ({row['retorno_total']:.2%})\n")
    
    # Gera relatório JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # Gera relatório CSV consolidado
    summary_rows = []
    for ticker_result in ticker_results:
        if ticker_result:
            ticker = ticker_result['ticker']
            ml_metrics = ticker_result['ml_metrics']
            
            # Encontra resultados financeiros do ticker
            ticker_financial = results_df[results_df['ticker'] == ticker]
            
            for _, row in ticker_financial.iterrows():
                summary_rows.append({
                    'Ticker': ticker,
                    'Estratégia': row['label'],
                    'Capital_Inicial': row['capital_inicial'],
                    'Capital_Final': row['capital_final'],
                    'Retorno_Total': row['retorno_total'],
                    'Acurácia_ML': ml_metrics['Acurácia'],
                    'Precisão_ML': ml_metrics['Precisão (Classe Alta)'],
                    'Recall_ML': ml_metrics['Recall (Classe Alta)'],
                    'F1_Score_ML': ml_metrics['F1-Score (Classe Alta)']
                })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"\nRelatórios salvos em:")
    print(f"  - TXT: {txt_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")

def main():
    """Função principal para executar o script de geração de relatório."""
    config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    results_path = Path(__file__).resolve().parents[2] / config['backtesting']['results_path']
    model_path = Path(__file__).resolve().parents[2] / config['model_training']['model_output_path']
    features_path = Path(__file__).resolve().parents[2] / config['data']['features_data_path']

    # Carrega resultados consolidados
    results_df = pd.read_csv(results_path / "results_simulated.csv")
    
    # Lista de tickers disponíveis
    tickers = config['data']['tickers']
    
    # Processa cada ticker individualmente
    ticker_results = []
    for ticker in tickers:
        print(f"\nProcessando {ticker}...")
        result = process_ticker_model(ticker, config, features_path, model_path)
        ticker_results.append(result)
    
    # Gera relatório consolidado
    ticker_results, model_results, bnh_results = generate_consolidated_report(
        results_df, ticker_results, config
    )
    
    # Salva relatórios em arquivos
    save_report_files(ticker_results, model_results, bnh_results, results_df, config)

if __name__ == '__main__':
    main()