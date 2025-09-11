import yaml
from pathlib import Path
import os
import pandas as pd
import xgboost as xgb
import json
from datetime import datetime
import numpy as np
import logging
from sklearn.metrics import precision_recall_curve, f1_score

from ..models.train_models import split_data, create_target_variable

def optimize_threshold(model, x_val, y_val):
    """
    Otimiza o threshold baseado no F1-Score no conjunto de validação
    
    Args:
        model: Modelo XGBoost treinado
        x_val: Features de validação
        y_val: Target de validação
    
    Returns:
        best_threshold: Threshold otimizado
        best_f1: Melhor F1-Score encontrado
    """
    # Obter probabilidades de predição
    y_proba = model.predict_proba(x_val)[:, 1]
    
    # Calcular precision, recall e thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Calcular F1-Score para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Encontrar o threshold com melhor F1-Score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1

def predict_with_threshold(model, x_data, threshold=0.5):
    """
    Faz predições usando um threshold customizado
    
    Args:
        model: Modelo XGBoost treinado
        x_data: Features para predição
        threshold: Threshold para classificação
    
    Returns:
        predictions: Predições binárias
        probabilities: Probabilidades de predição
    """
    # Obter probabilidades
    probabilities = model.predict_proba(x_data)[:, 1]
    
    # Aplicar threshold customizado
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

def generate_actions_from_probabilities(probabilities, buy_threshold=0.6, sell_threshold=0.4, hold_min_days=1):
    """
    Converte probabilidades em ações 0/1 com zona neutra e histerese.
    - Se p >= buy_threshold -> alvo 1 (comprado)
    - Se p <= sell_threshold -> alvo 0 (zerado)
    - Caso contrário -> mantém posição anterior
    - hold_min_days: mínimo de dias entre trocas de estado
    """
    actions = []
    position = 0
    days_since_change = hold_min_days
    for p in probabilities:
        desired = position
        if p >= buy_threshold:
            desired = 1
        elif p <= sell_threshold:
            desired = 0

        if desired != position and days_since_change >= hold_min_days:
            position = desired
            days_since_change = 0
        else:
            days_since_change += 1

        actions.append(position)
    return np.array(actions, dtype=int)

def simulate_portfolio_next_open(test_df, actions, initial_capital, transaction_cost_pct):
    """
    Simula portfólio com execução em Open(t+1) sem logging (uso interno para otimização).
    actions: array 0/1 representando posição alvo para cada dia (usar estado acumulado).
    """
    cash = initial_capital
    stocks_held = 0.0
    portfolio_history = []
    for i in range(len(test_df)-1):
        desired_pos = actions[i]
        execution_price = test_df['Open'].iloc[i+1]

        if desired_pos == 1 and stocks_held == 0 and cash > 0:
            qty = cash / (execution_price * (1 + transaction_cost_pct))
            total_cost = qty * execution_price * (1 + transaction_cost_pct)
            if total_cost <= cash:
                stocks_held += qty
                cash -= total_cost
        elif desired_pos == 0 and stocks_held > 0:
            sale_value = stocks_held * execution_price
            cash += sale_value * (1 - transaction_cost_pct)
            stocks_held = 0.0

        portfolio_history.append(cash + stocks_held * test_df['Close'].iloc[i])

    final_value = cash + stocks_held * test_df['Close'].iloc[-1]
    portfolio_history.append(final_value)
    return pd.Series(portfolio_history, index=test_df.index)

def calculate_sharpe(portfolio_series):
    returns = portfolio_series.pct_change().dropna()
    if returns.std() == 0 or len(returns) == 0:
        return -np.inf
    return (returns.mean() / returns.std()) * np.sqrt(252)

def optimize_thresholds_financial(model, x_val, val_df, initial_capital, transaction_cost_pct, buy_grid=None, sell_grid=None, hold_min_days=1):
    """
    Busca thresholds que maximizam Sharpe no conjunto de validação.
    Retorna (best_buy_th, best_sell_th, best_sharpe)
    """
    if buy_grid is None:
        buy_grid = np.arange(0.55, 0.86, 0.05)
    if sell_grid is None:
        sell_grid = np.arange(0.15, 0.46, 0.05)

    proba = model.predict_proba(x_val)[:, 1]
    best_sharpe = -np.inf
    best_pair = (0.5, 0.5)

    for buy_th in buy_grid:
        for sell_th in sell_grid:
            if sell_th >= buy_th:
                continue
            actions = generate_actions_from_probabilities(proba, buy_threshold=buy_th, sell_threshold=sell_th, hold_min_days=hold_min_days)
            # alinhar tamanhos se necessário
            size = min(len(val_df), len(actions))
            pf = simulate_portfolio_next_open(val_df.iloc[:size], actions[:size], initial_capital, transaction_cost_pct)
            sharpe = calculate_sharpe(pf)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_pair = (buy_th, sell_th)

    return best_pair[0], best_pair[1], best_sharpe

def setup_logging(ticker, backtest_config):
    """
    Configura sistema de logging estruturado para cada ticker
    """
    # Criar diretório de logs se não existir
    logs_dir = Path(backtest_config['results_path']) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{ticker}_{timestamp}.log"
    log_filepath = logs_dir / log_filename
    
    # Configurar logger
    logger = logging.getLogger(f"backtest_{ticker}")
    logger.setLevel(logging.INFO)
    
    # Evitar duplicação de handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Formato do log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger, log_filepath

def run_backtest(test_df, predictions, initial_capital, transaction_cost_pct, logger):
    """
    Backtesting corrigido com logging estruturado
    """
    logger.info("="*60)
    logger.info("INICIANDO BACKTESTING")
    logger.info("="*60)
    
    logger.info(f"Tamanho test_df: {len(test_df)}")
    logger.info(f"Tamanho predictions: {len(predictions)}")
    logger.info(f"Capital inicial: R$ {initial_capital:,.2f}")
    logger.info(f"Custo de transação: {transaction_cost_pct*100:.3f}%")
    
    # ✅ CORREÇÃO 1: Verificar e corrigir alinhamento entre dados e predições
    if len(test_df) != len(predictions):
        logger.warning(f"ALINHAMENTO: test_df ({len(test_df)}) != predictions ({len(predictions)})")
        min_size = min(len(test_df), len(predictions))
        test_df = test_df.iloc[:min_size]
        predictions = predictions[:min_size]
        logger.info(f"ALINHAMENTO: Ajustado para {min_size} registros")
    
    # Log da distribuição das predições
    logger.info("DISTRIBUIÇÃO DAS PREDIÇÕES:")
    logger.info(f"  - Total: {len(predictions)}")
    logger.info(f"  - Predições de alta (1): {np.sum(predictions == 1)}")
    logger.info(f"  - Predições de baixa (0): {np.sum(predictions == 0)}")
    logger.info(f"  - Percentual de alta: {np.sum(predictions == 1)/len(predictions)*100:.1f}%")
    
    cash = initial_capital
    stocks_held = 0.0
    portfolio_history = []
    trade_count = 0
    buy_count = 0
    sell_count = 0
    
    logger.info("EXECUÇÃO DOS TRADES:")
    
    for i in range(len(test_df)-1):
        current_date = test_df.index[i]
        prediction = predictions[i]
        
        # Log a cada 50 iterações para acompanhar progresso
        if i % 50 == 0:
            logger.info(f"PROGRESSO: Data {current_date.date()} | Pred: {prediction} | Cash: R$ {cash:,.2f} | Ações: {stocks_held:.0f}")
        
        if i >= len(predictions):
            break
        
        # Execução realista: decisão em t executa em Open(t+1)
        next_open = test_df['Open'].iloc[i+1]
        estimated_execution_price = next_open
        
        # Lógica de compra/venda com execução no próximo pregão
        # Position sizing (se habilitado)
        sizing_cfg = backtest_config.get('position_sizing', {"enabled": False}) if 'backtest_config' in locals() else {"enabled": False}
        if isinstance(sizing_cfg, dict) and sizing_cfg.get('enabled', False):
            min_fraction = float(sizing_cfg.get('min_fraction', 0.2))
            max_fraction = float(sizing_cfg.get('max_fraction', 1.0))
        else:
            min_fraction = 1.0
            max_fraction = 1.0

        if prediction == 1 and cash > 0:  # SINAL DE COMPRA
            # Fracção alvo (simplificada: usar 100% quando habilitado, ou min/max)
            target_fraction = max_fraction
            investable_cash = cash * target_fraction
            max_stocks = investable_cash / (estimated_execution_price * (1 + transaction_cost_pct))
            stocks_to_buy = max_stocks
            total_cost = stocks_to_buy * estimated_execution_price * (1 + transaction_cost_pct)
            
            if cash >= total_cost:
                stocks_held += stocks_to_buy
                cash -= total_cost
                trade_count += 1
                buy_count += 1
                logger.info(f"COMPRA: {current_date.date()} | Preço: R$ {estimated_execution_price:.2f} | Ações: {stocks_to_buy:.0f} | Custo: R$ {total_cost:.2f}")
            else:
                logger.warning(f"COMPRA REJEITADA: Cash insuficiente (R$ {cash:.2f} < R$ {total_cost:.2f})")
                
        # Lógica de venda sem data leakage
        elif prediction == 0 and stocks_held > 0:  # SINAL DE VENDA
            sale_value = stocks_held * estimated_execution_price
            transaction_cost = sale_value * transaction_cost_pct
            net_sale_value = sale_value - transaction_cost
            
            cash += net_sale_value
            trade_count += 1
            sell_count += 1
            logger.info(f"VENDA: {current_date.date()} | Preço: R$ {estimated_execution_price:.2f} | Ações: {stocks_held:.0f} | Valor líquido: R$ {net_sale_value:.2f}")
            stocks_held = 0
        
        # Marcar o valor do portfólio no fechamento do dia corrente
        current_portfolio_value = cash + stocks_held * test_df['Close'].iloc[i]
        portfolio_history.append(current_portfolio_value)
    
    # Resumo detalhado do backtesting
    final_portfolio_value = cash + stocks_held * test_df['Close'].iloc[-1]
    
    logger.info("RESUMO FINAL DO BACKTESTING:")
    logger.info(f"  Total de trades: {trade_count}")
    logger.info(f"  Compras realizadas: {buy_count}")
    logger.info(f"  Vendas realizadas: {sell_count}")
    logger.info(f"  Capital inicial: R$ {initial_capital:,.2f}")
    logger.info(f"  Capital final: R$ {final_portfolio_value:,.2f}")
    logger.info(f"  Cash final: R$ {cash:,.2f}")
    logger.info(f"  Ações finais: {stocks_held:.0f}")
    logger.info(f"  Retorno total: {((final_portfolio_value/initial_capital)-1)*100:+.2f}%")
    
    if trade_count == 0:
        logger.warning("AVISO: Nenhum trade foi executado!")
        logger.warning("  - Verificar se há predições de alta (1) nas predições")
        logger.warning("  - Verificar se há predições de baixa (0) quando há ações")
        logger.warning("  - Verificar se os dados têm variação suficiente")
    
    return pd.Series(portfolio_history, index=test_df.index[:-1])

def run_buy_and_hold(test_df, initial_capital):

    price_col = 'Adj Close' if 'Adj Close' in test_df.columns else 'Close'
    first_day_price = test_df[price_col].iloc[0]
    stocks_bought = initial_capital / first_day_price

    portfolio_history = stocks_bought * test_df[price_col]

    return portfolio_history

def run_backtest_simple(test_df, predictions, initial_capital):

    if len(test_df) != len(predictions):
        min_size = min(len(test_df), len(predictions))
        test_df = test_df.iloc[:min_size]
        predictions = predictions[:min_size]

    cash = initial_capital
    stocks_held = 0.0
    portfolio_history = []

    for i in range(len(test_df)):
        price = test_df['Close'].iloc[i]
        signal = predictions[i]

        if signal == 1 and cash > 0:
            qty = cash / price
            stocks_held += qty
            cash = 0.0
        elif signal == 0 and stocks_held > 0:
            cash += stocks_held * price
            stocks_held = 0.0

        portfolio_history.append(cash + stocks_held * price)

    return pd.Series(portfolio_history, index=test_df.index)

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
        simple_results = [r for r in results_data if r['label'] == 'Modelo Simples']
        buyhold_results = [r for r in results_data if r['label'] == 'Buy and Hold']
        
        if model_results:
            avg_model_return = sum(r['retorno_total'] for r in model_results) / len(model_results)
            f.write(f"Retorno médio do modelo XGBoost: {avg_model_return:+.2f}%\n")
        if simple_results:
            avg_simple_return = sum(r['retorno_total'] for r in simple_results) / len(simple_results)
            f.write(f"Retorno médio do modelo simples: {avg_simple_return:+.2f}%\n")
        
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
            'model_simple': next((r for r in results if r['label'] == 'Modelo Simples'), None),
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
    simple_results = [r for r in results_data if r['label'] == 'Modelo Simples']
    if buyhold_results:
        report_data['summary']['buyhold_performance'] = {
            'average_return': sum(r['retorno_total'] for r in buyhold_results) / len(buyhold_results),
            'best_performer': max(buyhold_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(buyhold_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in buyhold_results)
        }
    if simple_results:
        report_data['summary']['simple_model_performance'] = {
            'average_return': sum(r['retorno_total'] for r in simple_results) / len(simple_results),
            'best_performer': max(simple_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(simple_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in simple_results)
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

    print(f"=== CONFIGURAÇÃO DE BACKTESTING ===")
    print(f"Capital inicial: R$ {backtest_config['initial_capital']:,.2f}")
    print(f"Custo de transação: {backtest_config['transaction_cost_pct']*100:.3f}%")
    print(f"Data inicial simulação: {backtest_config['initial_simulation_date']}")
    print(f"Data final simulação: {backtest_config['final_simulation_date']}")
    print(f"Período de simulação: {backtest_config['initial_simulation_date']} até {backtest_config['final_simulation_date']}")

    for feature_data_file in os.listdir(features_path):

        model_name = str(feature_data_file.replace(".csv", ".json"))
        ticker = feature_data_file.replace(".csv", "")

        print(f"PROCESSANDO: {ticker}")
        
        # Verificação da existência do modelo
        model_file_path = model_path / model_name
        if not model_file_path.exists():
            print(f"ERRO: Modelo não encontrado: {model_file_path}")
            continue
            
        df = pd.read_csv(features_path / feature_data_file, index_col='Date', parse_dates=True)
        
        # Verificação dos dados carregados
        print(f" Dados: {df.index[0].date()} até {df.index[-1].date()} ({len(df)} registros)")

        model = xgb.XGBClassifier()
        model.load_model(model_file_path)
        print(f"  Modelo carregado")

        # Criação do target com debug
        df = create_target_variable(df, config['model_training']['target_column'])
        
        # Split dos dados com verificação
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(
            df,
            config['model_training']['train_final_date'],
            config['model_training']['validation_start_date'],
            config['model_training']['validation_end_date'],
            config['model_training']['test_start_date'],
            config['model_training']['test_end_date'],
            config['model_training']['target_column']
        )
        
        # Otimização de thresholds (compra/venda) por métrica financeira na validação
        threshold_config = backtest_config.get('threshold_optimization', {})
        hold_min_days = threshold_config.get('hold_min_days', 3)

        # Reconstruir janela de validação com OHLC para simulação
        val_start = config['model_training']['validation_start_date']
        val_end = config['model_training']['validation_end_date']
        val_df_window = df[val_start:val_end]

        print(f"  Otimizando thresholds financeiros na validação...")
        buy_th, sell_th, best_sharpe = optimize_thresholds_financial(
            model,
            x_val,
            val_df_window,
            backtest_config['initial_capital'],
            backtest_config['transaction_cost_pct'],
            hold_min_days=hold_min_days
        )
        print(f"  Thresholds: buy={buy_th:.2f} sell={sell_th:.2f} | Sharpe validação={best_sharpe:.2f}")

        # Verificação do período de simulação
        simulation_df = df[backtest_config['initial_simulation_date']:backtest_config['final_simulation_date']]
        
        print(f"  Simulação: {len(simulation_df)} registros")
        
        # Verificação crítica dos dados de simulação
        if len(simulation_df) == 0:
            print(f"  ERRO: Nenhum dado para simulação")
            continue
        
        if len(simulation_df) < 50:
            print(f"  AVISO: Poucos dados ({len(simulation_df)})")
        
        # Criação das features para simulação
        x_simulation = simulation_df.drop(columns=[config['model_training']['target_column']])

        # Probabilidades e ações com zona neutra + histerese
        probabilities = model.predict_proba(x_simulation)[:, 1]
        predictions = generate_actions_from_probabilities(
            probabilities,
            buy_threshold=buy_th,
            sell_threshold=sell_th,
            hold_min_days=hold_min_days
        )

        print(f"  Ações geradas: {np.sum(predictions == 1)} dias comprado, {np.sum(predictions == 0)} dias zerado")
        print(f"  Probabilidade média de alta: {np.mean(probabilities):.3f}")
        
        # Verificação se há predições válidas
        if np.sum(predictions == 1) == 0:
            print(f"  AVISO: Nenhuma predição de alta!")
        
        if np.sum(predictions == 0) == 0:
            print(f"  AVISO: Nenhuma predição de baixa!")
        
        # Execução do backtesting com logging
        logger, log_filepath = setup_logging(ticker, backtest_config)
        print(f"  Log salvo em: {log_filepath.name}")
        
        # Log do threshold otimizado
        logger.info(f"THRESHOLDS FINANCEIROS: buy={buy_th:.2f} sell={sell_th:.2f} hold_min_days={hold_min_days}")
        logger.info(f"PROBABILIDADE MÉDIA: {np.mean(probabilities):.3f}")
        
        model_portfolio = run_backtest(
            simulation_df,
            predictions,
            backtest_config['initial_capital'],
            backtest_config['transaction_cost_pct'],
            logger
        )

        # Execução do backtesting simples (sem custos, execução no fechamento)
        simple_portfolio = run_backtest_simple(
            simulation_df,
            predictions,
            backtest_config['initial_capital']
        )

        # Execução do buy and hold
        buy_and_hold_portfolio = run_buy_and_hold(simulation_df, backtest_config['initial_capital'])
        
        # Cálculo das métricas
        model_metrics = calculate_metrics(model_portfolio, "Modelo de Predição", ticker)
        simple_metrics = calculate_metrics(simple_portfolio, "Simulação Simples", ticker)
        buy_and_hold_metrics = calculate_metrics(buy_and_hold_portfolio, "Buy and Hold", ticker)
        
        print(f"  Resultado: Modelo Realista {model_metrics['retorno_total']:+.2f}% | Modelo Simples {simple_metrics['retorno_total']:+.2f}% | Buy&Hold {buy_and_hold_metrics['retorno_total']:+.2f}%")
        
        results_simulated.append(model_metrics)
        results_simulated.append(simple_metrics)
        results_simulated.append(buy_and_hold_metrics)

        # Salvamento dos resultados
        results_df = pd.DataFrame({
            'Model_strategy_realistic': model_portfolio,
            'Model_strategy_simple': simple_portfolio,
            'Buy_and_Hold': buy_and_hold_portfolio
        })

        result_file_name = str(feature_data_file.replace(".csv", "_results.csv"))
        results_path = f"{backtest_config['results_path']}/{result_file_name}"
        os.makedirs(backtest_config['results_path'], exist_ok=True)
        results_df.to_csv(results_path)

    # Resumo final
    print(f"\n{'='*80}")
    print(f"RESUMO FINAL DO BACKTESTING")
    print(f"{'='*80}")
    
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
    
    # Estatísticas finais
    model_results = [r for r in results_simulated if r['label'] == 'Modelo de Predição']
    buyhold_results = [r for r in results_simulated if r['label'] == 'Buy and Hold']
    
    if model_results:
        avg_model_return = sum(r['retorno_total'] for r in model_results) / len(model_results)
        print(f"\nPERFORMANCE FINAL:")
        print(f"  - Retorno médio do modelo: {avg_model_return:+.2f}%")
        print(f"  - Total de tickers processados: {len(model_results)}")
    
    if buyhold_results:
        avg_buyhold_return = sum(r['retorno_total'] for r in buyhold_results) / len(buyhold_results)
        print(f"  - Retorno médio Buy&Hold: {avg_buyhold_return:+.2f}%")


if __name__ == "__main__":
    main()