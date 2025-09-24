"""
Módulo de Backtesting para Modelos de Trading

Este módulo implementa um sistema completo de backtesting para modelos de predição
de preços de ações, incluindo simulação realista com custos de transação,
otimização de thresholds e geração de relatórios detalhados.

Funcionalidades principais:
- Otimização de thresholds baseada em métricas financeiras
- Simulação de portfólio com execução realista
- Geração de relatórios em múltiplos formatos
- Sistema de logging estruturado
- Comparação com estratégias Buy & Hold
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import precision_recall_curve, classification_report

from ..models.train_models import split_data

# Constantes de configuração
DEFAULT_SLIPPAGE_BPS = 0.2
DEFAULT_LOT_SIZE = 100
DEFAULT_CASH_DAILY_RATE = 0.0
DEFAULT_HOLD_MIN_DAYS = 1
DEFAULT_BUY_GRID = np.arange(0.55, 0.86, 0.05)
DEFAULT_SELL_GRID = np.arange(0.15, 0.46, 0.05)
EXPANDED_BUY_GRID = np.arange(0.50, 0.95, 0.02)
EXPANDED_SELL_GRID = np.arange(0.05, 0.55, 0.02)

# Configurações de slippage realista
SLIPPAGE_CONFIG = {
    'base_bps': 0.2,           # Slippage base (0.2 bps)
    'min_bps': 0.1,            # Slippage mínimo (0.1 bps)
    'max_bps': 2.0,            # Slippage máximo (2.0 bps)
    'volume_factor': 0.5,      # Fator de impacto do volume
    'volatility_factor': 1.0,  # Fator de impacto da volatilidade
    'size_factor': 0.3         # Fator de impacto do tamanho da ordem
}

def calculate_ml_metrics(y_true_multiclass, y_pred_actions):
    """
    Calcula métricas de ML comparando ações 0/1 (long/flat) com rótulo binário up-vs-not-up.

    Teoria: O modelo é multiclasse (down=0, flat=1, up=2), mas a política é long/flat.
    Para avaliar a qualidade do sinal de compra, comparamos y_true_up = 1{classe==2}
    contra as ações 0/1 geradas por thresholds e histerese.
    """
    y_true_up = (pd.Series(y_true_multiclass).astype(int) == 2).astype(int)

    report_dict = classification_report(
        y_true_up, y_pred_actions,
        target_names=['Not-Up', 'Up'],
        output_dict=True,
        zero_division=0
    )

    up_metrics = report_dict.get('Up', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
    metrics = {
        "Acurácia": f"{report_dict.get('accuracy', 0.0):.2%}",
        "Precisão (Up)": f"{up_metrics['precision']:.2%}",
        "Recall (Up)": f"{up_metrics['recall']:.2%}",
        "F1-Score (Up)": f"{up_metrics['f1-score']:.2f}"
    }

    pred_counts = pd.Series(y_pred_actions).value_counts()
    metrics["Distribuição Predições"] = f"Up: {pred_counts.get(1, 0)}, Not-Up: {pred_counts.get(0, 0)}"

    return metrics

def generate_actions_from_score(score_series, buy_threshold=0.25, sell_threshold=-0.05, minimum_hold_days=3):
    """
    Converte score s = P(up) - P(down) em ações 0/1 com banda neutra e histerese.

    Teoria: s em [-1,1] cria um eixo único de convicção direcional. Bandas assimétricas
    (buy > 0, sell <= 0) e hold mínimo reduzem churn e melhoram robustez.
    """
    trading_actions = []
    current_position = 0
    days_since_last_change = minimum_hold_days

    for s in score_series:
        desired_position = current_position
        if s >= buy_threshold:
            desired_position = 1
        elif s <= sell_threshold:
            desired_position = 0

        if desired_position != current_position and days_since_last_change >= minimum_hold_days:
            current_position = desired_position
            days_since_last_change = 0
        else:
            days_since_last_change += 1

        trading_actions.append(current_position)

    return np.array(trading_actions, dtype=int)

def compute_up_down_score_from_proba(probabilities_matrix):
    """
    Calcula score s = P(up) - P(down) a partir de probabilidades multiclasse.

    Assumimos ordenação das classes: down=0, flat=1, up=2.
    """
    p_down = probabilities_matrix[:, 0]
    p_up = probabilities_matrix[:, 2]
    return p_up - p_down

def simulate_portfolio_execution_next_open(
    test_dataframe, 
    trading_actions, 
    initial_capital, 
    transaction_cost_percentage
):
    """
    Simula execução de portfólio com execução no próximo pregão (Open t+1).
    
    Esta função é otimizada para uso interno durante otimização de thresholds,
    sem logging detalhado para melhor performance.
    
    Args:
        test_dataframe: DataFrame com dados OHLCV para simulação
        trading_actions: Array 0/1 representando posição desejada para cada dia
        initial_capital: Capital inicial para simulação
        transaction_cost_percentage: Custo de transação como percentual (ex: 0.001 = 0.1%)
    
    Returns:
        pd.Series: Histórico do valor do portfólio indexado por data
    """
    available_cash = initial_capital
    stocks_quantity = 0.0
    portfolio_value_history = []
    
    # Simular execução dia a dia
    for day_index in range(len(test_dataframe) - 1):
        desired_position = trading_actions[day_index]
        next_day_open_price = test_dataframe['Open'].iloc[day_index + 1]

        # Lógica de compra: se deseja posição 1 e não tem ações
        if desired_position == 1 and stocks_quantity == 0 and available_cash > 0:
            # Calcular quantidade máxima que pode comprar considerando custos
            max_affordable_quantity = available_cash / (next_day_open_price * (1 + transaction_cost_percentage))
            total_transaction_cost = max_affordable_quantity * next_day_open_price * (1 + transaction_cost_percentage)
            
            if total_transaction_cost <= available_cash:
                stocks_quantity += max_affordable_quantity
                available_cash -= total_transaction_cost
                
        # Lógica de venda: se deseja posição 0 e tem ações
        elif desired_position == 0 and stocks_quantity > 0:
            sale_value = stocks_quantity * next_day_open_price
            net_sale_value = sale_value * (1 - transaction_cost_percentage)
            available_cash += net_sale_value
            stocks_quantity = 0.0

        # Marcar valor do portfólio no fechamento do dia atual
        current_portfolio_value = available_cash + stocks_quantity * test_dataframe['Close'].iloc[day_index]
        portfolio_value_history.append(current_portfolio_value)

    # Valor final do portfólio
    final_portfolio_value = available_cash + stocks_quantity * test_dataframe['Close'].iloc[-1]
    portfolio_value_history.append(final_portfolio_value)
    
    return pd.Series(portfolio_value_history, index=test_dataframe.index)

def calculate_dynamic_slippage(price, volume, order_size, volatility=None, slippage_config=None):
    """
    Calcula slippage dinâmico baseado em características do mercado e da ordem.
    
    Args:
        price: Preço atual da ação
        volume: Volume negociado no dia
        order_size: Tamanho da ordem (quantidade de ações)
        volatility: Volatilidade do dia (opcional, calculada se não fornecida)
        slippage_config: Configuração do slippage (usa padrão se não fornecida)
    
    Returns:
        float: Slippage em basis points (bps)
    """
    if slippage_config is None:
        slippage_config = SLIPPAGE_CONFIG
    
    # Slippage base
    base_slippage = slippage_config['base_bps']
    
    # Fator de volume (maior volume = menor slippage)
    if volume > 0:
        # Normalizar volume (assumindo volume médio de 1M ações)
        volume_factor = min(1.0, 1000000 / volume)
        volume_impact = volume_factor * slippage_config['volume_factor']
    else:
        volume_impact = 1.0
    
    # Fator de volatilidade (maior volatilidade = maior slippage)
    if volatility is not None:
        # Normalizar volatilidade (assumindo volatilidade média de 2%)
        volatility_factor = min(2.0, volatility / 0.02)
        volatility_impact = volatility_factor * slippage_config['volatility_factor']
    else:
        volatility_impact = 1.0
    
    # Fator de tamanho da ordem (ordens maiores = maior slippage)
    # Normalizar tamanho da ordem (assumindo ordem média de 1000 ações)
    size_factor = min(2.0, order_size / 1000)
    size_impact = size_factor * slippage_config['size_factor']
    
    # Calcular slippage total
    total_slippage = base_slippage * (1 + volume_impact + volatility_impact + size_impact)
    
    # Aplicar limites
    min_slippage = slippage_config['min_bps']
    max_slippage = slippage_config['max_bps']
    
    return max(min_slippage, min(max_slippage, total_slippage))

def calculate_position_fraction(probability, sizing_config):
    """
    Calcula a fração do capital a ser investida baseada na probabilidade de predição.
    
    Args:
        probability: Probabilidade de predição (0 a 1)
        sizing_config: Configuração do position sizing
    
    Returns:
        float: Fração do capital a ser investida (0 a 1)
    """
    if not isinstance(sizing_config, dict) or not sizing_config.get('enabled', False):
        return 1.0
    
    min_fraction = float(sizing_config.get('min_fraction', 0.2))
    max_fraction = float(sizing_config.get('max_fraction', 1.0))
    method = sizing_config.get('method', 'linear')
    
    # Garantir que min_fraction <= max_fraction
    min_fraction = min(min_fraction, max_fraction)
    max_fraction = max(min_fraction, max_fraction)
    
    if method == 'linear':
        # Mapeamento linear: prob 0.5 -> min_fraction, prob 1.0 -> max_fraction
        # Para probabilidades < 0.5, usar min_fraction
        if probability < 0.5:
            return min_fraction
        else:
            # Mapear [0.5, 1.0] para [min_fraction, max_fraction]
            normalized_prob = (probability - 0.5) / 0.5
            return min_fraction + normalized_prob * (max_fraction - min_fraction)
    
    elif method == 'sigmoid':
        # Mapeamento sigmóide para suavizar transições
        # Centro em 0.6, com sensibilidade ajustável
        center = 0.6
        sensitivity = 10.0
        sigmoid = 1 / (1 + np.exp(-sensitivity * (probability - center)))
        return min_fraction + sigmoid * (max_fraction - min_fraction)
    
    elif method == 'step':
        # Mapeamento em degraus baseado em thresholds
        if probability >= 0.8:
            return max_fraction
        elif probability >= 0.6:
            return min_fraction + 0.6 * (max_fraction - min_fraction)
        elif probability >= 0.4:
            return min_fraction + 0.3 * (max_fraction - min_fraction)
        else:
            return min_fraction
    
    elif method == 'quadratic':
        # Mapeamento quadrático para dar mais peso a probabilidades altas
        if probability < 0.5:
            return min_fraction
        else:
            normalized_prob = (probability - 0.5) / 0.5
            quadratic_prob = normalized_prob ** 2
            return min_fraction + quadratic_prob * (max_fraction - min_fraction)
    
    else:
        # Fallback para linear se método não reconhecido
        return calculate_position_fraction(probability, {**sizing_config, 'method': 'linear'})

def calculate_sharpe_ratio(portfolio_value_series):
    """
    Calcula o Índice de Sharpe anualizado para uma série de valores de portfólio.
    
    O Sharpe Ratio mede o retorno ajustado ao risco, considerando a volatilidade
    dos retornos. Valores mais altos indicam melhor performance ajustada ao risco.
    
    Args:
        portfolio_value_series: Série temporal com valores do portfólio
    
    Returns:
        float: Índice de Sharpe anualizado (252 dias úteis por ano)
    """
    # Calcular retornos diários
    daily_returns = portfolio_value_series.pct_change().dropna()
    
    # Verificar se há dados suficientes e variância não nula
    if daily_returns.std() == 0 or len(daily_returns) == 0:
        return -np.inf
    
    # Calcular Sharpe Ratio anualizado
    # Assumindo 252 dias úteis por ano
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    return sharpe_ratio

def adaptive_threshold_optimization(model, x_validation, validation_dataframe, initial_capital, transaction_cost_percentage, minimum_hold_days=3):

    coarse_buy_grid = np.arange(0.50, 0.95, 0.10)
    coarse_sell_grid = np.arange(0.05, 0.55, 0.10)

    best_buy_coarse, best_sell_coarse, _ = optimize_trading_thresholds_financial(
        model, x_validation, validation_dataframe, initial_capital, transaction_cost_percentage, buy_threshold_grid=coarse_buy_grid, sell_threshold_grid=coarse_sell_grid, minimum_hold_days=minimum_hold_days
    )

    fine_buy_grid = np.arange(max(0.50, best_buy_coarse - 0.1), 
                              min(0.95, best_buy_coarse + 0.1), 0.01)
    fine_sell_grid = np.arange(max(0.05, best_sell_coarse - 0.1), 
                               min(0.55, best_sell_coarse + 0.1), 0.01)

    return optimize_trading_thresholds_financial(
        model, x_validation, validation_dataframe, initial_capital, transaction_cost_percentage, buy_threshold_grid=fine_buy_grid, sell_threshold_grid=fine_sell_grid, minimum_hold_days=minimum_hold_days
    )

def optimize_trading_thresholds_financial(
    model, 
    x_validation,
    validation_dataframe, 
    initial_capital, 
    transaction_cost_percentage, 
    buy_threshold_grid=None, 
    sell_threshold_grid=None, 
    minimum_hold_days=3
):
    """
    Otimiza thresholds de compra e venda para maximizar o Sharpe Ratio no conjunto de validação.
    
    Esta função realiza uma busca em grade para encontrar a combinação de thresholds
    que maximiza o Sharpe Ratio, considerando custos de transação e período mínimo de manutenção.
    
    Args:
        model: Modelo XGBoost treinado
        x_validation: Features do conjunto de validação
        validation_dataframe: DataFrame OHLCV para simulação de validação
        initial_capital: Capital inicial para simulação
        transaction_cost_percentage: Custo de transação como percentual
        buy_threshold_grid: Grade de thresholds de compra para testar
        sell_threshold_grid: Grade de thresholds de venda para testar
        minimum_hold_days: Mínimo de dias entre mudanças de posição
    
    Returns:
        tuple: (best_buy_threshold, best_sell_threshold, best_sharpe_ratio)
    """
    # Definir grades padrão se não fornecidas
    if buy_threshold_grid is None:
        buy_threshold_grid = DEFAULT_BUY_GRID
    if sell_threshold_grid is None:
        sell_threshold_grid = DEFAULT_SELL_GRID

    # Probabilidades multiclasse e score s = P(up) - P(down)
    probabilities = model.predict_proba(x_validation)
    score = compute_up_down_score_from_proba(probabilities)

    # Inicializar variáveis de otimização
    best_sharpe_ratio = -np.inf
    best_threshold_pair = (0.5, 0.5)

    # Busca em grade pelos melhores thresholds
    for buy_threshold in buy_threshold_grid:
        for sell_threshold in sell_threshold_grid:
            # Pular combinações inválidas (sell >= buy)
            if sell_threshold >= buy_threshold:
                continue
                
            # Gerar ações de trading com thresholds atuais
            trading_actions = generate_actions_from_score(
                score,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                minimum_hold_days=minimum_hold_days
            )
            
            # Alinhar tamanhos dos dados se necessário
            data_size = min(len(validation_dataframe), len(trading_actions))
            validation_data_subset = validation_dataframe.iloc[:data_size]
            actions_subset = trading_actions[:data_size]
            
            # Simular portfólio e calcular Sharpe
            portfolio_simulation = simulate_portfolio_execution_next_open(
                validation_data_subset, 
                actions_subset, 
                initial_capital, 
                transaction_cost_percentage
            )
            current_sharpe_ratio = calculate_sharpe_ratio(portfolio_simulation)
            
            # Atualizar melhor combinação se encontrou Sharpe maior
            if current_sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = current_sharpe_ratio
                best_threshold_pair = (buy_threshold, sell_threshold)

    return best_threshold_pair[0], best_threshold_pair[1], best_sharpe_ratio

def setup_structured_logging(ticker_symbol, backtest_configuration):
    """
    Configura sistema de logging estruturado para cada ticker.
    
    Cria um logger específico para cada ticker com arquivo de log único,
    incluindo timestamp para evitar conflitos entre execuções.
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: 'PETR4.SA')
        backtest_configuration: Configuração do backtest contendo 'results_path'
    
    Returns:
        tuple: (logger, log_file_path)
            - logger: Objeto logger configurado
            - log_file_path: Caminho do arquivo de log criado
    """
    # Criar diretório de logs se não existir
    logs_directory = Path(backtest_configuration['results_path']) / "logs"
    logs_directory.mkdir(exist_ok=True)
    
    # Gerar nome único do arquivo de log com timestamp
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{ticker_symbol}_{current_timestamp}.log"
    log_file_path = logs_directory / log_filename
    
    # Configurar logger específico para o ticker
    logger_name = f"backtest_{ticker_symbol}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Evitar duplicação de handlers (limpar handlers existentes)
    if logger.handlers:
        logger.handlers.clear()
    
    # Configurar handler para arquivo
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Definir formato das mensagens de log
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    
    logger.addHandler(file_handler)
    
    return logger, log_file_path

def _load_cdi_data(backtest_configuration, logger):
    """
    Carrega dados de CDI real se habilitado na configuração.
    
    Args:
        backtest_configuration: Configuração do backtest
        logger: Logger para mensagens
    
    Returns:
        pd.Series ou None: Dados de CDI ou None se não disponível
    """
    if not backtest_configuration.get('use_real_cdi', False):
        return None
        
    cdi_file_path = Path(__file__).resolve().parents[2] / "src" / "renda_fixa_simulação" / "cdi_daily.csv"
    
    if cdi_file_path.exists():
        cdi_dataframe = pd.read_csv(cdi_file_path, index_col='Date', parse_dates=True)
        cdi_series = cdi_dataframe['cdi'] / 100.0  # Converter de % para decimal
        logger.info(f"CDI real carregado: {len(cdi_series)} dias disponíveis")
        return cdi_series
    else:
        logger.warning(f"Arquivo CDI não encontrado: {cdi_file_path}")
        return None

def _apply_daily_cash_return(cash_amount, current_date, cdi_data, fixed_daily_rate, logger):
    """
    Aplica rendimento diário ao caixa disponível.
    
    Args:
        cash_amount: Quantia em caixa
        current_date: Data atual
        cdi_data: Dados de CDI real (se disponível)
        fixed_daily_rate: Taxa fixa diária (se CDI real não disponível)
        logger: Logger para mensagens
    
    Returns:
        float: Nova quantia em caixa após aplicação do rendimento
    """
    if cash_amount <= 0:
        return cash_amount
        
    if cdi_data is not None:
        # Usar CDI real do dia
        if current_date in cdi_data.index:
            daily_rate = cdi_data[current_date]
            return cash_amount * (1.0 + daily_rate)
        else:
            # Se não encontrar data exata, usar última disponível
            available_dates = cdi_data.index[cdi_data.index <= current_date]
            if len(available_dates) > 0:
                last_available_date = available_dates[-1]
                daily_rate = cdi_data[last_available_date]
                return cash_amount * (1.0 + daily_rate)
    elif fixed_daily_rate != 0.0:
        # Usar taxa fixa se CDI real não estiver disponível
        return cash_amount * (1.0 + fixed_daily_rate)
    
    return cash_amount

def _execute_buy_order(cash, stocks_held, execution_price, transaction_cost_pct, 
                      lot_size, sizing_config, probability, volume, volatility, 
                      slippage_config, logger, current_date):
    """
    Executa ordem de compra com todos os custos e restrições, incluindo position sizing dinâmico.
    
    Args:
        cash: Caixa disponível
        stocks_held: Quantidade de ações já possuídas
        execution_price: Preço de execução base
        transaction_cost_pct: Custo de transação como percentual
        lot_size: Tamanho do lote mínimo
        sizing_config: Configuração do position sizing
        probability: Probabilidade de predição (0 a 1)
        volume: Volume negociado no dia
        volatility: Volatilidade do dia
        slippage_config: Configuração do slippage
        logger: Logger para mensagens
        current_date: Data atual
    
    Returns:
        tuple: (new_cash, new_stocks_held, trade_executed, trade_cost)
    """
    # Calcular fração do capital baseada na probabilidade
    target_fraction = calculate_position_fraction(probability, sizing_config)
    investable_cash = cash * target_fraction
    max_affordable_stocks = investable_cash / (execution_price * (1 + transaction_cost_pct))
    
    # Arredondar para lote
    stocks_to_buy = np.floor(max_affordable_stocks / lot_size) * lot_size
    
    # Calcular slippage dinâmico baseado no tamanho da ordem
    dynamic_slippage_bps = calculate_dynamic_slippage(
        execution_price, volume, stocks_to_buy, volatility, slippage_config
    )
    slippage_pct = dynamic_slippage_bps / 10000.0
    
    # Calcular preço com slippage dinâmico
    buy_price = execution_price * (1 + slippage_pct)
    
    # Recalcular quantidade com preço final
    max_affordable_stocks = investable_cash / (buy_price * (1 + transaction_cost_pct))
    stocks_to_buy = np.floor(max_affordable_stocks / lot_size) * lot_size
    total_transaction_cost = stocks_to_buy * buy_price * (1 + transaction_cost_pct)
    
    if cash >= total_transaction_cost and stocks_to_buy > 0:
        new_cash = cash - total_transaction_cost
        new_stocks_held = stocks_held + stocks_to_buy
        logger.info(f"COMPRA: {current_date.date()} | Preço base: R$ {execution_price:.2f} | "
                   f"Preço exec c/ slippage: R$ {buy_price:.2f} | Ações: {stocks_to_buy:.0f} | "
                   f"Custo: R$ {total_transaction_cost:.2f} | Prob: {probability:.3f} | "
                   f"Fração: {target_fraction:.1%}")
        return new_cash, new_stocks_held, True, total_transaction_cost
    else:
        if stocks_to_buy == 0:
            logger.warning(f"COMPRA REJEITADA: Quantidade calculada = 0 (prob: {probability:.3f}, "
                          f"fração: {target_fraction:.1%})")
        else:
            logger.warning(f"COMPRA REJEITADA: Cash insuficiente (R$ {cash:.2f} < R$ {total_transaction_cost:.2f})")
        return cash, stocks_held, False, 0

def _execute_sell_order(cash, stocks_held, execution_price, transaction_cost_pct, 
                       volume, volatility, slippage_config, logger, current_date):
    """
    Executa ordem de venda com todos os custos e slippage dinâmico.
    
    Args:
        cash: Caixa disponível
        stocks_held: Quantidade de ações possuídas
        execution_price: Preço de execução base
        transaction_cost_pct: Custo de transação como percentual
        volume: Volume negociado no dia
        volatility: Volatilidade do dia
        slippage_config: Configuração do slippage
        logger: Logger para mensagens
        current_date: Data atual
    
    Returns:
        tuple: (new_cash, new_stocks_held, trade_executed, trade_value)
    """
    # Calcular slippage dinâmico para venda
    dynamic_slippage_bps = calculate_dynamic_slippage(
        execution_price, volume, stocks_held, volatility, slippage_config
    )
    slippage_pct = dynamic_slippage_bps / 10000.0
    
    # Calcular preço com slippage dinâmico
    sell_price = execution_price * (1 - slippage_pct)
    gross_sale_value = stocks_held * sell_price
    transaction_cost = gross_sale_value * transaction_cost_pct
    net_sale_value = gross_sale_value - transaction_cost
    
    new_cash = cash + net_sale_value
    new_stocks_held = 0.0
    
    logger.info(f"VENDA: {current_date.date()} | Preço base: R$ {execution_price:.2f} | "
               f"Preço exec c/ slippage: R$ {sell_price:.2f} | Ações: {stocks_held:.0f} | "
               f"Valor líquido: R$ {net_sale_value:.2f} | Slippage: {dynamic_slippage_bps:.1f} bps")
    
    return new_cash, new_stocks_held, True, net_sale_value

def run_realistic_backtest(test_dataframe, predictions, probabilities, initial_capital, transaction_cost_percentage, 
                          logger, backtest_configuration=None):
    """
    Executa backtesting realista com logging estruturado e custos de transação.
    
    Esta função implementa um backtesting completo considerando:
    - Execução no próximo pregão (Open t+1)
    - Custos de transação e slippage
    - Position sizing dinâmico baseado em probabilidade
    - Rendimento diário do caixa (CDI real ou fixo)
    - Logging detalhado de todas as operações
    
    Args:
        test_dataframe: DataFrame com dados OHLCV para simulação
        predictions: Array de predições binárias (0 ou 1)
        probabilities: Array de probabilidades de predição (0 a 1)
        initial_capital: Capital inicial para simulação
        transaction_cost_percentage: Custo de transação como percentual
        logger: Logger configurado para o ticker
        backtest_configuration: Configuração adicional do backtest
    
    Returns:
        pd.Series: Histórico do valor do portfólio indexado por data
    """
    # Inicializar logging
    logger.info("="*60)
    logger.info("INICIANDO BACKTESTING REALISTA")
    logger.info("="*60)
    
    # Log de parâmetros iniciais
    logger.info(f"Tamanho test_df: {len(test_dataframe)}")
    logger.info(f"Tamanho predictions: {len(predictions)}")
    logger.info(f"Tamanho probabilities: {len(probabilities)}")
    logger.info(f"Capital inicial: R$ {initial_capital:,.2f}")
    logger.info(f"Custo de transação: {transaction_cost_percentage*100:.3f}%")
    
    # Carregar configurações adicionais
    lot_size = DEFAULT_LOT_SIZE
    cash_daily_rate = DEFAULT_CASH_DAILY_RATE
    sizing_config = {"enabled": True}
    slippage_config = SLIPPAGE_CONFIG.copy()
    cdi_data = None
    
    if isinstance(backtest_configuration, dict):
        lot_size = int(backtest_configuration.get('lot_size', DEFAULT_LOT_SIZE))
        cash_daily_rate = float(backtest_configuration.get('cash_daily_rate', DEFAULT_CASH_DAILY_RATE))
        sizing_config = backtest_configuration.get('position_sizing', {"enabled": False}) or {"enabled": False}
        cdi_data = _load_cdi_data(backtest_configuration, logger)
        
        # Configuração de slippage personalizada se fornecida
        if 'slippage_config' in backtest_configuration:
            slippage_config.update(backtest_configuration['slippage_config'])
    
    logger.info(f"Slippage Dinâmico: {slippage_config['base_bps']:.1f}-{slippage_config['max_bps']:.1f} bps | "
               f"Lote: {lot_size} | CDI: {'Real' if cdi_data is not None else f'Fixo {cash_daily_rate*100:.3f}%'}")
    
    # Calcular volatilidade histórica (20 dias)
    test_dataframe['volatility'] = test_dataframe['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # Verificar e corrigir alinhamento entre dados, predições e probabilidades
    min_size = min(len(test_dataframe), len(predictions), len(probabilities))
    if len(test_dataframe) != min_size or len(predictions) != min_size or len(probabilities) != min_size:
        logger.warning(f"ALINHAMENTO: test_df ({len(test_dataframe)}) != predictions ({len(predictions)}) != probabilities ({len(probabilities)})")
        test_dataframe = test_dataframe.iloc[:min_size]
        predictions = predictions[:min_size]
        probabilities = probabilities[:min_size]
        logger.info(f"ALINHAMENTO: Ajustado para {min_size} registros")
    
    # Log da distribuição das predições
    logger.info("DISTRIBUIÇÃO DAS PREDIÇÕES:")
    logger.info(f"  - Total: {len(predictions)}")
    logger.info(f"  - Predições de alta (1): {np.sum(predictions == 1)}")
    logger.info(f"  - Predições de baixa (0): {np.sum(predictions == 0)}")
    logger.info(f"  - Percentual de alta: {np.sum(predictions == 1)/len(predictions)*100:.1f}%")
    
    # Inicializar variáveis do portfólio
    available_cash = initial_capital
    stocks_quantity = 0.0
    portfolio_value_history = []
    total_trades = 0
    buy_trades = 0
    sell_trades = 0
    
    logger.info("EXECUÇÃO DOS TRADES:")
    
    # Simular execução dia a dia
    for day_index in range(len(test_dataframe) - 1):
        current_date = test_dataframe.index[day_index]
        current_prediction = predictions[day_index]
        
        # Log de progresso a cada 50 iterações
        if day_index % 50 == 0:
            logger.info(f"PROGRESSO: Data {current_date.date()} | Pred: {current_prediction} | "
                       f"Cash: R$ {available_cash:,.2f} | Ações: {stocks_quantity:.0f}")
        
        if day_index >= len(predictions):
            break
        
        # Preço de execução no próximo pregão
        next_day_open_price = test_dataframe['Open'].iloc[day_index + 1]
        
        # Executar trades baseados na predição
        if current_prediction == 1 and available_cash > 0:  # SINAL DE COMPRA
            current_probability = probabilities[day_index] if day_index < len(probabilities) else 0.5
            current_volume = test_dataframe['Volume'].iloc[day_index] if 'Volume' in test_dataframe.columns else 1000000
            current_volatility = test_dataframe['volatility'].iloc[day_index] if 'volatility' in test_dataframe.columns else 0.02
            
            available_cash, stocks_quantity, trade_executed, trade_cost = _execute_buy_order(
                available_cash, stocks_quantity, next_day_open_price, transaction_cost_percentage,
                lot_size, sizing_config, current_probability, current_volume, current_volatility,
                slippage_config, logger, current_date
            )
            if trade_executed:
                total_trades += 1
                buy_trades += 1
                
        elif current_prediction == 0 and stocks_quantity > 0:  # SINAL DE VENDA
            current_volume = test_dataframe['Volume'].iloc[day_index] if 'Volume' in test_dataframe.columns else 1000000
            current_volatility = test_dataframe['volatility'].iloc[day_index] if 'volatility' in test_dataframe.columns else 0.02
            
            available_cash, stocks_quantity, trade_executed, trade_value = _execute_sell_order(
                available_cash, stocks_quantity, next_day_open_price, transaction_cost_percentage,
                current_volume, current_volatility, slippage_config, logger, current_date
            )
            if trade_executed:
                total_trades += 1
                sell_trades += 1
        
        # # Aplicar rendimento diário ao caixa
        # available_cash = _apply_daily_cash_return(
        #     available_cash, current_date, cdi_data, cash_daily_rate, logger
        # )
        
        # Marcar valor do portfólio no fechamento do dia
        current_portfolio_value = available_cash + stocks_quantity * test_dataframe['Close'].iloc[day_index]
        portfolio_value_history.append(current_portfolio_value)
    
    # Calcular valor final do portfólio
    final_portfolio_value = available_cash + stocks_quantity * test_dataframe['Close'].iloc[-1]
    
    # Log do resumo final
    logger.info("RESUMO FINAL DO BACKTESTING:")
    logger.info(f"  Total de trades: {total_trades}")
    logger.info(f"  Compras realizadas: {buy_trades}")
    logger.info(f"  Vendas realizadas: {sell_trades}")
    logger.info(f"  Capital inicial: R$ {initial_capital:,.2f}")
    logger.info(f"  Capital final: R$ {final_portfolio_value:,.2f}")
    logger.info(f"  Cash final: R$ {available_cash:,.2f}")
    logger.info(f"  Ações finais: {stocks_quantity:.0f}")
    logger.info(f"  Retorno total: {((final_portfolio_value/initial_capital)-1)*100:+.2f}%")
    
    if total_trades == 0:
        logger.warning("AVISO: Nenhum trade foi executado!")
        logger.warning("  - Verificar se há predições de alta (1) nas predições")
        logger.warning("  - Verificar se há predições de baixa (0) quando há ações")
        logger.warning("  - Verificar se os dados têm variação suficiente")
    
    return pd.Series(portfolio_value_history, index=test_dataframe.index[:-1])

def run_buy_and_hold_strategy(test_dataframe, initial_capital):
    """
    Simula estratégia Buy and Hold (comprar e manter).
    
    Esta estratégia compra ações no primeiro dia e mantém até o final,
    sem realizar nenhuma venda ou compra adicional.
    
    Args:
        test_dataframe: DataFrame com dados OHLCV para simulação
        initial_capital: Capital inicial para investimento
    
    Returns:
        pd.Series: Histórico do valor do portfólio indexado por data
    """
    # Usar preço ajustado se disponível, senão usar preço de fechamento
    price_column = 'Adj Close' if 'Adj Close' in test_dataframe.columns else 'Close'
    first_day_price = test_dataframe[price_column].iloc[0]
    
    # Calcular quantidade de ações que pode comprar no primeiro dia
    stocks_quantity = initial_capital / first_day_price
    
    # Simular portfólio mantendo a mesma quantidade de ações
    portfolio_value_history = stocks_quantity * test_dataframe[price_column]
    
    return portfolio_value_history

def run_profit_techniques_backtest(test_dataframe, predictions, probabilities, initial_capital, logger, backtest_configuration=None):

    cash_daily_rate = DEFAULT_CASH_DAILY_RATE
    cdi_data = None

    avaliable_cash = initial_capital
    portfolio_value_history = []
    stocks_quantity = 0.0
    total_trades = 0
    buy_trades = 0
    sell_trades = 0

    for day_index in range(len(test_dataframe) - 1):

        current_prediction = predictions[day_index]
        
        next_day_open_price = test_dataframe['Open'].iloc[day_index + 1]
        current_date = test_dataframe.index[day_index]

        if current_prediction == 1 and avaliable_cash > 0:

            current_probability = probabilities[day_index] if day_index < len(probabilities) else 0.5

            stocks_to_buy = avaliable_cash / next_day_open_price
            stocks_quantity += stocks_to_buy
            avaliable_cash = 0.0

            total_trades += 1
            buy_trades += 1
        
        elif current_prediction == 0 and stocks_quantity > 0:

            stocks_to_sell = stocks_quantity
            stocks_quantity -= stocks_to_sell
            avaliable_cash += stocks_to_sell * next_day_open_price
            
            total_trades += 1
            sell_trades += 1
            
        
        # # Aplicar rendimento diário ao caixa
        # avaliable_cash = _apply_daily_cash_return(
        #     avaliable_cash, current_date, cdi_data, cash_daily_rate, logger
        # )

        current_portfolio_value = avaliable_cash + stocks_quantity * test_dataframe['Close'].iloc[day_index]

        portfolio_value_history.append(current_portfolio_value)

    final_portfolio_value = avaliable_cash + stocks_quantity * test_dataframe['Close'].iloc[-1]

    return pd.Series(portfolio_value_history, index=test_dataframe.index[:-1])

def run_simple_backtest(test_dataframe, predictions, initial_capital):
    """
    Executa backtesting simplificado sem custos de transação.
    
    Esta versão simplificada executa trades no fechamento do mesmo dia
    e não considera custos de transação, sendo útil para comparação rápida.
    
    Args:
        test_dataframe: DataFrame com dados OHLCV para simulação
        predictions: Array de predições binárias (0 ou 1)
        initial_capital: Capital inicial para simulação
    
    Returns:
        pd.Series: Histórico do valor do portfólio indexado por data
    """
    # Verificar e corrigir alinhamento entre dados e predições
    if len(test_dataframe) != len(predictions):
        min_size = min(len(test_dataframe), len(predictions))
        test_dataframe = test_dataframe.iloc[:min_size]
        predictions = predictions[:min_size]

    # Inicializar variáveis do portfólio
    available_cash = initial_capital
    stocks_quantity = 0.0
    portfolio_value_history = []

    # Simular execução dia a dia
    for day_index in range(len(test_dataframe) - 1):
        current_price = test_dataframe['Open'].iloc[day_index + 1]
        current_signal = predictions[day_index]

        # Lógica de compra: sinal 1 e tem caixa
        if current_signal == 1 and available_cash > 0:
            quantity_to_buy = available_cash / current_price
            stocks_quantity += quantity_to_buy
            available_cash = 0.0
            
        # Lógica de venda: sinal 0 e tem ações
        elif current_signal == 0 and stocks_quantity > 0:
            available_cash += stocks_quantity * current_price
            stocks_quantity = 0.0

        # Marcar valor do portfólio no fechamento do dia
        current_portfolio_value = available_cash + stocks_quantity * current_price
        portfolio_value_history.append(current_portfolio_value)

    return pd.Series(portfolio_value_history, index=test_dataframe.index[:-1])

def calculate_portfolio_metrics(portfolio_value_history, strategy_label, ticker_symbol):
    """
    Calcula métricas básicas de performance do portfólio.
    
    Args:
        portfolio_value_history: Série temporal com valores do portfólio
        strategy_label: Nome da estratégia (ex: 'Modelo de Predição')
        ticker_symbol: Símbolo do ticker (ex: 'PETR4.SA')
    
    Returns:
        dict: Dicionário com métricas calculadas
    """
    initial_capital = portfolio_value_history.iloc[0]
    final_capital = portfolio_value_history.iloc[-1]
    total_return_percentage = (final_capital / initial_capital - 1) * 100
    
    metrics = {
        'ticker': ticker_symbol,
        'label': strategy_label,
        'capital_inicial': initial_capital,
        'capital_final': final_capital,
        'retorno_total': total_return_percentage
    }
    
    return metrics

def generate_text_report(results_data, backtest_configuration, output_path):
    """
    Gera relatório em formato TXT legível com resultados do backtesting.
    
    Args:
        results_data: Lista de dicionários com resultados de cada estratégia
        backtest_configuration: Configuração do backtest
        output_path: Caminho base para salvar o relatório
    """
    txt_file_path = output_path.replace('.csv', '.txt')
    # print(results_data)
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        # Cabeçalho do relatório
        file.write("=" * 80 + "\n")
        file.write("RELATÓRIO DE BACKTESTING - MODELO XGBOOST vs BUY AND HOLD\n")
        file.write("=" * 80 + "\n\n")
        
        # Informações da simulação
        file.write(f"Data da simulação: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write(f"Capital inicial: R$ {backtest_configuration['initial_capital']:,.2f}\n")
        file.write(f"Custo de transação: {backtest_configuration['transaction_cost_pct']*100:.3f}%\n\n")
        
        # Agrupar resultados por ticker
        ticker_results = _group_results_by_ticker(results_data)

        # Seção de resultados por ticker
        file.write("-" * 80 + "\n")
        file.write("RESULTADOS MONETÁRIOS E DE MÉTRICAS POR TICKER\n")
        file.write("-" * 80 + "\n\n")
        
        for ticker in sorted(ticker_results.keys()):
            file.write(f"TICKER: {ticker}\n")
            file.write("-" * 40 + "\n")
            
            for result in ticker_results[ticker]:
                # Apenas entradas com métricas financeiras possuem estes campos
                if all(key in result for key in ['label', 'capital_inicial', 'capital_final', 'retorno_total']):
                    file.write(f"Estratégia: {result['label']}\n")
                    file.write(f"  Capital Inicial: R$ {result['capital_inicial']:,.2f}\n")
                    file.write(f"  Capital Final: R$ {result['capital_final']:,.2f}\n")
                    file.write(f"  Retorno Total: {result['retorno_total']:+.2f}%\n")
                    file.write("\n")


                if 'ml_metrics' in result:
                    file.write("Métricas de ML:\n")
                    file.write("-" * 40 + "\n")

                    ml_metrics = result['ml_metrics']
                    for metric, value in ml_metrics.items():
                        file.write(f"{metric}: {value}\n")
                    file.write("\n")
            file.write("\n")
        
        # Resumo geral
        file.write("=" * 80 + "\n")
        file.write("RESUMO GERAL\n")
        file.write("=" * 80 + "\n\n")
        
        _write_summary_statistics(file, results_data)

def _group_results_by_ticker(results_data):
    """
    Agrupa resultados por ticker para facilitar exibição.
    
    Args:
        results_data: Lista de dicionários com resultados
    
    Returns:
        dict: Dicionário com ticker como chave e lista de resultados como valor
    """
    ticker_results = {}
    print(results_data)
    for result in results_data:
        print(result)
        ticker = result['ticker']
        print(ticker)
        if ticker not in ticker_results:
            ticker_results[ticker] = []
        ticker_results[ticker].append(result)
    return ticker_results

def _write_summary_statistics(file, results_data):
    """
    Escreve estatísticas resumidas no arquivo de relatório.
    
    Args:
        file: Objeto arquivo aberto para escrita
        results_data: Lista de dicionários com resultados
    """
    # Filtrar resultados por estratégia
    model_results = [r for r in results_data if r['label'] == 'Modelo de Predição']
    simple_results = [r for r in results_data if r['label'] == 'Simulação Simples']
    buyhold_results = [r for r in results_data if r['label'] == 'Buy and Hold']
    
    # Calcular "e" escrever estatísticas
    if model_results:
        avg_model_return = sum(r['retorno_total'] for r in model_results) / len(model_results)
        file.write(f"Retorno médio do modelo XGBoost: {avg_model_return:+.2f}%\n")
    
    if simple_results:
        avg_simple_return = sum(r['retorno_total'] for r in simple_results) / len(simple_results)
        file.write(f"Retorno médio do modelo simples: {avg_simple_return:+.2f}%\n")
    
    if buyhold_results:
        avg_buyhold_return = sum(r['retorno_total'] for r in buyhold_results) / len(buyhold_results)
        file.write(f"Retorno médio Buy and Hold: {avg_buyhold_return:+.2f}%\n")
    
    file.write(f"Total de tickers analisados: {len(set(r['ticker'] for r in results_data))}\n")
    file.write(f"Total de simulações realizadas: {len(results_data)}\n")

def generate_json_report(results_data, backtest_configuration, output_path):
    """
    Gera relatório em formato JSON estruturado com resultados do backtesting.
    
    Args:
        results_data: Lista de dicionários com resultados de cada estratégia
        backtest_configuration: Configuração do backtest
        output_path: Caminho base para salvar o relatório
    """
    json_file_path = output_path.replace('.csv', '.json')
    
    # Estrutura base do relatório
    report_data = {
        'metadata': {
            'simulation_date': datetime.now().isoformat(),
            'initial_capital': backtest_configuration['initial_capital'],
            'transaction_cost_pct': backtest_configuration['transaction_cost_pct'],
            'total_tickers': len(set(r['ticker'] for r in results_data)),
            'total_simulations': len(results_data)
        },
        'results_by_ticker': {},
        'summary': {
            'model_performance': {},
            'buyhold_performance': {},
            'simple_model_performance': {}
        }
    }
    
    # Organizar resultados por ticker
    ticker_results = _group_results_by_ticker(results_data)
    
    # Adicionar resultados organizados por ticker
    for ticker, results in ticker_results.items():
        report_data['results_by_ticker'][ticker] = {
            'model_prediction': next((r for r in results if r['label'] == 'Modelo de Predição'), None),
            'model_simple': next((r for r in results if r['label'] == 'Simulação Simples'), None),
            'buy_and_hold': next((r for r in results if r['label'] == 'Buy and Hold'), None)
        }
    
    # Calcular estatísticas de performance
    _calculate_performance_statistics(report_data, results_data)
    
    # Salvar relatório JSON
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(report_data, file, indent=2, ensure_ascii=False, default=str)

def _calculate_performance_statistics(report_data, results_data):
    """
    Calcula estatísticas de performance para cada estratégia.
    
    Args:
        report_data: Dicionário com estrutura do relatório (modificado in-place)
        results_data: Lista de dicionários com resultados
    """
    # Filtrar resultados por estratégia
    model_results = [r for r in results_data if r['label'] == 'Modelo de Predição']
    simple_results = [r for r in results_data if r['label'] == 'Simulação Simples']
    buyhold_results = [r for r in results_data if r['label'] == 'Buy and Hold']
    
    # Estatísticas do modelo XGBoost
    if model_results:
        report_data['summary']['model_performance'] = {
            'average_return': sum(r['retorno_total'] for r in model_results) / len(model_results),
            'best_performer': max(model_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(model_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in model_results)
        }
    
    # Estatísticas do modelo simples
    if simple_results:
        report_data['summary']['simple_model_performance'] = {
            'average_return': sum(r['retorno_total'] for r in simple_results) / len(simple_results),
            'best_performer': max(simple_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(simple_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in simple_results)
        }
    
    # Estatísticas do Buy and Hold
    if buyhold_results:
        report_data['summary']['buyhold_performance'] = {
            'average_return': sum(r['retorno_total'] for r in buyhold_results) / len(buyhold_results),
            'best_performer': max(buyhold_results, key=lambda x: x['retorno_total']),
            'worst_performer': min(buyhold_results, key=lambda x: x['retorno_total']),
            'total_return': sum(r['retorno_total'] for r in buyhold_results)
        }

def _load_configuration():
    """
    Carrega configuração do arquivo YAML.
    
    Returns:
        tuple: (config, backtest_config, features_path, model_path)
    """
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    features_path = Path(__file__).resolve().parents[2] / config['data']['features_data_path']
    model_path = Path(__file__).resolve().parents[2] / config['model_training']['model_output_path']
    backtest_config = config["backtesting"]
    
    return config, backtest_config, features_path, model_path

def _process_single_ticker(ticker_symbol, features_path, model_path, config, backtest_configuration):
    """
    Processa um único ticker através de todo o pipeline de backtesting.
    
    Args:
        ticker_symbol: Símbolo do ticker (ex: 'PETR4.SA')
        features_path: Caminho para dados de features
        model_path: Caminho para modelos salvos
        config: Configuração completa
        backtest_configuration: Configuração específica do backtest
    
    Returns:
        tuple: (results_list, simulation_dataframe) ou (None, None) se erro
    """
    print(f"PROCESSANDO: {ticker_symbol}")
    
    # Verificar existência do modelo
    model_file_path = model_path / f"{ticker_symbol}.json"
    if not model_file_path.exists():
        print(f"ERRO: Modelo não encontrado: {model_file_path}")
        return None, None
    
    # Carregar dados de features
    features_file_path = features_path / f"{ticker_symbol}.csv"
    dataframe = pd.read_csv(features_file_path, index_col='Date', parse_dates=True)
    print(f" Dados: {dataframe.index[0].date()} até {dataframe.index[-1].date()} ({len(dataframe)} registros)")

    # Carregar modelo
    model = xgb.XGBClassifier()
    model.load_model(model_file_path)
    print(f"  Modelo carregado")

    # Preparar dados - usar a mesma função de target do treinamento (triclasse)
    from ..models.train_models import create_dynamic_triple_barrier_target
    
    # Usar os mesmos parâmetros do triple barrier method do config
    triple_barrier_config = config['model_training'].get("triple_barrier", {})
    holding_days = triple_barrier_config.get("holding_days", 5)
    profit_multiplier = triple_barrier_config.get("profit_multiplier", 2.0)
    loss_multiplier = triple_barrier_config.get("loss_multiplier", 1.5)
    
    dataframe = create_dynamic_triple_barrier_target(
        dataframe, 
        config['model_training']['target_column'],
        holding_days=holding_days,
        profit_multiplier=profit_multiplier,
        loss_multiplier=loss_multiplier,
    )
    
    # Split dos dados
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(
        dataframe,
        config['model_training']['train_final_date'],
        config['model_training']['validation_start_date'],
        config['model_training']['validation_end_date'],
        config['model_training']['test_start_date'],
        config['model_training']['test_end_date'],
        config['model_training']['target_column']
    )
    
    # Otimizar thresholds
    threshold_config = backtest_configuration.get('threshold_optimization', {})
    hold_min_days = threshold_config.get('hold_min_days', 3)
    
    val_start = config['model_training']['validation_start_date']
    val_end = config['model_training']['validation_end_date']
    validation_dataframe = dataframe[val_start:val_end]

    print(f"  Otimizando thresholds financeiros na validação...")
    buy_threshold, sell_threshold, best_sharpe = adaptive_threshold_optimization(
        model, x_val, validation_dataframe, backtest_configuration['initial_capital'],
        backtest_configuration['transaction_cost_pct'], minimum_hold_days=hold_min_days
    )
    print(f"  Thresholds: buy={buy_threshold:.2f} sell={sell_threshold:.2f} | Sharpe validação={best_sharpe:.2f}")

    # Preparar dados de simulação
    simulation_dataframe = dataframe[backtest_configuration['initial_simulation_date']:backtest_configuration['final_simulation_date']]
    print(f"  Simulação: {len(simulation_dataframe)} registros")
    
    if len(simulation_dataframe) == 0:
        print(f"  ERRO: Nenhum dado para simulação")
        return None, None
    
    if len(simulation_dataframe) < 50:
        print(f"  AVISO: Poucos dados ({len(simulation_dataframe)})")
    
    # Gerar predições multiclasse e converter para ações via score
    x_simulation = simulation_dataframe.drop(columns=[config['model_training']['target_column']])
    probabilities = model.predict_proba(x_simulation)
    score_sim = compute_up_down_score_from_proba(probabilities)
    predictions = generate_actions_from_score(
        score_sim, buy_threshold=buy_threshold, sell_threshold=sell_threshold, minimum_hold_days=hold_min_days
    )

    # Executar backtesting
    logger, log_filepath = setup_structured_logging(ticker_symbol, backtest_configuration)
    print(f"  Log salvo em: {log_filepath.name}")
    
    logger.info(f"THRESHOLDS FINANCEIROS: buy={buy_threshold:.2f} sell={sell_threshold:.2f} hold_min_days={hold_min_days}")
    logger.info(f"SCORE MÉDIO: {np.mean(score_sim):.3f}")
    
    # Executar diferentes estratégias
    realistic_portfolio = run_realistic_backtest(
        simulation_dataframe, predictions, score_sim, backtest_configuration['initial_capital'],
        backtest_configuration['transaction_cost_pct'], logger, backtest_configuration
    )

    # simple_portfolio_with_profit_techniques = run_profit_techniques_backtest(
    #     simulation_dataframe, predictions, score_sim, backtest_configuration['initial_capital'], backtest_configuration
    # )

    simple_portfolio = run_profit_techniques_backtest(
        simulation_dataframe, predictions, score_sim, backtest_configuration['initial_capital'], logger, backtest_configuration
    )

    buy_and_hold_portfolio = run_buy_and_hold_strategy(
        simulation_dataframe, backtest_configuration['initial_capital']
    )
    
    # Calcular métricas
    model_metrics = calculate_portfolio_metrics(realistic_portfolio, "Modelo de Predição", ticker_symbol)
    simple_metrics = calculate_portfolio_metrics(simple_portfolio, "Simulação Simples", ticker_symbol)
    buy_and_hold_metrics = calculate_portfolio_metrics(buy_and_hold_portfolio, "Buy and Hold", ticker_symbol)
    ml_metrics = calculate_ml_metrics(y_test, predictions)
    ml_metrics_entry = {
        'ticker': ticker_symbol,
        'label': 'Métricas de ML',
        'ml_metrics': ml_metrics
    }
    
    # Preparar dados para salvamento
    results_dataframe = pd.DataFrame({
        'Model_strategy_realistic': realistic_portfolio,
        'Model_strategy_simple': simple_portfolio,
        'Buy_and_Hold': buy_and_hold_portfolio
    })

    return [model_metrics, simple_metrics, buy_and_hold_metrics, ml_metrics_entry], results_dataframe

def _save_results_and_generate_reports(all_results, backtest_configuration):
    """
    Salva resultados e gera relatórios em múltiplos formatos.
    
    Args:
        all_results: Lista de todos os resultados de backtesting
        backtest_configuration: Configuração do backtest
    """
    now_local = datetime.now().astimezone()
    timestamp = now_local.strftime("%Y-%m-%d_%H-%M-%S_%z")
    human_timestamp = now_local.strftime("%Y-%m-%d %H:%M:%S %Z%z")

    
    # Criar DataFrame com todos os resultados
    results_dataframe = pd.DataFrame(all_results)
    final_results_path = f"{backtest_configuration['results_path']}/results_simulated_{timestamp}.csv"
    results_dataframe.to_csv(final_results_path)
    
    # Criar relatórios em TXT e JSON
    print("\nCriando relatórios em TXT e JSON...")
    generate_text_report(all_results, backtest_configuration, final_results_path)
    generate_json_report(all_results, backtest_configuration, final_results_path)
    
    print(f"Relatórios criados com sucesso em: {backtest_configuration['results_path']}")
    print("- results_simulated.csv (formato CSV)")
    print("- results_simulated.txt (formato TXT legível)")
    print("- results_simulated.json (formato JSON estruturado)")

def main():
    """
    Função principal que executa o pipeline completo de backtesting.
    
    Esta função coordena todo o processo de backtesting, incluindo:
    - Carregamento de configurações e dados
    - Processamento de cada ticker
    - Otimização de thresholds
    - Execução de diferentes estratégias
    - Geração de relatórios
    """
    

    # Carregar configurações
    config, backtest_configuration, features_path, model_path = _load_configuration()
    
    all_results = []
    
    # Processar cada ticker
    for feature_data_file in os.listdir(features_path):
        ticker_symbol = feature_data_file.replace(".csv", "")
        
        results, simulation_dataframe = _process_single_ticker(
            ticker_symbol, features_path, model_path, config, backtest_configuration
        )
        
        if results is not None:
            all_results.extend(results)
            
            # Salvar resultados individuais
            result_file_name = f"{ticker_symbol}_results.csv"
            results_path = f"{backtest_configuration['results_path']}/{result_file_name}"
            os.makedirs(backtest_configuration['results_path'], exist_ok=True)
            simulation_dataframe.to_csv(results_path)
    
    # Salvar resultados e gerar relatórios
    _save_results_and_generate_reports(all_results, backtest_configuration)
    


if __name__ == "__main__":
    main()