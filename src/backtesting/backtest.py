
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

from ..models.train_models import split_data, find_optimal_target_params, create_FIXED_triple_barrier_target

# Constantes de configuração
DEFAULT_SLIPPAGE_BPS = 0.2
DEFAULT_LOT_SIZE = 100
DEFAULT_CASH_DAILY_RATE = 0.0
DEFAULT_HOLD_MIN_DAYS = 1

# Grades mais sensíveis para melhor recall e precisão
DEFAULT_BUY_GRID = np.arange(0.00, 0.85, 0.05)  # Mais sensível: 0.20-0.55
DEFAULT_SELL_GRID = np.arange(-0.30, 0.30, 0.05)  # Mais sensível: -0.30 a 0.05
EXPANDED_BUY_GRID = np.arange(0.10, 0.70, 0.02)  # Grade expandida mais sensível
EXPANDED_SELL_GRID = np.arange(-0.40, 0.20, 0.02)  # Grade expandida mais sensível

# Grades mais sensíveis para melhor recall e precisão - OTIMIZADAS PARA TICKERS PROBLEMÁTICOS
# DEFAULT_BUY_GRID = np.arange(0.15, 0.50, 0.05)  # Mais sensível: 0.15-0.45 (era 0.20-0.55)
# DEFAULT_SELL_GRID = np.arange(-0.35, 0.05, 0.05)  # Mais sensível: -0.35 a 0.00 (era -0.30 a 0.05)
# EXPANDED_BUY_GRID = np.arange(0.05, 0.60, 0.02)  # Grade expandida mais sensível (era 0.10-0.70)
# EXPANDED_SELL_GRID = np.arange(-0.45, 0.15, 0.02)  # Grade expandida mais sensível (era -0.40 a 0.20)

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

def analyze_class_distribution_and_adjust_thresholds(probabilities, current_buy_threshold, current_sell_threshold):
    """
    Analisa a distribuição de classes e sugere ajustes nos thresholds para melhorar precisão.
    
    Args:
        probabilities: Array de probabilidades multiclasse
        current_buy_threshold: Threshold atual de compra
        current_sell_threshold: Threshold atual de venda
    
    Returns:
        tuple: (suggested_buy_threshold, suggested_sell_threshold, analysis_info)
    """
    # Calcular score s = P(up) - P(down)
    score = compute_up_down_score_from_proba(probabilities)
    
    # Análise da distribuição de scores
    score_stats = {
        'mean': np.mean(score),
        'std': np.std(score),
        'percentile_25': np.percentile(score, 25),
        'percentile_75': np.percentile(score, 75),
        'percentile_90': np.percentile(score, 90),
        'percentile_95': np.percentile(score, 95)
    }
    
    # Sugerir thresholds mais sensíveis baseados na distribuição
    # Usar percentis mais baixos para compra e mais altos para venda para melhor recall
    suggested_buy = max(0.20, min(0.60, score_stats['percentile_60']))
    suggested_sell = min(0.10, max(-0.20, score_stats['percentile_40']))
    
    # Garantir que buy > sell
    if suggested_buy <= suggested_sell:
        suggested_buy = suggested_sell + 0.20
        suggested_sell = max(0.05, suggested_buy - 0.40)
    
    analysis_info = {
        'current_buy': current_buy_threshold,
        'current_sell': current_sell_threshold,
        'suggested_buy': suggested_buy,
        'suggested_sell': suggested_sell,
        'score_stats': score_stats,
        'adjustment_reason': 'Ajuste baseado na distribuição de scores para melhorar precisão'
    }
    
    return suggested_buy, suggested_sell, analysis_info


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

def generate_actions_from_prob(prob_series, buy_threshold=0.6, sell_threshold=0.4, minimum_hold_days=3, gate_threshold=None):
    """
    Converte probabilidade p_up em ações 0/1 com histerese e gating opcional.
    """
    trading_actions = []
    current_position = 0
    days_since_last_change = minimum_hold_days

    for p_up in prob_series:
        desired_position = current_position
        can_buy = (p_up >= buy_threshold) and (gate_threshold is None or p_up >= gate_threshold)
        can_sell = (p_up <= sell_threshold)

        if can_buy:
            desired_position = 1
        elif can_sell:
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


def calculate_max_drawdown(portfolio_value_series):
    """
    Calcula o Maximum Drawdown (MDD) de uma série de valores de portfólio.
    
    O MDD é a maior perda observada desde um pico até um vale subsequente.
    Valores mais próximos de 0 indicam menor risco de perda máxima.
    
    Args:
        portfolio_value_series: Série temporal com valores do portfólio
    
    Returns:
        float: Maximum Drawdown como percentual (0 a 1)
    """
    # Calcular rolling maximum (pico histórico)
    rolling_max = portfolio_value_series.expanding().max()
    
    # Calcular drawdown em cada ponto
    drawdown = (portfolio_value_series - rolling_max) / rolling_max
    
    # Maximum drawdown é o menor valor (mais negativo)
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)  # Retornar valor absoluto


def calculate_win_rate(portfolio_value_series, trading_actions=None):
    """
    Calcula a taxa de vitórias baseada em trades ou retornos positivos.
    
    Args:
        portfolio_value_series: Série temporal com valores do portfólio
        trading_actions: Array de ações de trading (opcional)
    
    Returns:
        float: Taxa de vitórias (0 a 1)
    """
    if trading_actions is not None:
        # Calcular win rate baseado em trades
        daily_returns = portfolio_value_series.pct_change().dropna()
        
        # Identificar períodos de posição (quando trading_actions == 1)
        position_periods = trading_actions == 1
        
        if len(position_periods) != len(daily_returns):
            # Ajustar tamanhos se necessário
            min_len = min(len(position_periods), len(daily_returns))
            position_periods = position_periods[:min_len]
            daily_returns = daily_returns[:min_len]
        
        # Retornos apenas durante posições
        position_returns = daily_returns[position_periods]
        
        if len(position_returns) == 0:
            return 0.0
        
        # Win rate = proporção de retornos positivos
        win_rate = (position_returns > 0).mean()
        
    else:
        # Calcular win rate baseado em retornos diários
        daily_returns = portfolio_value_series.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return 0.0
        
        win_rate = (daily_returns > 0).mean()
    
    return win_rate


def calculate_advanced_metrics(portfolio_value_series, trading_actions=None):
    """
    Calcula um conjunto completo de métricas avançadas de performance.
    
    Args:
        portfolio_value_series: Série temporal com valores do portfólio
        trading_actions: Array de ações de trading (opcional)
    
    Returns:
        dict: Dicionário com todas as métricas calculadas
    """
    # Métricas básicas
    initial_value = portfolio_value_series.iloc[0]
    final_value = portfolio_value_series.iloc[-1]
    total_return = (final_value / initial_value - 1) * 100
    
    # Retornos diários
    daily_returns = portfolio_value_series.pct_change().dropna()
    
    # Métricas avançadas
    sharpe_ratio = calculate_sharpe_ratio(portfolio_value_series)
    max_drawdown = calculate_max_drawdown(portfolio_value_series)
    win_rate = calculate_win_rate(portfolio_value_series, trading_actions)
    
    # Métricas adicionais
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
    annual_return = ((final_value / initial_value) ** (252 / len(portfolio_value_series)) - 1) * 100 if len(portfolio_value_series) > 0 else 0
    
    # Calmar Ratio (Annual Return / Max Drawdown)
    calmar_ratio = annual_return / (max_drawdown * 100) if max_drawdown > 0 else 0
    
    # Sortino Ratio (similar ao Sharpe, mas considera apenas downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    return {
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown_pct': max_drawdown * 100,
        'volatility_pct': volatility * 100,
        'win_rate_pct': win_rate * 100,
        'initial_value': initial_value,
        'final_value': final_value
    }


def adaptive_threshold_optimization(model, x_validation, validation_dataframe, initial_capital, transaction_cost_percentage, minimum_hold_days=3, gate_threshold=None):

    # Grades mais sensíveis para melhor recall e precisão
    coarse_buy_grid = np.arange(0.10, 0.80, 0.10)  # 0.20, 0.30, 0.40, 0.50
    coarse_sell_grid = np.arange(-0.50, 0.20, 0.10)  # -0.30, -0.20, -0.10, 0.00

    best_buy_coarse, best_sell_coarse, _ = optimize_trading_thresholds_financial(
        model, x_validation, validation_dataframe, initial_capital, transaction_cost_percentage, buy_threshold_grid=coarse_buy_grid, sell_threshold_grid=coarse_sell_grid, minimum_hold_days=minimum_hold_days
    )

    # Grade de refinamento conservadora para probabilidade calibrada
    fine_buy_grid = np.arange(0.55, 0.85, 0.02)
    fine_sell_grid = np.arange(0.35, 0.55, 0.02)

    return optimize_trading_thresholds_financial(
        model, x_validation, validation_dataframe, initial_capital, transaction_cost_percentage, buy_threshold_grid=fine_buy_grid, sell_threshold_grid=fine_sell_grid, minimum_hold_days=minimum_hold_days, gate_threshold=gate_threshold
    )


def adaptive_threshold_optimization_with_probs(probabilities, validation_dataframe, initial_capital, transaction_cost_percentage, minimum_hold_days=3, gate_threshold=None):
    """
    Otimiza thresholds usando probabilidades pré-calculadas (calibradas ou não).
    
    Args:
        probabilities: Array de probabilidades multiclasse já calculadas
        validation_dataframe: DataFrame OHLCV para simulação de validação
        initial_capital: Capital inicial para simulação
        transaction_cost_percentage: Custo de transação como percentual
        minimum_hold_days: Mínimo de dias entre mudanças de posição
        gate_threshold: Threshold de confiança opcional
    
    Returns:
        tuple: (best_buy_threshold, best_sell_threshold, best_sharpe_ratio)
    """
    
    # Grades mais sensíveis para melhor recall e precisão
    coarse_buy_grid = np.arange(0.10, 0.80, 0.10)
    coarse_sell_grid = np.arange(-0.50, 0.20, 0.10)

    best_buy_coarse, best_sell_coarse, _ = optimize_trading_thresholds_financial_with_probs(
        probabilities, validation_dataframe, initial_capital, transaction_cost_percentage, 
        buy_threshold_grid=coarse_buy_grid, sell_threshold_grid=coarse_sell_grid, 
        minimum_hold_days=minimum_hold_days, gate_threshold=gate_threshold
    )

    # Grade de refinamento conservadora para probabilidade calibrada
    fine_buy_grid = np.arange(0.55, 0.85, 0.02)
    fine_sell_grid = np.arange(0.35, 0.55, 0.02)

    return optimize_trading_thresholds_financial_with_probs(
        probabilities, validation_dataframe, initial_capital, transaction_cost_percentage, 
        buy_threshold_grid=fine_buy_grid, sell_threshold_grid=fine_sell_grid, 
        minimum_hold_days=minimum_hold_days, gate_threshold=gate_threshold
    )


def optimize_trading_thresholds_financial(
    model, 
    x_validation,
    validation_dataframe, 
    initial_capital, 
    transaction_cost_percentage, 
    buy_threshold_grid=None, 
    sell_threshold_grid=None, 
    minimum_hold_days=3,
    y_validation=None,
    gate_threshold=None
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

    # Probabilidades multiclasse
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(x_validation)
    else:
        dtest = xgb.DMatrix(x_validation)
        probabilities = model.predict(dtest)
    # Probabilidade de alta
    p_up_vec = probabilities[:, 2] if probabilities.ndim == 2 and probabilities.shape[1] >= 3 else (
        probabilities[:, -1] if probabilities.ndim == 2 else probabilities
    )

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
            trading_actions = generate_actions_from_prob(
                p_up_vec,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                minimum_hold_days=minimum_hold_days,
                gate_threshold=gate_threshold
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


def optimize_trading_thresholds_financial_with_probs(
    probabilities, 
    validation_dataframe, 
    initial_capital, 
    transaction_cost_percentage, 
    buy_threshold_grid=None, 
    sell_threshold_grid=None, 
    minimum_hold_days=3,
    gate_threshold=None
):
    """
    Otimiza thresholds de compra e venda usando probabilidades pré-calculadas.
    
    Args:
        probabilities: Array de probabilidades multiclasse já calculadas
        validation_dataframe: DataFrame OHLCV para simulação de validação
        initial_capital: Capital inicial para simulação
        transaction_cost_percentage: Custo de transação como percentual
        buy_threshold_grid: Grade de thresholds de compra para testar
        sell_threshold_grid: Grade de thresholds de venda para testar
        minimum_hold_days: Mínimo de dias entre mudanças de posição
        gate_threshold: Threshold de confiança opcional
    
    Returns:
        tuple: (best_buy_threshold, best_sell_threshold, best_sharpe_ratio)
    """
    
    # Definir grades padrão se não fornecidas
    if buy_threshold_grid is None:
        buy_threshold_grid = DEFAULT_BUY_GRID
    if sell_threshold_grid is None:
        sell_threshold_grid = DEFAULT_SELL_GRID

    # Probabilidade de alta
    p_up_vec = probabilities[:, 2] if probabilities.ndim == 2 and probabilities.shape[1] >= 3 else (
        probabilities[:, -1] if probabilities.ndim == 2 else probabilities
    )

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
            trading_actions = generate_actions_from_prob(
                p_up_vec,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                minimum_hold_days=minimum_hold_days,
                gate_threshold=gate_threshold
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


def _load_cdi_data(backtest_configuration):
    """
    Carrega dados de CDI real se habilitado na configuração.
    
    Args:
        backtest_configuration: Configuração do backtest
    
    Returns:
        pd.Series ou None: Dados de CDI ou None se não disponível
    """
    if not backtest_configuration.get('use_real_cdi', False):
        return None
        
    cdi_file_path = Path(__file__).resolve().parents[2] / "src" / "renda_fixa_simulação" / "cdi_daily.csv"
    
    if cdi_file_path.exists():
        cdi_dataframe = pd.read_csv(cdi_file_path, index_col='Date', parse_dates=True)
        cdi_series = cdi_dataframe['cdi'] / 100.0  # Converter de % para decimal
        return cdi_series
    else:
        return None


def _apply_daily_cash_return(cash_amount, current_date, cdi_data, fixed_daily_rate):
    """
    Aplica rendimento diário ao caixa disponível.
    
    Args:
        cash_amount: Quantia em caixa
        current_date: Data atual
        cdi_data: Dados de CDI real (se disponível)
        fixed_daily_rate: Taxa fixa diária (se CDI real não disponível)
    
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
                      slippage_config, current_date):
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
        
        return new_cash, new_stocks_held, True, total_transaction_cost
    else:
        if stocks_to_buy == 0:
            print("Não foi possível comprar ações")
        else:
            print("kdsakdaskdsakdksa")

        return cash, stocks_held, False, 0


def _execute_sell_order(cash, stocks_held, execution_price, transaction_cost_pct, 
                       volume, volatility, slippage_config, current_date):
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
    
    return new_cash, new_stocks_held, True, net_sale_value


def run_realistic_backtest(test_dataframe, predictions, probabilities, initial_capital, transaction_cost_percentage, backtest_configuration=None):
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
        transaction_cost_percentage: Custo de transação como percentual (0 a 1)
        backtest_configuration: Configuração adicional do backtest
    
    Returns:
        pd.Series: Histórico do valor do portfólio indexado por data
    """

    # Carregar configurações adicionais
    lot_size = DEFAULT_LOT_SIZE
    cash_daily_rate = DEFAULT_CASH_DAILY_RATE
    sizing_config = {"enabled": True}
    slippage_config = SLIPPAGE_CONFIG.copy()
    cdi_data = None
    test_dataframe = test_dataframe.copy()
    
    if isinstance(backtest_configuration, dict):
        lot_size = int(backtest_configuration.get('lot_size', DEFAULT_LOT_SIZE))
        cash_daily_rate = float(backtest_configuration.get('cash_daily_rate', DEFAULT_CASH_DAILY_RATE))
        sizing_config = backtest_configuration.get('position_sizing', {"enabled": False}) or {"enabled": False}
        cdi_data = _load_cdi_data(backtest_configuration)
        
        # Configuração de slippage personalizada se fornecida
        if 'slippage_config' in backtest_configuration:
            slippage_config.update(backtest_configuration['slippage_config'])
    
    
    # Calcular volatilidade histórica (20 dias)
    test_dataframe['volatility'] = test_dataframe['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    
    # Verificar e corrigir alinhamento entre dados, predições e probabilidades
    min_size = min(len(test_dataframe), len(predictions), len(probabilities))
    if len(test_dataframe) != min_size or len(predictions) != min_size or len(probabilities) != min_size:
        test_dataframe = test_dataframe.iloc[:min_size]
        predictions = predictions[:min_size]
        probabilities = probabilities[:min_size]
    
    # Inicializar variáveis do portfólio
    available_cash = initial_capital
    stocks_quantity = 0.0
    portfolio_value_history = []
    total_trades = 0
    buy_trades = 0
    sell_trades = 0
    
    
    # Simular execução dia a dia
    for day_index in range(len(test_dataframe) - 1):
        current_date = test_dataframe.index[day_index]
        current_prediction = predictions[day_index]
        
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
                slippage_config, current_date
            )
            if trade_executed:
                total_trades += 1
                buy_trades += 1
                
        elif current_prediction == 0 and stocks_quantity > 0:  # SINAL DE VENDA
            current_volume = test_dataframe['Volume'].iloc[day_index] if 'Volume' in test_dataframe.columns else 1000000
            current_volatility = test_dataframe['volatility'].iloc[day_index] if 'volatility' in test_dataframe.columns else 0.02
            
            available_cash, stocks_quantity, trade_executed, trade_value = _execute_sell_order(
                available_cash, stocks_quantity, next_day_open_price, transaction_cost_percentage,
                current_volume, current_volatility, slippage_config, current_date
            )
            if trade_executed:
                total_trades += 1
                sell_trades += 1
        
        # Aplicar rendimento diário ao caixa
        # available_cash = _apply_daily_cash_return(
        #     available_cash, current_date, cdi_data, cash_daily_rate
        # )
        
        # Marcar valor do portfólio no fechamento do dia
        current_portfolio_value = available_cash + stocks_quantity * test_dataframe['Close'].iloc[day_index]
        portfolio_value_history.append(current_portfolio_value)
    
    # Calcular valor final do portfólio
    final_portfolio_value = available_cash + stocks_quantity * test_dataframe['Close'].iloc[-1]
    
    if total_trades == 0:
        print("AVISO: Nenhum trade foi executado!")
        print("  - Verificar se há predições de alta (1) nas predições")
        print("  - Verificar se há predições de baixa (0) quando há ações")
        print("  - Verificar se os dados têm variação suficiente")

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


def run_profit_techniques_backtest(test_dataframe, predictions, probabilities, initial_capital, backtest_configuration=None):

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
            
        
        # Aplicar rendimento diário ao caixa
        # avaliable_cash = _apply_daily_cash_return(
        #     avaliable_cash, current_date, cdi_data, cash_daily_rate
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


def calculate_portfolio_metrics(portfolio_value_history, strategy_label, ticker_symbol, trading_actions=None):
    """
    Calcula métricas básicas e avançadas de performance do portfólio.
    
    Args:
        portfolio_value_history: Série temporal com valores do portfólio
        strategy_label: Nome da estratégia (ex: 'Modelo de Predição')
        ticker_symbol: Símbolo do ticker (ex: 'PETR4.SA')
        trading_actions: Array de ações de trading (opcional)
    
    Returns:
        dict: Dicionário com métricas calculadas
    """
    # Calcular métricas avançadas
    advanced_metrics = calculate_advanced_metrics(portfolio_value_history, trading_actions)
    
    # Métricas básicas (mantidas para compatibilidade)
    metrics = {
        'ticker': ticker_symbol,
        'label': strategy_label,
        'capital_inicial': advanced_metrics['initial_value'],
        'capital_final': advanced_metrics['final_value'],
        'retorno_total': advanced_metrics['total_return_pct']
    }
    
    # Adicionar métricas avançadas
    metrics.update({
        'retorno_anual_pct': advanced_metrics['annual_return_pct'],
        'sharpe_ratio': advanced_metrics['sharpe_ratio'],
        'sortino_ratio': advanced_metrics['sortino_ratio'],
        'calmar_ratio': advanced_metrics['calmar_ratio'],
        'max_drawdown_pct': advanced_metrics['max_drawdown_pct'],
        'volatilidade_pct': advanced_metrics['volatility_pct'],
        'win_rate_pct': advanced_metrics['win_rate_pct']
    })
    
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
                    
                    # Adicionar métricas avançadas se disponíveis
                    if 'sharpe_ratio' in result:
                        file.write(f"  Retorno Anual: {result.get('retorno_anual_pct', 0):+.2f}%\n")
                        file.write(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}\n")
                        file.write(f"  Sortino Ratio: {result.get('sortino_ratio', 0):.3f}\n")
                        file.write(f"  Calmar Ratio: {result.get('calmar_ratio', 0):.3f}\n")
                        file.write(f"  Max Drawdown: {result.get('max_drawdown_pct', 0):.2f}%\n")
                        file.write(f"  Volatilidade: {result.get('volatilidade_pct', 0):.2f}%\n")
                        file.write(f"  Win Rate: {result.get('win_rate_pct', 0):.2f}%\n")
                    
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
    for result in results_data:
        ticker = result['ticker']
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


def _process_single_ticker(ticker_symbol,features_path, model_path, config, backtest_configuration):
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
    
    #Carregar dataframe com target
    labeled_dir = Path(__file__).resolve().parents[2] / "data" / "04_labeled"
    labeled_csv = labeled_dir / f"{ticker_symbol}.csv"
    labeled_meta = labeled_dir / f"{ticker_symbol}_meta.json"
    
    if labeled_csv.exists() and labeled_meta.exists():
        dataframe = pd.read_csv(labeled_csv, index_col='Date', parse_dates=True)
        saved_meta = json.loads(labeled_meta.read_text())
    else:
        base_params = {
            'objective': 'multi:softprob',  
            'num_class': 3,                 
            'eval_metric': 'aucpr',      
            'n_estimators': 500,
            'max_depth': 5,
            'n_jobs': -1,
            'tree_method': 'hist',          
            'seed': 42,   
            'max_delta_step': 1.0                   
        }
        features_file_path = features_path / f"{ticker_symbol}.csv"
        dataframe = pd.read_csv(features_file_path, index_col='Date', parse_dates=True)
        best_target_params = find_optimal_target_params(dataframe, config, base_params)
        dataframe = create_FIXED_triple_barrier_target(
            dataframe,
            config['model_training']['target_column'],
            **best_target_params
        )
    
    # Verificar existência do modelo
    model_file_path = model_path / "01_original" / f"{ticker_symbol}.json"
    if not model_file_path.exists():
        print(f"ERRO: Modelo não encontrado: {model_file_path}")
        return None, None
    

    print(f"  Carregando modelo original...")
    booster = xgb.Booster()
    booster.load_model(model_file_path)
    model = booster

    # Carregar dados essenciais para simulação de portfólio (Open, High, Low, Close)
    # Estes dados são usados APENAS para simulação de portfólio, NÃO para treinamento/predição
    essential_dir = Path(__file__).resolve().parents[2] / "data" / "04_essential_columns"
    essential_csv = essential_dir / f"{ticker_symbol}_essential.csv"
    
    if essential_csv.exists():
        essential_dataframe = pd.read_csv(essential_csv, index_col='Date', parse_dates=True)
        print(f"  Dados essenciais carregados: {len(essential_dataframe)} registros")
    else:
        print(f"  AVISO: Arquivo de dados essenciais não encontrado: {essential_csv}")
        essential_dataframe = None
    
    # Split dos dados
    x_train, y_train, x_val, y_val, _, _ = split_data(
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
    
    # Usar dados essenciais para simulação de portfólio se disponíveis
    if essential_dataframe is not None:
        validation_dataframe = essential_dataframe[val_start:val_end]
        print(f"  Usando dados essenciais para simulação de portfólio: {len(validation_dataframe)} registros")
    else:
        validation_dataframe = dataframe[val_start:val_end]
        print(f"  Usando dados de features para simulação de portfólio: {len(validation_dataframe)} registros")

    # Calcular gate_threshold (percentil) se habilitado
    gate_threshold = None
    cg_cfg = threshold_config.get('confidence_gating') if isinstance(threshold_config, dict) else None
    if isinstance(cg_cfg, dict) and cg_cfg.get('enabled', False):
        if hasattr(model, 'predict_proba'):
            val_probs = model.predict_proba(x_val)
        else:
            dval = xgb.DMatrix(x_val)
            val_probs = model.predict(dval)
        p_up_val = val_probs[:, 2] if val_probs.ndim == 2 and val_probs.shape[1] >= 3 else (
            val_probs[:, -1] if val_probs.ndim == 2 else val_probs
        )
        perc = float(cg_cfg.get('percentile', 80))
        gate_threshold = float(np.percentile(p_up_val, perc))

    # Usar probabilidades originais do XGBoost para otimização de thresholds
    print(f"  Usando probabilidades originais do XGBoost para otimização de thresholds...")
    dval = xgb.DMatrix(x_val)
    val_probs = model.predict(dval)
    
    buy_threshold, sell_threshold, best_sharpe = adaptive_threshold_optimization_with_probs(
        val_probs, validation_dataframe, backtest_configuration['initial_capital'],
        backtest_configuration['transaction_cost_pct'], minimum_hold_days=hold_min_days, gate_threshold=gate_threshold
    )

    # Preparar dados de simulação
    if essential_dataframe is not None:
        simulation_dataframe = essential_dataframe[backtest_configuration['initial_simulation_date']:backtest_configuration['final_simulation_date']]
        print(f"  Simulação (dados essenciais): {len(simulation_dataframe)} registros")
    else:
        simulation_dataframe = dataframe[backtest_configuration['initial_simulation_date']:backtest_configuration['final_simulation_date']]
        print(f"  Simulação (dados de features): {len(simulation_dataframe)} registros")
    
    if len(simulation_dataframe) == 0:
        print(f"  ERRO: Nenhum dado para simulação")
        return None, None
    
    if len(simulation_dataframe) < 50:
        print(f"  AVISO: Poucos dados ({len(simulation_dataframe)})")
    
    # Gerar predições multiclasse e converter para ações via probabilidade
    # IMPORTANTE: Para predições, usar APENAS as features selecionadas (sem colunas essenciais)
    if config['model_training']['target_column'] in simulation_dataframe.columns:
        x_simulation = simulation_dataframe.drop(columns=[config['model_training']['target_column']])
    else:
        # Se não há coluna target, usar todas as colunas disponíveis
        x_simulation = simulation_dataframe.copy()
    
    # Garantir que temos apenas as features necessárias para o modelo
    # O modelo foi treinado com features específicas, então devemos usar apenas essas
    if essential_dataframe is not None:
        # Se estamos usando dados essenciais para simulação, precisamos das features originais
        # Carregar dados de features originais para predição
        features_file_path = features_path / f"{ticker_symbol}.csv"
        if features_file_path.exists():
            features_dataframe = pd.read_csv(features_file_path, index_col='Date', parse_dates=True)
            # Filtrar para o período de simulação
            x_simulation = features_dataframe[backtest_configuration['initial_simulation_date']:backtest_configuration['final_simulation_date']]
            
            # IMPORTANTE: Remover colunas OHLC que não foram usadas no treinamento do modelo
            # O modelo foi treinado apenas com features técnicas, não com OHLC
            essential_cols_to_exclude = ['Open', 'High', 'Low', 'Close']
            model_features = [col for col in x_simulation.columns 
                            if col not in essential_cols_to_exclude]
            x_simulation = x_simulation[model_features]
            
            print(f"  Usando dados de features originais para predição: {len(x_simulation)} registros")
            print(f"  Features do modelo: {list(x_simulation.columns)}")
        else:
            print(f"  ERRO: Arquivo de features não encontrado: {features_file_path}")
            return None, None

    # Usar modelo XGBoost original
    dtest = xgb.DMatrix(x_simulation)
    probabilities = model.predict(dtest)
    
    p_up = probabilities[:, 2]
    score_sim = compute_up_down_score_from_proba(probabilities) if probabilities.ndim == 2 else p_up
    predictions = generate_actions_from_prob(
        p_up, buy_threshold=buy_threshold, sell_threshold=sell_threshold, minimum_hold_days=hold_min_days, gate_threshold=gate_threshold
    )
    
    # Executar diferentes estratégias
    realistic_portfolio = run_realistic_backtest(
        simulation_dataframe, predictions, p_up, backtest_configuration['initial_capital'],
        backtest_configuration['transaction_cost_pct'], backtest_configuration
    )

    simple_portfolio = run_profit_techniques_backtest(
        simulation_dataframe, predictions, score_sim, backtest_configuration['initial_capital'], backtest_configuration
    )

    buy_and_hold_portfolio = run_buy_and_hold_strategy(
        simulation_dataframe, backtest_configuration['initial_capital']
    )
    
    # Calcular métricas
    model_metrics = calculate_portfolio_metrics(realistic_portfolio, "Modelo de Predição", ticker_symbol, predictions)
    simple_metrics = calculate_portfolio_metrics(simple_portfolio, "Simulação Simples", ticker_symbol, predictions)
    buy_and_hold_metrics = calculate_portfolio_metrics(buy_and_hold_portfolio, "Buy and Hold", ticker_symbol)
    # Calcular métricas de ML apenas se target estiver disponível
    if config['model_training']['target_column'] in simulation_dataframe.columns:
        ml_metrics = calculate_ml_metrics(simulation_dataframe[config['model_training']['target_column']], predictions)
    else:
        # Se não há coluna target no simulation_dataframe, tentar carregar dos dados labeled
        try:
            labeled_simulation = dataframe[backtest_configuration['initial_simulation_date']:backtest_configuration['final_simulation_date']]
            if config['model_training']['target_column'] in labeled_simulation.columns:
                # Alinhar tamanhos se necessário
                min_size = min(len(labeled_simulation), len(predictions))
                target_values = labeled_simulation[config['model_training']['target_column']].iloc[:min_size]
                predictions_aligned = predictions[:min_size]
                ml_metrics = calculate_ml_metrics(target_values, predictions_aligned)
                print(f"  Métricas de ML calculadas usando dados labeled: {len(target_values)} registros")
            else:
                ml_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
                print(f"  AVISO: Coluna 'target' não encontrada nos dados labeled")
        except Exception as e:
            ml_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
            print(f"  ERRO ao calcular métricas de ML: {e}")
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
    # final_results_path = f"{backtest_configuration['results_path']}/ITUB4SA_Teste.csv"
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
    tickers = config['data']['tickers']
    all_results = []
    
    # Processar cada ticker
    for ticker_symbol in tickers:
        
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