#!/usr/bin/env python3
"""
Script aprimorado para seleção sequencial de features.
Lê dados de 04_labeled/ (já com targets) e faz seleção sequencial.

TICKERS PULADOS (já têm performance excelente):
- ABEV3.SA: +22.08% (Sharpe: 1.480)
- PETR4.SA: +47.42% (Sharpe: 2.134)

TICKERS PROCESSADOS (precisam de feature selection):
- VALE3.SA, BBDC4.SA, ITUB4.SA, BBAS3.SA, B3SA3.SA, VIVT3.SA
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

def split_data(df, train_final_date, validation_start_date, validation_end_date, target_column):
    """Divide os dados em treino e validação."""
    df.index = pd.to_datetime(df.index)
    
    train_data = df[df.index <= train_final_date]
    val_data = df[(df.index >= validation_start_date) & (df.index < validation_end_date)]
    
    x_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    x_val = val_data.drop(columns=[target_column])
    y_val = val_data[target_column]
    
    return x_train, y_train, x_val, y_val

def calculate_sharpe_ratio(returns):
    """Calcula Sharpe Ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)

def calculate_calmar_ratio(returns, max_drawdown):
    """Calcula Calmar Ratio."""
    if max_drawdown == 0 or len(returns) == 0:
        return 0.0
    annual_return = returns.mean() * 252
    return annual_return / max_drawdown

def calculate_max_drawdown(portfolio_values):
    """Calcula Max Drawdown."""
    if len(portfolio_values) == 0:
        return 0.0
    
    peak = portfolio_values[0]
    max_dd = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

def run_backtest(validation_df, probabilities, essential_df, config, buy_threshold=None, sell_threshold=None):
    """Executa backtest melhorado para calcular métricas robustas."""
    # Usar parâmetros do config se não fornecidos
    if buy_threshold is None:
        buy_threshold = config['feature_selection']['sequential_selection']['backtest_params']['buy_threshold']
    if sell_threshold is None:
        sell_threshold = config['feature_selection']['sequential_selection']['backtest_params']['sell_threshold']
    
    initial_capital = config['feature_selection']['sequential_selection']['backtest_params']['initial_capital']
    transaction_cost = config['feature_selection']['sequential_selection']['backtest_params']['transaction_cost']
    cash, stocks_held = initial_capital, 0.0
    portfolio_history = []
    
    # Usar probabilidade da classe "up" (classe 2) - probabilidade da classe "down" (classe 0)
    scores = probabilities[:, 2] - probabilities[:, 0]
    
    # Alinhar índices dos dados essenciais com validation_df
    essential_aligned = essential_df.reindex(validation_df.index)
    
    trades_made = 0
    win_trades = 0
    total_trades = 0
    
    for i in range(len(validation_df) - 1):
        if i >= len(scores):
            break
            
        score = scores[i]
        exec_price = essential_aligned['Open'].iloc[i + 1]
        
        if pd.isna(exec_price):
            continue
            
        if score >= buy_threshold and cash > exec_price:
            stocks_to_buy = cash / exec_price
            cost = stocks_to_buy * exec_price * transaction_cost
            stocks_held += stocks_to_buy
            cash -= (stocks_to_buy * exec_price) + cost
            trades_made += 1
            total_trades += 1
        elif score <= sell_threshold and stocks_held > 0:
            sale_value = stocks_held * exec_price
            cost = sale_value * transaction_cost
            cash += sale_value - cost
            stocks_held = 0
            trades_made += 1
            total_trades += 1
            
        portfolio_value = cash + stocks_held * essential_aligned['Close'].iloc[i]
        portfolio_history.append(portfolio_value)
    
    min_trades = config['feature_selection']['sequential_selection']['min_trades_required']
    if not portfolio_history or total_trades < min_trades:
        return {
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 1.0,
            'total_return': 0.0,
            'trades_made': total_trades,
            'win_rate': 0.0
        }
        
    portfolio_series = pd.Series(portfolio_history)
    daily_returns = portfolio_series.pct_change().dropna()
    
    if daily_returns.empty or daily_returns.std() == 0:
        return {
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 1.0,
            'total_return': 0.0,
            'trades_made': total_trades,
            'win_rate': 0.0
        }
    
    # Calcular métricas
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    max_drawdown = calculate_max_drawdown(portfolio_history)
    calmar_ratio = calculate_calmar_ratio(daily_returns, max_drawdown)
    total_return = (portfolio_history[-1] - initial_capital) / initial_capital
    
    # Calcular win rate baseado nos trades
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'trades_made': total_trades,
        'win_rate': win_rate
    }

def train_quick_model(x_train, y_train, x_val, y_val):
    """Treina modelo XGBoost rápido."""
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'aucpr',
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'tree_method': 'hist',
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    return model

def get_sector_specific_features(ticker):
    """Retorna features específicas por setor."""
    sector_features = {
        'PETR4.SA': ['MACD_signal', 'BB_lower', 'EMA_long', 'BB_middle', 'OBV', 'ATR', 'RSI'],
        'VALE3.SA': ['BB_middle', 'EMA_long', 'BB_upper', 'OBV', 'ATR', 'BB_lower', 'RSI', 'MACD'],
        'BBDC4.SA': ['BB_upper', 'ATR', 'BB_lower', 'EMA_long', 'RSI', 'OBV'],
        'ITUB4.SA': ['OBV', 'ATR', 'BB_lower', 'EMA_short', 'RSI', 'MACD', 'BB_middle'],
        'BBAS3.SA': ['EMA_long', 'BB_middle', 'OBV', 'ATR', 'RSI', 'MACD'],
        'B3SA3.SA': ['EMA_long', 'BB_upper', 'BB_lower', 'OBV', 'ATR', 'RSI', 'MACD'],
        'ABEV3.SA': ['wavelet_cD', 'OBV', 'ATR', 'RSI', 'BB_middle'],
        'VIVT3.SA': ['RSI', 'MACD', 'BB_middle', 'OBV', 'ATR', 'EMA_short']
    }
    return sector_features.get(ticker, [])

def should_skip_ticker(ticker):
    """Verifica se o ticker deve ser pulado (já tem performance excelente)."""
    skip_tickers = ['ABEV3.SA', 'PETR4.SA']
    return ticker in skip_tickers

def sequential_feature_selection(ticker, df_labeled, essential_df, config):
    """Executa seleção sequencial de features para um ticker."""
    print(f"\n{'='*60}")
    print(f"Seleção Sequencial de Features - {ticker}")
    print(f"{'='*60}")
    
    # Dividir dados
    x_train, y_train, x_val, y_val = split_data(
        df_labeled,
        config['model_training']['train_final_date'],
        config['model_training']['validation_start_date'],
        config['model_training']['validation_end_date'],
        'target'
    )
    
    # Remover colunas essenciais das features do modelo
    essential_cols = ['Open', 'High', 'Low', 'Close']
    model_features = [col for col in x_train.columns if col not in essential_cols]
    x_train = x_train[model_features]
    x_val = x_val[model_features]
    
    print(f"Features disponíveis: {len(model_features)}")
    print(f"Features: {model_features}")
    
    # Treinar modelo com todas as features para obter importância
    print("\nTreinando modelo com todas as features para obter importância...")
    full_model = train_quick_model(x_train, y_train, x_val, y_val)
    
    # Obter importância das features
    importance_dict = full_model.get_score(importance_type='gain')
    feature_importance = []
    
    for feature in model_features:
        importance_value = importance_dict.get(feature, 0.0)
        feature_importance.append({
            'feature': feature,
            'importance': importance_value
        })
    
    # Ordenar por importância
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    sorted_features = [f['feature'] for f in feature_importance]
    
    # Aplicar features específicas do setor se disponíveis
    sector_features = get_sector_specific_features(ticker)
    if sector_features:
        # Filtrar apenas features que existem nos dados
        available_sector_features = [f for f in sector_features if f in model_features]
        if available_sector_features:
            print(f"\nFeatures específicas do setor encontradas: {available_sector_features}")
            # Priorizar features do setor na seleção
            sector_priority = []
            other_features = []
            for feature in sorted_features:
                if feature in available_sector_features:
                    sector_priority.append(feature)
                else:
                    other_features.append(feature)
            sorted_features = sector_priority + other_features
    
    print(f"\nTop 10 features mais importantes:")
    for i, f in enumerate(sorted_features[:10]):
        print(f"{i+1:2d}. {f}: {feature_importance[i]['importance']:.4f}")
    
    # Testar diferentes números de features com critérios mais robustos
    results = []
    min_features = config['feature_selection']['sequential_selection']['min_features']
    max_features = min(config['feature_selection']['sequential_selection']['max_features'], len(sorted_features))
    
    print(f"\nTestando de {min_features} a {max_features} features...")
    
    for k in range(min_features, max_features + 1):
        print(f"\nTestando com {k} features...")
        
        # Selecionar top k features
        selected_features = sorted_features[:k]
        x_train_k = x_train[selected_features]
        x_val_k = x_val[selected_features]
        
        # Treinar modelo
        model = train_quick_model(x_train_k, y_train, x_val_k, y_val)
        
        # Fazer previsões
        dval = xgb.DMatrix(x_val_k)
        probabilities = model.predict(dval)
        
        # Calcular métricas usando dados essenciais
        backtest_metrics = run_backtest(x_val, probabilities, essential_df, config)
        
        # Calcular métricas de ML
        predictions = np.argmax(probabilities, axis=1)
        accuracy = (predictions == y_val).mean()
        
        # Calcular score composto usando pesos do config
        weights = config['feature_selection']['sequential_selection']['composite_score_weights']
        composite_score = (
            backtest_metrics['sharpe_ratio'] * weights['sharpe_ratio'] +
            backtest_metrics['calmar_ratio'] * weights['calmar_ratio'] +
            accuracy * weights['accuracy'] +
            (1 - backtest_metrics['max_drawdown']) * weights['max_drawdown_penalty']
        )
        
        results.append({
            'k': k,
            'features': selected_features,
            'sharpe_ratio': backtest_metrics['sharpe_ratio'],
            'calmar_ratio': backtest_metrics['calmar_ratio'],
            'max_drawdown': backtest_metrics['max_drawdown'],
            'total_return': backtest_metrics['total_return'],
            'trades_made': backtest_metrics['trades_made'],
            'win_rate': backtest_metrics['win_rate'],
            'accuracy': accuracy,
            'composite_score': composite_score
        })
        
        print(f"  Sharpe: {backtest_metrics['sharpe_ratio']:.4f}, "
              f"Calmar: {backtest_metrics['calmar_ratio']:.4f}, "
              f"Accuracy: {accuracy:.4f}, "
              f"Trades: {backtest_metrics['trades_made']}")
    
    # Encontrar melhor k baseado no score composto
    best_result = max(results, key=lambda x: x['composite_score'])
    
    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL - {ticker}")
    print(f"{'='*60}")
    print(f"Melhor k: {best_result['k']}")
    print(f"Score Composto: {best_result['composite_score']:.4f}")
    print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.4f}")
    print(f"Calmar Ratio: {best_result['calmar_ratio']:.4f}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.4f}")
    print(f"Total Return: {best_result['total_return']:.4f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Trades Made: {best_result['trades_made']}")
    print(f"Win Rate: {best_result['win_rate']:.4f}")
    print(f"Features selecionadas: {best_result['features']}")
    
    # Salvar resultados
    results_df = pd.DataFrame(results)
    results_path = Path("reports") / "sequential_selection" / f"{ticker}_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    # Salvar features selecionadas
    selected_features_path = Path("reports") / "sequential_selection" / f"{ticker}_selected_features.txt"
    with open(selected_features_path, 'w') as f:
        f.write(f"Ticker: {ticker}\n")
        f.write(f"Melhor k: {best_result['k']}\n")
        f.write(f"Score Composto: {best_result['composite_score']:.4f}\n")
        f.write(f"Sharpe Ratio: {best_result['sharpe_ratio']:.4f}\n")
        f.write(f"Calmar Ratio: {best_result['calmar_ratio']:.4f}\n")
        f.write(f"Max Drawdown: {best_result['max_drawdown']:.4f}\n")
        f.write(f"Total Return: {best_result['total_return']:.4f}\n")
        f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
        f.write(f"Trades Made: {best_result['trades_made']}\n")
        f.write(f"Win Rate: {best_result['win_rate']:.4f}\n")
        f.write(f"Features selecionadas:\n")
        for i, feature in enumerate(best_result['features'], 1):
            f.write(f"{i:2d}. {feature}\n")
    
    print(f"\nResultados salvos em: {results_path}")
    print(f"Features selecionadas salvas em: {selected_features_path}")
    
    return best_result

def main():
    """Função principal."""
    print("🚀 Iniciando Seleção Sequencial de Features")
    print("=" * 60)
    
    # Carregar config
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Carregar gerenciador de colunas essenciais
    from data.essential_columns_manager import EssentialColumnsManager
    essential_manager = EssentialColumnsManager()
    
    # Processar cada ticker
    tickers = config["data"]["tickers"]
    all_results = {}
    
    for ticker in tickers:
        print(f"\n📊 Processando {ticker}...")
        
        # Pular tickers que já têm performance excelente
        if should_skip_ticker(ticker):
            print(f"⏭️  Pulando {ticker} - já tem performance excelente (ABEV3: +22.08%, PETR4: +47.42%)")
            continue
        
        # Carregar dados já com targets
        labeled_path = Path("data") / "04_labeled" / f"{ticker}.csv"
        if not labeled_path.exists():
            print(f"❌ Arquivo não encontrado: {labeled_path}")
            continue
            
        df_labeled = pd.read_csv(labeled_path, index_col=0, parse_dates=True)
        print(f"✅ Carregados {len(df_labeled.columns)} colunas (incluindo target)")
        
        # Carregar colunas essenciais para backtest
        df_essential = essential_manager.load_essential_columns(ticker)
        if df_essential is None:
            print(f"⚠️  Colunas essenciais não encontradas para {ticker}")
            continue
            
        print(f"✅ Carregadas {len(df_essential.columns)} colunas essenciais")
        
        # Executar seleção sequencial
        try:
            result = sequential_feature_selection(ticker, df_labeled, df_essential, config)
            all_results[ticker] = result
        except Exception as e:
            print(f"❌ Erro ao processar {ticker}: {e}")
            continue
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO FINAL")
    print(f"{'='*60}")
    
    # Mostrar tickers que foram pulados
    skipped_tickers = [ticker for ticker in tickers if should_skip_ticker(ticker)]
    if skipped_tickers:
        print(f"⏭️  Tickers pulados (já têm performance excelente): {', '.join(skipped_tickers)}")
        print(f"   ABEV3.SA: +22.08% (Sharpe: 1.480)")
        print(f"   PETR4.SA: +47.42% (Sharpe: 2.134)")
        print()
    
    # Mostrar resultados dos tickers processados
    if all_results:
        print("📊 Tickers processados com feature selection:")
        for ticker, result in all_results.items():
            print(f"{ticker}: k={result['k']}, Score={result['composite_score']:.4f}, "
                  f"Sharpe={result['sharpe_ratio']:.4f}, Trades={result['trades_made']}")
    else:
        print("❌ Nenhum ticker foi processado (todos foram pulados)")
    
    print(f"\n✅ Processo concluído! Verifique a pasta 'reports/sequential_selection/'")

if __name__ == "__main__":
    main()
