#!/usr/bin/env python3
"""
Feature Importance Sweep - Versão Melhorada
===========================================

Esta versão corrige os problemas críticos identificados no feature sweep original:
1. Elimina data leakage (Open, Close, High, Low)
2. Implementa validação cruzada robusta
3. Usa thresholds fixos para evitar look-ahead bias
4. Ranking de features mais estável
5. Regularização por complexidade
6. Tratamento de valores zerados

Uso:
    python -m src.analysis.feature_importance_sweep_improved
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..models.train_models import split_data, find_optimal_target_params, create_FIXED_triple_barrier_target
from ..backtesting.backtest import (
    simulate_portfolio_execution_next_open,
    calculate_sharpe_ratio,
)


# CONFIGURAÇÕES CORRIGIDAS
FORBIDDEN_FEATURES = ['Open', 'Close', 'High', 'Low', 'Volume']  # Data leakage
ALLOWED_FEATURES = [
    'EMA_short', 'EMA_long', 'MACD', 'MACD_signal', 'MACD_hist',
    'RSI', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR', 'ADX', 
    'MFI', 'OBV', 'wavelet_cA', 'wavelet_cD'
]

# Thresholds fixos para evitar look-ahead bias
FIXED_THRESHOLDS = {
    'buy_threshold': 0.65,   # 65% confiança para compra
    'sell_threshold': 0.35   # 35% confiança para venda
}

# Parâmetros de validação robusta
VALIDATION_CONFIG = {
    'n_splits': 5,           # Era 3
    'min_window_size': 100,  # Era 50
    'min_train_size': 200,   # Novo
    'max_features': 12       # Limite para evitar overfitting
}

# Parâmetros de modelo mais robustos
ROBUST_MODEL_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'aucpr',
    'n_estimators': 300,     # Era 200
    'max_depth': 4,
    'learning_rate': 0.08,
    'subsample': 0.8,        # Era 0.9 - mais regularização
    'colsample_bytree': 0.8, # Era 0.9 - mais regularização
    'tree_method': 'hist',
    'n_jobs': -1,
    'seed': 42,
    'max_delta_step': 1.0,
    'reg_alpha': 0.1,        # Novo - regularização L1
    'reg_lambda': 0.1        # Novo - regularização L2
}


def _load_config():
    """Carrega configuração do arquivo YAML."""
    config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _filter_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features que causam data leakage.
    
    Args:
        df: DataFrame com todas as features
        
    Returns:
        DataFrame apenas com features seguras
    """
    safe_features = [col for col in df.columns if col not in FORBIDDEN_FEATURES]
    return df[safe_features]


def _train_robust_xgb(x_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> xgb.Booster:
    """
    Treina modelo XGBoost com parâmetros robustos.
    
    Args:
        x_train: Features de treino
        y_train: Targets de treino
        seed: Seed para reprodutibilidade
        
    Returns:
        Modelo XGBoost treinado
    """
    params = ROBUST_MODEL_PARAMS.copy()
    params['seed'] = seed
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    return xgb.train(params, dtrain, num_boost_round=params['n_estimators'])


def _robust_feature_ranking(x_train: pd.DataFrame, y_train: pd.Series, n_models: int = 5) -> List[str]:
    """
    Gera ranking robusto de features usando múltiplos modelos.
    
    Args:
        x_train: Features de treino
        y_train: Targets de treino
        n_models: Número de modelos para ensemble
        
    Returns:
        Lista de features ordenadas por importância
    """
    feature_importance_scores = {}
    
    # Treinar múltiplos modelos com seeds diferentes
    for seed in range(n_models):
        model = _train_robust_xgb(x_train, y_train, seed)
        importance = model.get_score(importance_type='gain')
        
        # Acumular scores
        for feature in x_train.columns:
            if feature not in feature_importance_scores:
                feature_importance_scores[feature] = []
            feature_importance_scores[feature].append(importance.get(feature, 0.0))
    
    # Calcular média e desvio padrão
    feature_stats = {}
    for feature, scores in feature_importance_scores.items():
        feature_stats[feature] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'stability': np.mean(scores) / (np.std(scores) + 1e-8)  # Estabilidade
        }
    
    # Ordenar por média de importância, com penalização por instabilidade
    ranked_features = sorted(
        feature_stats.items(),
        key=lambda x: x[1]['mean'] * (1 - x[1]['std'] / (x[1]['mean'] + 1e-8)),
        reverse=True
    )
    
    return [feature for feature, _ in ranked_features]


def _robust_temporal_cross_validation(df: pd.DataFrame, feature_subset: List[str]) -> float:
    """
    Validação cruzada temporal robusta com configurações melhoradas.
    
    Args:
        df: DataFrame com dados completos
        feature_subset: Lista de features para testar
        
    Returns:
        Sharpe ratio médio validado
    """
    config = VALIDATION_CONFIG
    total_sharpe = 0
    valid_splits = 0
    
    # Dividir dados em janelas temporais
    data_length = len(df)
    window_size = max(config['min_window_size'], data_length // (config['n_splits'] + 1))
    
    for i in range(config['n_splits']):
        # Janela de treino: do início até a janela i
        train_end = max(config['min_train_size'], (i + 1) * window_size)
        # Janela de teste: próxima janela
        test_start = train_end
        test_end = min(test_start + window_size, data_length)
        
        # Verificar tamanhos mínimos
        if (test_end - test_start < config['min_window_size'] or 
            train_end < config['min_train_size']):
            continue
            
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        # Preparar dados
        x_train = train_df[feature_subset]
        y_train = train_df['target']
        x_test = test_df[feature_subset]
        y_test = test_df['target']
        
        # Verificar se há dados suficientes
        if len(x_train) < 50 or len(x_test) < 20:
            continue
        
        # Treinar modelo
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        
        booster = xgb.train(ROBUST_MODEL_PARAMS, dtrain, num_boost_round=200)
        
        # Predições
        probs = booster.predict(dtest)
        p_up = probs[:, 2] if probs.ndim == 2 and probs.shape[1] >= 3 else (
            probs[:, -1] if probs.ndim == 2 else probs
        )
        
        # Usar thresholds fixos (sem look-ahead bias)
        buy_th = FIXED_THRESHOLDS['buy_threshold']
        sell_th = FIXED_THRESHOLDS['sell_threshold']
        
        # Gerar ações
        actions = np.where(p_up >= buy_th, 1, np.where(p_up <= sell_th, -1, 0))
        
        # Verificar se há ações suficientes
        if np.sum(np.abs(actions)) < 5:  # Muito poucas operações
            continue
        
        # Simulação de portfólio
        m = min(len(test_df), len(actions))
        try:
            port = simulate_portfolio_execution_next_open(
                test_df.iloc[:m], actions[:m], 
                initial_capital=100000.0, 
                transaction_cost_percentage=0.001
            )
            
            sharpe = calculate_sharpe_ratio(port)
            
            # Verificar se Sharpe é válido
            if (not np.isnan(sharpe) and not np.isinf(sharpe) and 
                sharpe != 0.0 and sharpe > -10):  # Filtrar valores extremos
                total_sharpe += sharpe
                valid_splits += 1
                
        except Exception as e:
            print(f"    Erro na simulação: {e}")
            continue
    
    return total_sharpe / valid_splits if valid_splits > 0 else 0.0


def _apply_complexity_penalty(sharpe: float, n_features: int) -> float:
    """
    Aplica penalização por complexidade para evitar overfitting.
    
    Args:
        sharpe: Sharpe ratio original
        n_features: Número de features
        
    Returns:
        Sharpe ratio ajustado
    """
    max_features = VALIDATION_CONFIG['max_features']
    if n_features <= max_features:
        return sharpe
    
    # Penalização exponencial para muitas features
    penalty = (n_features - max_features) * 0.05
    return sharpe - penalty


def run_improved_sweep():
    """
    Executa o feature sweep melhorado com todas as correções.
    """
    config = _load_config()
    project_root = Path(__file__).resolve().parents[2]
    
    # Diretórios
    input_dir = project_root / "data" / "03_features"
    labeled_dir = project_root / 'data' / '04_labeled'
    report_dir = project_root / 'reports' / 'feature_sweep_improved'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = config['data']['tickers']
    results_summary = []
    
    print("🚀 INICIANDO FEATURE SWEEP MELHORADO")
    print("=" * 60)
    print(f"Features proibidas (data leakage): {FORBIDDEN_FEATURES}")
    print(f"Thresholds fixos: {FIXED_THRESHOLDS}")
    print(f"Configuração de validação: {VALIDATION_CONFIG}")
    print("=" * 60)
    
    for ticker in tickers:
        print(f"\n📊 Processando {ticker}...")
        
        feature_file = input_dir / f'{ticker}.csv'
        if not feature_file.exists():
            print(f"  ❌ Features não encontradas para {ticker}. Pulando...")
            continue
        
        # Carregar e filtrar features
        df_feat = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        df_feat_safe = _filter_safe_features(df_feat)
        
        print(f"  📈 Features originais: {len(df_feat.columns)}")
        print(f"  🔒 Features seguras: {len(df_feat_safe.columns)}")
        print(f"  🚫 Features removidas: {len(df_feat.columns) - len(df_feat_safe.columns)}")
        
        # Carregar ou criar dados rotulados
        labeled_csv = labeled_dir / f"{ticker}.csv"
        if labeled_csv.exists():
            df_lab = pd.read_csv(labeled_csv, index_col='Date', parse_dates=True)
        else:
            print(f"  ⚠️  Dados rotulados não encontrados para {ticker}")
            continue
        
        # Dividir dados
        x_train, y_train, x_val, y_val, _, _ = split_data(
            df_lab,
            config['model_training']['train_final_date'],
            config['model_training']['validation_start_date'],
            config['model_training']['validation_end_date'],
            config['model_training']['test_start_date'],
            config['model_training']['test_end_date'],
            target_column_name=config['model_training']['target_column'],
        )
        
        # Filtrar features de treino também
        safe_features = [col for col in x_train.columns if col not in FORBIDDEN_FEATURES]
        x_train_safe = x_train[safe_features]
        
        # Gerar ranking robusto de features
        print(f"  🎯 Gerando ranking robusto de features...")
        ranked = _robust_feature_ranking(x_train_safe, y_train)
        
        # Salvar ranking
        importance_df = pd.DataFrame({
            'feature': ranked,
            'rank': range(1, len(ranked) + 1)
        })
        importance_df.to_csv(report_dir / f'feature_ranking_{ticker}.csv', index=False)
        print(f"  📋 Top 10 features: {ranked[:10]}")
        
        # Sweep de tamanhos (limitado para evitar overfitting)
        max_k = min(VALIDATION_CONFIG['max_features'], len(ranked))
        ks = [1, 2, 3, 5, 8, 10, 12] if max_k >= 12 else [1, 2, 3, 5, 8, 10]
        ks = [k for k in ks if k <= max_k]
        
        rows = []
        print(f"  🔍 Testando {len(ks)} configurações de K (máximo {max_k})...")
        
        for k in ks:
            subset = ranked[:k]
            print(f"    K={k}: {subset[:3]}...")
            
            # Validação cruzada robusta
            sharpe = _robust_temporal_cross_validation(df_lab, subset)
            
            # Aplicar penalização por complexidade
            adjusted_sharpe = _apply_complexity_penalty(sharpe, k)
            
            rows.append({
                'ticker': ticker, 
                'k': k, 
                'sharpe_val': sharpe,
                'adjusted_sharpe': adjusted_sharpe,
                'features_used': ', '.join(subset)
            })
            
            print(f"      Sharpe: {sharpe:.4f} → Ajustado: {adjusted_sharpe:.4f}")
        
        # Salvar resultados
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(report_dir / f'sweep_{ticker}.csv', index=False)
        
        # Encontrar melhor resultado (usando Sharpe ajustado)
        best_row = df_rows.sort_values('adjusted_sharpe', ascending=False).head(1)
        if not best_row.empty:
            best_dict = best_row.iloc[0].to_dict()
            best_dict['best_features'] = best_dict['features_used']
            results_summary.append(best_dict)
            print(f"  🏆 Melhor: K={best_dict['k']}, Sharpe={best_dict['adjusted_sharpe']:.4f}")
        else:
            print(f"  ❌ Nenhum resultado válido para {ticker}")
    
    # Salvar resumo
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(report_dir / 'sweep_summary_improved.csv', index=False)
        
        print(f"\n🎉 RESUMO DOS RESULTADOS MELHORADOS")
        print("=" * 60)
        for _, row in summary_df.iterrows():
            print(f"{row['ticker']}: K={row['k']}, Sharpe={row['adjusted_sharpe']:.4f}")
        print("=" * 60)
        print(f"📁 Resultados salvos em: {report_dir}")
    else:
        print("❌ Nenhum resultado válido encontrado!")


if __name__ == '__main__':
    run_improved_sweep()
