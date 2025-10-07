#!/usr/bin/env python3
"""
Grid Search para Seleção de Features por Ticker
===============================================

Este módulo executa um grid search completo para encontrar a melhor combinação de:
- top_k: número de features por importância
- corr_threshold: limite de correlação para poda
- calibration: método de calibração (sigmoid, isotonic, none)

Para cada combinação, treina um XGBoost rápido e avalia no conjunto de validação
usando Sharpe Ratio como métrica principal.

Uso:
    python -m src.features.feature_selection_grid_search
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import brier_score_loss

from ..models.train_models import (
    split_data,
    find_optimal_target_params,
    create_FIXED_triple_barrier_target,
    create_calibrated_model,
    evaluate_calibration_quality,
)
from ..backtesting.backtest import (
    optimize_trading_thresholds_financial_with_probs,
    simulate_portfolio_execution_next_open,
    calculate_sharpe_ratio,
)


DEFAULT_PROTECTED_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'ATR', 'Volume'
]

# Grid de parâmetros para testar
GRID_PARAMS = {
    'top_k': [20, 30, 40, 50],
    'corr_threshold': [0.90, 0.95, 0.98],
    'calibration': ['sigmoid', 'isotonic', 'none']
}


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _ensure_labeled_dataframe(ticker: str, df_features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Garante que existe dataframe com target para o ticker."""
    labeled_dir = Path(__file__).resolve().parents[2] / 'data' / '04_labeled'
    labeled_dir.mkdir(parents=True, exist_ok=True)
    labeled_csv = labeled_dir / f"{ticker}.csv"
    
    if labeled_csv.exists():
        return pd.read_csv(labeled_csv, index_col='Date', parse_dates=True)

    # Gerar target se não existir
    base_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'auc',
        'n_estimators': 200,
        'max_depth': 4,
        'n_jobs': -1,
        'tree_method': 'hist',
        'seed': 42,
        'max_delta_step': 1.0,
    }
    best_target_params = find_optimal_target_params(df_features, config, base_params, ticker)

    if best_target_params is None or 'profit_threshold' in best_target_params:
        df_labeled = create_FIXED_triple_barrier_target(
            df_features.copy(),
            config['model_training']['target_column'],
            **(best_target_params or {'holding_days': 7, 'profit_threshold': 0.03, 'loss_threshold': -0.02})
        )
    else:
        df_labeled = df_features.copy()

    df_labeled.to_csv(labeled_csv)
    return df_labeled


def _train_quick_xgb(x_train: pd.DataFrame, y_train: pd.Series) -> xgb.Booster:
    """Treina XGBoost rápido para avaliação."""
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'auc',
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.08,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'tree_method': 'hist',
        'n_jobs': -1,
        'seed': 42,
        'max_delta_step': 1.0,
    }
    dtrain = xgb.DMatrix(x_train, label=y_train)
    booster = xgb.train(params, dtrain, num_boost_round=params['n_estimators'])
    return booster


def _get_feature_importance_gain(model: xgb.Booster, feature_names: List[str]) -> pd.DataFrame:
    """Obtém importância das features por gain."""
    importance_dict = model.get_score(importance_type='gain')
    data = [{'feature': f, 'gain': float(importance_dict.get(f, 0.0))} for f in feature_names]
    df = pd.DataFrame(data).sort_values('gain', ascending=False)
    return df


def _prune_by_correlation(df: pd.DataFrame, selected_cols: List[str], corr_threshold: float) -> List[str]:
    """Remove features correlatas mantendo as mais importantes."""
    if not selected_cols:
        return selected_cols
    
    corr = df[selected_cols].corr().abs()
    keep = []
    for col in selected_cols:
        if all(corr[col][k] <= corr_threshold for k in keep):
            keep.append(col)
    return keep


def _select_features_for_params(
    df_features: pd.DataFrame,
    df_labeled: pd.DataFrame,
    top_k: int,
    corr_threshold: float,
    protected_columns: List[str],
    config: dict
) -> Tuple[List[str], pd.DataFrame]:
    """Seleciona features para uma combinação específica de parâmetros."""
    
    # Garantir colunas protegidas existentes
    protected_present = [c for c in protected_columns if c in df_features.columns]
    
    # Split temporal para treinar modelo
    x_train, y_train, x_val, y_val, _, _ = split_data(
        df_labeled,
        config['model_training']['train_final_date'],
        config['model_training']['validation_start_date'],
        config['model_training']['validation_end_date'],
        config['model_training']['test_start_date'],
        config['model_training']['test_end_date'],
        target_column_name=config['model_training']['target_column'],
    )
    
    # Usar treino+val para obter importâncias
    x_imp = pd.concat([x_train, x_val])
    y_imp = pd.concat([y_train, y_val])
    
    # Treinar modelo rápido para obter importâncias
    model = _train_quick_xgb(x_imp, y_imp)
    
    # Obter importâncias e selecionar top_k
    importance_df = _get_feature_importance_gain(model, x_imp.columns.tolist())
    ranked_cols = importance_df['feature'].tolist()
    top_cols = ranked_cols[:top_k]
    
    # Poda por correlação
    pruned_cols = _prune_by_correlation(x_imp, top_cols, corr_threshold)
    
    # Garantir colunas protegidas
    final_cols = list(dict.fromkeys(protected_present + pruned_cols))
    
    return final_cols, importance_df


def _evaluate_combination(
    ticker: str,
    df_labeled: pd.DataFrame,
    selected_features: List[str],
    calibration_method: str,
    config: dict
) -> Dict[str, Any]:
    """
    Avalia uma combinação específica de parâmetros.
    
    Retorna métricas de validação incluindo Sharpe Ratio.
    """
    
    # Split temporal
    x_train, y_train, x_val, y_val, _, _ = split_data(
        df_labeled,
        config['model_training']['train_final_date'],
        config['model_training']['validation_start_date'],
        config['model_training']['validation_end_date'],
        config['model_training']['test_start_date'],
        config['model_training']['test_end_date'],
        target_column_name=config['model_training']['target_column'],
    )
    
    # Filtrar apenas features selecionadas
    x_train_selected = x_train[selected_features]
    x_val_selected = x_val[selected_features]
    
    # Treinar modelo final
    model = _train_quick_xgb(x_train_selected, y_train)
    
    # Obter probabilidades
    dval = xgb.DMatrix(x_val_selected)
    probabilities = model.predict(dval)
    
    # Aplicar calibração se necessário
    if calibration_method != 'none':
        try:
            calibrator = create_calibrated_model(
                model, x_train_selected, y_train, x_val_selected, y_val, calibration_method
            )
            probabilities = calibrator.predict_proba(x_val_selected)
        except Exception as e:
            print(f"  Erro na calibração {calibration_method}: {e}")
            calibration_method = 'none'
    
    # Avaliar calibração
    calib_metrics = evaluate_calibration_quality(y_val, probabilities, f"{ticker}_{calibration_method}")
    
    # Preparar dados de validação para backtest
    val_start = config['model_training']['validation_start_date']
    val_end = config['model_training']['validation_end_date']
    validation_dataframe = df_labeled[val_start:val_end]
    
    # Otimizar thresholds
    try:
        buy_threshold, sell_threshold, best_sharpe = optimize_trading_thresholds_financial_with_probs(
            probabilities, validation_dataframe, 
            config['backtesting']['initial_capital'],
            config['backtesting']['transaction_cost_pct'],
            minimum_hold_days=3
        )
    except Exception as e:
        print(f"  Erro na otimização de thresholds: {e}")
        best_sharpe = -np.inf
        buy_threshold, sell_threshold = 0.5, 0.5
    
    # Calcular métricas adicionais
    p_up = probabilities[:, 2] if probabilities.ndim == 2 and probabilities.shape[1] >= 3 else probabilities
    y_val_bin = (y_val == 2).astype(int)
    
    from sklearn.metrics import f1_score, average_precision_score
    f1_up = f1_score(y_val_bin, (p_up >= 0.5).astype(int)) if len(np.unique(y_val_bin)) > 1 else 0.0
    auprc = average_precision_score(y_val_bin, p_up) if len(np.unique(y_val_bin)) > 1 else 0.0
    
    return {
        'sharpe_ratio': best_sharpe,
        'f1_up': f1_up,
        'auprc': auprc,
        'brier_score': calib_metrics['brier_score'],
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'n_features': len(selected_features),
        'calibration_method': calibration_method
    }


def run_grid_search_for_ticker(
    ticker: str,
    df_features: pd.DataFrame,
    config: dict,
    protected_columns: List[str] = None
) -> Dict[str, Any]:
    """Executa grid search para um ticker específico."""
    
    if protected_columns is None:
        protected_columns = DEFAULT_PROTECTED_COLUMNS
    
    print(f"\n=== Grid Search para {ticker} ===")
    
    # Garantir dados com target
    df_labeled = _ensure_labeled_dataframe(ticker, df_features, config)
    
    results = []
    best_score = -np.inf
    best_params = None
    
    # Gerar todas as combinações
    combinations = list(itertools.product(
        GRID_PARAMS['top_k'],
        GRID_PARAMS['corr_threshold'],
        GRID_PARAMS['calibration']
    ))
    
    print(f"Testando {len(combinations)} combinações...")
    
    for i, (top_k, corr_threshold, calibration) in enumerate(combinations):
        print(f"  {i+1}/{len(combinations)}: K={top_k}, corr={corr_threshold}, calib={calibration}")
        
        try:
            # Selecionar features
            selected_features, importance_df = _select_features_for_params(
                df_features, df_labeled, top_k, corr_threshold, protected_columns, config
            )
            
            if len(selected_features) < 5:  # Muito poucas features
                print(f"    Pulando: apenas {len(selected_features)} features")
                continue
            
            # Avaliar combinação
            metrics = _evaluate_combination(
                ticker, df_labeled, selected_features, calibration, config
            )
            
            # Adicionar parâmetros
            metrics.update({
                'top_k': top_k,
                'corr_threshold': corr_threshold,
                'calibration': calibration,
                'selected_features': selected_features
            })
            
            results.append(metrics)
            
            # Atualizar melhor combinação (Sharpe como critério principal)
            if metrics['sharpe_ratio'] > best_score:
                best_score = metrics['sharpe_ratio']
                best_params = metrics.copy()
            
            print(f"    Sharpe: {metrics['sharpe_ratio']:.3f}, F1: {metrics['f1_up']:.3f}, Features: {len(selected_features)}")
            
        except Exception as e:
            print(f"    Erro: {e}")
            continue
    
    if best_params is None:
        print(f"  Nenhuma combinação válida encontrada para {ticker}")
        return {}
    
    print(f"\n  Melhor combinação para {ticker}:")
    print(f"    K={best_params['top_k']}, corr={best_params['corr_threshold']}, calib={best_params['calibration']}")
    print(f"    Sharpe: {best_params['sharpe_ratio']:.3f}, F1: {best_params['f1_up']:.3f}")
    print(f"    Features: {len(best_params['selected_features'])}")
    
    return {
        'ticker': ticker,
        'best_params': best_params,
        'all_results': results
    }


def run_grid_search_all_tickers() -> Dict[str, Any]:
    """Executa grid search para todos os tickers."""
    
    config = _load_config()
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / 'data' / '03_features'
    output_dir = project_root / 'data' / '03_features_selected'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_dir = project_root / 'reports' / 'feature_selection_grid'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = config['data']['tickers']
    all_results = {}
    
    for ticker in tickers:
        feature_file = input_dir / f'{ticker}.csv'
        if not feature_file.exists():
            print(f"⚠️  Features não encontradas para {ticker}: {feature_file}. Pulando...")
            continue
        
        df_features = pd.read_csv(feature_file, index_col='Date', parse_dates=True)
        
        ticker_results = run_grid_search_for_ticker(ticker, df_features, config)
        
        if ticker_results:
            all_results[ticker] = ticker_results
            
            # Salvar melhor combinação
            best_params = ticker_results['best_params']
            selected_features = best_params['selected_features']
            
            # Salvar CSV com features selecionadas
            df_selected = df_features[selected_features].copy()
            df_selected.to_csv(output_dir / f'{ticker}.csv')
            
            # Salvar manifest
            manifest = {
                'ticker': ticker,
                'best_params': {
                    'top_k': best_params['top_k'],
                    'corr_threshold': best_params['corr_threshold'],
                    'calibration': best_params['calibration'],
                    'sharpe_ratio': best_params['sharpe_ratio'],
                    'f1_up': best_params['f1_up'],
                    'auprc': best_params['auprc'],
                    'brier_score': best_params['brier_score'],
                    'n_features': best_params['n_features']
                },
                'selected_features': selected_features
            }
            
            (manifest_dir / f'best_config_{ticker}.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2)
            )
            
            # Salvar todos os resultados
            results_df = pd.DataFrame(ticker_results['all_results'])
            results_df.to_csv(manifest_dir / f'grid_results_{ticker}.csv', index=False)
    
    # Salvar resumo global
    summary = {}
    for ticker, results in all_results.items():
        if results:
            summary[ticker] = results['best_params']
    
    (manifest_dir / 'grid_search_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    
    print(f"\n=== RESUMO DO GRID SEARCH ===")
    for ticker, results in all_results.items():
        if results:
            best = results['best_params']
            print(f"{ticker}: K={best['top_k']}, corr={best['corr_threshold']}, "
                  f"calib={best['calibration']}, Sharpe={best['sharpe_ratio']:.3f}")
    
    return all_results


def main():
    """Função principal."""
    run_grid_search_all_tickers()


if __name__ == '__main__':
    main()
