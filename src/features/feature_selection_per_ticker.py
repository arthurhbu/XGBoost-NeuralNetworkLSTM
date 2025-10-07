#!/usr/bin/env python3
"""
Seleção de Features por Ticker
==============================

Este módulo executa seleção de features individual por ticker, baseada em:
- Importância (gain) de um XGBoost rápido treinado no período de treino/val
- Poda por correlação entre features selecionadas
- Colunas protegidas que sempre são mantidas (ex.: OHLC, ATR, Volume)

Fluxo por ticker:
1) Carrega `config.yaml` e localiza dados de features em data/03_features/{TICKER}.csv
2) Gera (ou carrega) os rótulos em data/04_labeled/{TICKER}.csv
3) Faz split temporal (train/val/test) conforme configuração
4) Treina um XGBoost rápido no conjunto de treino (ou treino+val)
5) Obtém importâncias (gain), seleciona Top-K e aplica poda por correlação
6) Garante colunas protegidas
7) Salva CSV por ticker em data/03_features_selected/{TICKER}.csv
8) Salva manifest JSON com a lista final de features em reports/feature_selection/

Uso:
    python -m src.features.feature_selection_per_ticker --top_k 30 --corr_threshold 0.9

Observações importantes:
- A geração do target usa as funções já existentes em src.models.train_models
- A seleção usa apenas dados até a fronteira de treino/val para evitar vazamento
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from ..models.train_models import (
    split_data,
    find_optimal_target_params,
    create_FIXED_triple_barrier_target,
)


DEFAULT_PROTECTED_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'ATR', 'Volume'
]


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _ensure_labeled_dataframe(ticker: str, df_features: pd.DataFrame, config: dict) -> pd.DataFrame:
    labeled_dir = Path(__file__).resolve().parents[2] / 'data' / '04_labeled'
    labeled_dir.mkdir(parents=True, exist_ok=True)
    labeled_csv = labeled_dir / f"{ticker}.csv"
    if labeled_csv.exists():
        return pd.read_csv(labeled_csv, index_col='Date', parse_dates=True)

    # Se não existir, determinar parâmetros do target e gerar labels
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
        # Grid percentual FIXO
        df_labeled = create_FIXED_triple_barrier_target(
            df_features.copy(),
            config['model_training']['target_column'],
            **(best_target_params or {'holding_days': 7, 'profit_threshold': 0.03, 'loss_threshold': -0.02})
        )
    else:
        # Caso ATR tenha sido escolhido em find_optimal_target_params, o target já foi salvo lá
        # Para manter simplicidade, reutilizamos o CSV salvo por train_models quando existir
        # Se não existir, cairíamos no FIXED acima
        df_labeled = df_features.copy()

    # Persistir para reuso
    df_labeled.to_csv(labeled_csv)
    return df_labeled


def _train_quick_xgb(x_train: pd.DataFrame, y_train: pd.Series) -> xgb.Booster:
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
    importance_dict = model.get_score(importance_type='gain')
    data = [{'feature': f, 'gain': float(importance_dict.get(f, 0.0))} for f in feature_names]
    df = pd.DataFrame(data).sort_values('gain', ascending=False)
    return df


def _prune_by_correlation(df: pd.DataFrame, selected_cols: List[str], corr_threshold: float) -> List[str]:
    if not selected_cols:
        return selected_cols
    # Correlação baseada no período de treino para evitar vazamento
    corr = df[selected_cols].corr().abs()
    keep = []
    for col in selected_cols:
        if all(corr[col][k] <= corr_threshold for k in keep):
            keep.append(col)
    return keep


def _select_features_for_ticker(
    ticker: str,
    df_features: pd.DataFrame,
    config: dict,
    top_k: int,
    corr_threshold: float,
    protected_columns: List[str],
) -> Tuple[List[str], pd.DataFrame]:

    # Garantir colunas protegidas existentes no DF
    protected_present = [c for c in protected_columns if c in df_features.columns]

    # Garantir rótulos
    df_labeled = _ensure_labeled_dataframe(ticker, df_features, config)

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

    # Conjunto para importância: treino+val (sem teste)
    x_imp = pd.concat([x_train, x_val])
    y_imp = pd.concat([y_train, y_val])

    # Treino rápido
    model = _train_quick_xgb(x_imp, y_imp)

    # Importância e Top-K
    importance_df = _get_feature_importance_gain(model, x_imp.columns.tolist())
    ranked_cols = importance_df['feature'].tolist()
    top_cols = ranked_cols[:top_k]

    # Poda por correlação no conjunto treino+val
    pruned_cols = _prune_by_correlation(x_imp, top_cols, corr_threshold)

    # Garantir colunas protegidas
    final_cols = list(dict.fromkeys(protected_present + pruned_cols))

    return final_cols, importance_df


def run_selection(top_k: int = 30, corr_threshold: float = 0.9, protected: List[str] | None = None) -> Dict[str, List[str]]:
    
    config = _load_config()
    project_root = Path(__file__).resolve().parents[2]
    input_dir = project_root / 'data' / '03_features'
    output_dir = project_root / 'data' / '03_features_selected' 
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = project_root / 'reports' / 'feature_selection'
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tickers = config['data']['tickers']
    protected_columns = protected if protected is not None else DEFAULT_PROTECTED_COLUMNS

    selection_manifest: Dict[str, List[str]] = {}

    for ticker in tickers:
        feature_file = input_dir / f'{ticker}.csv'
        if not feature_file.exists():
            print(f"⚠️  Features não encontradas para {ticker}: {feature_file}. Pulando...")
            continue

        print(f"\n=== Seleção por ticker: {ticker} ===")
        df_features = pd.read_csv(feature_file, index_col='Date', parse_dates=True)

        selected_cols, importance_df = _select_features_for_ticker(
            ticker,
            df_features,
            config,
            top_k=top_k,
            corr_threshold=corr_threshold,
            protected_columns=protected_columns,
        )

        # Salvar CSV selecionado
        df_selected = df_features[selected_cols].copy()
        df_selected.to_csv(output_dir / f'{ticker}.csv')

        # Salvar manifest e importância
        selection_manifest[ticker] = selected_cols
        (manifest_dir / f'selected_features_{ticker}.json').write_text(
            json.dumps({'ticker': ticker, 'selected_features': selected_cols}, ensure_ascii=False, indent=2)
        )
        importance_path = manifest_dir / f'feature_importance_{ticker}.csv'
        importance_df.to_csv(importance_path, index=False)

        print(f"  Features selecionadas: {len(selected_cols)} | Salvo em: {output_dir / f'{ticker}.csv'}")

    # Manifest global
    (manifest_dir / 'selected_features_manifest.json').write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2)
    )

    return selection_manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Seleção de features por ticker (Top-K + correlação).')
    parser.add_argument('--top_k', type=int, default=30, help='Número máximo de features por importância (antes da poda).')
    parser.add_argument('--corr_threshold', type=float, default=0.9, help='Limite de correlação absoluta para poda.')
    parser.add_argument('--protect', type=str, nargs='*', default=None, help='Lista de colunas protegidas a manter.')
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    k = {30: 'k_30', 35: 'k_35'}
    corr_threshold = {0.90, 0.95, 0.98}
    for k in k:
        for corr_threshold in corr_threshold:
                run_selection(top_k=k, corr_threshold=corr_threshold, protected=args.protect)


if __name__ == '__main__':
    main()


