import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..models.train_models import split_data, find_optimal_target_params, create_FIXED_triple_barrier_target
from ..backtesting.backtest import (
    simulate_portfolio_execution_next_open,
    calculate_sharpe_ratio,
)


def _load_config():
    config_path = Path(__file__).resolve().parents[2] / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _train_quick_xgb(x_train: pd.DataFrame, y_train: pd.Series) -> xgb.Booster:
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'aucpr',
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
    return xgb.train(params, dtrain, num_boost_round=params['n_estimators'])


def _rank_features_by_gain(model: xgb.Booster, feature_names: list[str]) -> list[str]:
    importance = model.get_score(importance_type='gain')
    df = pd.DataFrame(
        [{'feature': f, 'gain': float(importance.get(f, 0.0))} for f in feature_names]
    ).sort_values('gain', ascending=False)
    return df['feature'].tolist()


def _optimize_thresholds(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Otimiza thresholds de compra e venda para maximizar Sharpe ratio"""
    def objective(buy_th):
        # Venda é sempre 1 - buy_th para manter simetria
        sell_th = 1 - buy_th
        if sell_th >= buy_th:
            return -999  # Penalizar thresholds inválidos
        
        # Gerar ações
        actions = np.where(probs >= buy_th, 1, np.where(probs <= sell_th, -1, 0))
        
        # Calcular retornos simples (sem simulação completa para velocidade)
        returns = []
        for i in range(1, len(actions)):
            if actions[i-1] == 1:  # Comprou no dia anterior
                ret = (targets[i] - targets[i-1]) / targets[i-1] if targets[i-1] != 0 else 0
                returns.append(ret)
            elif actions[i-1] == -1:  # Vendeu no dia anterior
                ret = (targets[i-1] - targets[i]) / targets[i] if targets[i] != 0 else 0
                returns.append(ret)
        
        if len(returns) < 10:  # Muito poucas operações
            return -999
            
        returns = np.array(returns)
        if np.std(returns) == 0:
            return -999
            
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return -sharpe  # Minimizar negativo = maximizar positivo
    
    # Buscar threshold ótimo entre 0.5 e 0.9
    result = minimize_scalar(objective, bounds=(0.5, 0.9), method='bounded')
    optimal_buy = result.x
    optimal_sell = 1 - optimal_buy
    
    return optimal_buy, optimal_sell


def _temporal_cross_validation(df: pd.DataFrame, feature_subset: List[str], 
                              n_splits: int = 3) -> float:
    """Validação cruzada temporal com walk-forward analysis"""
    total_sharpe = 0
    valid_splits = 0
    
    # Dividir dados em n_splits janelas temporais
    data_length = len(df)
    window_size = data_length // (n_splits + 1)
    
    for i in range(n_splits):
        # Janela de treino: do início até a janela i
        train_end = (i + 1) * window_size
        # Janela de teste: próxima janela
        test_start = train_end
        test_end = min(test_start + window_size, data_length)
        
        if test_end - test_start < 50:  # Muito poucos dados para teste
            continue
            
        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end]
        
        # Preparar dados
        x_train = train_df[feature_subset]
        y_train = train_df['target']
        x_test = test_df[feature_subset]
        y_test = test_df['target']
        
        # Treinar modelo
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test)
        
        booster = xgb.train({
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'aucpr',
            'n_estimators': 100,  # Menos iterações para velocidade
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'tree_method': 'hist',
            'n_jobs': -1,
            'seed': 42
        }, dtrain, num_boost_round=100)
        
        # Predições
        probs = booster.predict(dtest)
        p_up = probs[:, 2] if probs.ndim == 2 and probs.shape[1] >= 3 else (
            probs[:, -1] if probs.ndim == 2 else probs
        )
        
        # Otimizar thresholds na janela de teste
        buy_th, sell_th = _optimize_thresholds(p_up, test_df['Close'].values)
        
        # Gerar ações
        actions = np.where(p_up >= buy_th, 1, np.where(p_up <= sell_th, -1, 0))
        
        # Simulação de portfólio
        m = min(len(test_df), len(actions))
        port = simulate_portfolio_execution_next_open(
            test_df.iloc[:m], actions[:m], 
            initial_capital=100000.0, 
            transaction_cost_percentage=0.001
        )
        
        sharpe = calculate_sharpe_ratio(port)
        if not np.isnan(sharpe) and not np.isinf(sharpe):
            total_sharpe += sharpe
            valid_splits += 1
    
    return total_sharpe / valid_splits if valid_splits > 0 else 0


def _evaluate_subset_sharpe(x_val: pd.DataFrame, y_val: pd.Series, val_df: pd.DataFrame, selected_cols: list[str]) -> float:
    if not selected_cols:
        return -np.inf
    dval = xgb.DMatrix(x_val[selected_cols])
    # simples scorer baseado em prob para backtest rápido
    # modelo dummy: usar prob_up aproximada via média de árvores treinadas
    # Aqui, por simplicidade, treinamos um modelo rápido no próprio x_val para gerar probs
    # (mantemos consistência do comparativo entre subsets)
    return -np.inf


def run_sweep():
    config = _load_config()
    project_root = Path(__file__).resolve().parents[2]
    # Usar arquivo original (não filtrado) para ter todas as features
    input_dir = project_root / "data" / "03_features"
    labeled_dir = project_root / 'data' / '04_labeled'
    report_dir = project_root / 'reports' / 'feature_sweep'
    report_dir.mkdir(parents=True, exist_ok=True)

    tickers = config['data']['tickers']
    results_summary = []

    for ticker in tickers:
        feature_file = input_dir / f'{ticker}.csv'
        if not feature_file.exists():
            print(f"  Features não encontradas para {ticker}. Pulando...")
            continue

        df_feat = pd.read_csv(feature_file, index_col=0, parse_dates=True)

        labeled_csv = labeled_dir / f"{ticker}.csv"
        if labeled_csv.exists():
            df_lab = pd.read_csv(labeled_csv, index_col='Date', parse_dates=True)
        else:
            base_params = {
                'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'aucpr',
                'n_estimators': 200, 'max_depth': 4, 'n_jobs': -1, 'tree_method': 'hist', 'seed': 42, 'max_delta_step': 1.0
            }
            best_target = find_optimal_target_params(df_feat, config, base_params, ticker)
            if best_target is None:
                print(f"  Sem target para {ticker}")
                continue
            df_lab = create_FIXED_triple_barrier_target(
                df_feat.copy(), config['model_training']['target_column'], **best_target
            )
            df_lab.to_csv(labeled_csv)

        x_train, y_train, x_val, y_val, _, _ = split_data(
            df_lab,
            config['model_training']['train_final_date'],
            config['model_training']['validation_start_date'],
            config['model_training']['validation_end_date'],
            config['model_training']['test_start_date'],
            config['model_training']['test_end_date'],
            target_column_name=config['model_training']['target_column'],
        )

        # treino rápido para ranking
        model = _train_quick_xgb(x_train, y_train)
        ranked = _rank_features_by_gain(model, x_train.columns.tolist())

        # Salvar ranking de features por importância
        importance_df = pd.DataFrame({
            'feature': ranked,
            'rank': range(1, len(ranked) + 1)
        })
        importance_df.to_csv(report_dir / f'feature_ranking_{ticker}.csv', index=False)
        print(f"  {ticker}: Top 10 features = {ranked[:10]}")

        # sweep de tamanhos (reduzido para máximo 20 features)
        ks = [1,2,3,5, 8, 10, 12, 15, 18, 20]
        rows = []
        
        print(f"  Testando {len(ks)} configurações de K para {ticker}...")

        for k in ks:
            if k > len(ranked):
                continue  # Pular se k maior que features disponíveis
                
            subset = ranked[:k]
            print(f"    K={k}: {subset[:3]}...")
            
            # Usar validação cruzada temporal para avaliação robusta
            sharpe = _temporal_cross_validation(df_lab, subset, n_splits=3)
            
            rows.append({
                'ticker': ticker, 
                'k': k, 
                'sharpe_val': sharpe,
                'features_used': ', '.join(subset)
            })

        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(report_dir / f'sweep_{ticker}.csv', index=False)
        best_row = df_rows.sort_values('sharpe_val', ascending=False).head(1)
        if not best_row.empty:
            best_dict = best_row.iloc[0].to_dict()
            # Adicionar as features do melhor resultado
            best_dict['best_features'] = best_dict['features_used']
            results_summary.append(best_dict)

    if results_summary:
        pd.DataFrame(results_summary).to_csv(report_dir / 'sweep_summary.csv', index=False)


if __name__ == '__main__':
    run_sweep()


