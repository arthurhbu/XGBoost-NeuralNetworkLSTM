import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import optuna
import json
from itertools import product

class XGBoostWrapper:
    """Wrapper para compatibilidade com sklearn."""
    
    def __init__(self, model):
        self.model = model
        self.is_fitted_ = True
        self._estimator_type = 'classifier'
        self.classes_ = None

    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def create_dynamic_triple_barrier_target(df, target_column, profit_multiplier, loss_multiplier, holding_days=7):
    """
    Cria target ternário usando Triple Barrier com ATR dinâmico.
    
    Args:
        df: DataFrame com dados OHLCV e ATR
        target_column: Nome da coluna target
        profit_multiplier: Multiplicador ATR para lucro
        loss_multiplier: Multiplicador ATR para perda
        holding_days: Dias máximos de holding
        
    Returns:
        DataFrame com target criado
    """
    if 'ATR' not in df.columns:
        raise ValueError('Coluna ATR não encontrada no DataFrame')
    
    target = np.full(len(df), np.nan)
    
    for i in range(len(df) - holding_days):
        entry_price = df['Open'].iloc[i].item()
        atr_value = df['ATR'].iloc[i-1].item() if i > 0 else df['ATR'].iloc[i].item()
        
        if pd.isna(atr_value) or atr_value == 0:
            continue
            
        profit_barrier = entry_price + (profit_multiplier * atr_value)
        loss_barrier = entry_price - (loss_multiplier * atr_value)
        outcome = np.nan
        
        for j in range(1, holding_days + 1):
            if i + j >= len(df):
                break
                
            day_high, day_low = df['High'].iloc[i+j].item(), df['Low'].iloc[i+j].item()
            
            if day_high >= profit_barrier:
                outcome = 1
                break
            elif day_low <= loss_barrier:
                outcome = -1
                break
                
        if pd.isna(outcome):
            outcome = 0
            
        target[i] = outcome
    
    df[target_column] = target
    label_map = {-1: 0, 0: 1, 1: 2}
    df[target_column] = df[target_column].map(label_map)
    
    return df.dropna(subset=[target_column])


def create_FIXED_triple_barrier_target(df, target_column_name, holding_days, profit_threshold, loss_threshold):
    """
    Cria target ternário usando Triple Barrier com barreiras fixas.
    
    Args:
        df: DataFrame com dados OHLCV
        target_column_name: Nome da coluna target
        holding_days: Dias máximos de holding
        profit_threshold: Threshold de lucro (ex: 0.03 para 3%)
        loss_threshold: Threshold de perda (ex: -0.015 para -1.5%)
        
    Returns:
        DataFrame com target criado
    """
    target = np.full(len(df), np.nan) 

    for i in range(len(df) - holding_days):
        entry_price = df['Open'].iloc[i].item()
        
        profit_barrier = entry_price * (1 + profit_threshold)
        loss_barrier = entry_price * (1 + loss_threshold)
        
        outcome = np.nan

        for j in range(1, holding_days + 1):
            if i + j >= len(df): 
                break

            day_high = df['High'].iloc[i+j].item()
            day_low = df['Low'].iloc[i+j].item()

            if day_high >= profit_barrier:
                outcome = 1
                break
            elif day_low <= loss_barrier:
                outcome = -1
                break
        
        if pd.isna(outcome):
            outcome = 0
            
        target[i] = outcome

    df[target_column_name] = target
    
    label_map = {-1: 0, 0: 1, 1: 2}
    df[target_column_name] = df[target_column_name].map(label_map)
    
    return df.dropna(subset=[target_column_name])


def split_data(df, train_final_date, validation_start_date, validation_end_date, 
               test_start_date, test_end_date, target_column_name):
    """
    Divide dados em conjuntos de treino, validação e teste.
    
    Args:
        df: DataFrame com dados
        train_final_date: Data final do treino
        validation_start_date: Data inicial da validação
        validation_end_date: Data final da validação
        test_start_date: Data inicial do teste
        test_end_date: Data final do teste
        target_column_name: Nome da coluna target
        
    Returns:
        Tupla com (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    df.index = pd.to_datetime(df.index)
    
    train_data = df[df.index <= train_final_date]
    val_data = df[(df.index >= validation_start_date) & (df.index < validation_end_date)]
    test_data = df[(df.index >= test_start_date) & (df.index <= test_end_date)]
    
    x_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    x_val = val_data.drop(columns=[target_column_name])
    y_val = val_data[target_column_name]
    x_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def run_optimization_backtest(validation_df, probabilities, buy_threshold, sell_threshold):
    """
    Executa backtest simplificado para otimização de thresholds.
    
    Args:
        validation_df: DataFrame de validação
        probabilities: Probabilidades do modelo
        buy_threshold: Threshold de compra
        sell_threshold: Threshold de venda
        
    Returns:
        Sharpe ratio do backtest
    """
    initial_capital, transaction_cost = 100000.0, 0.001
    cash, stocks_held = initial_capital, 0.0
    portfolio_history = []
    scores = probabilities[:, 2] - probabilities[:, 0]
    
    if len(validation_df) < 10 or len(scores) < 10:
        return -np.inf
    
    for i in range(len(validation_df) - 1):
        if i >= len(scores):
            break
            
        score, exec_price = scores[i], validation_df['Open'].iloc[i + 1]
        
        if score >= buy_threshold and cash > exec_price:
            stocks_to_buy = cash / exec_price
            cost = stocks_to_buy * exec_price * transaction_cost
            stocks_held += stocks_to_buy
            cash -= (stocks_to_buy * exec_price) + cost
        elif score <= sell_threshold and stocks_held > 0:
            sale_value = stocks_held * exec_price
            cost = sale_value * transaction_cost
            cash += sale_value - cost
            stocks_held = 0
            
        portfolio_history.append(cash + stocks_held * validation_df['Close'].iloc[i])
    
    if not portfolio_history or len(portfolio_history) < 5:
        return -np.inf
        
    portfolio_series = pd.Series(portfolio_history)
    daily_returns = portfolio_series.pct_change().dropna()
    
    if daily_returns.empty or daily_returns.std() == 0:
        return -np.inf
        
    ann_return = (1 + (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1)) ** (252.0 / len(portfolio_series)) - 1
    ann_volatility = daily_returns.std() * np.sqrt(252)
    
    return ann_return / ann_volatility if ann_volatility != 0 else -np.inf


def find_optimal_score_thresholds(validation_df, probabilities):
    """
    Encontra thresholds ótimos baseado em score.
    
    Args:
        validation_df: DataFrame de validação
        probabilities: Probabilidades do modelo
        
    Returns:
        Tupla com (best_thresholds, best_sharpe)
    """
    buy_grid = np.arange(0.05, 0.8, 0.05)
    sell_grid = np.arange(-0.8, 0.1, 0.05)
    best_sharpe, best_thresholds = -np.inf, (0.5, -0.5)
    
    for th_buy in buy_grid:
        for th_sell in sell_grid:
            if th_buy <= th_sell:
                continue
                
            sharpe = run_optimization_backtest(validation_df, probabilities, th_buy, th_sell)
            
            if sharpe > best_sharpe:
                best_sharpe, best_thresholds = sharpe, (th_buy, th_sell)
    
    return best_thresholds, best_sharpe


def find_optimal_target_params(df_features, config, base_model_params, ticker):
    """
    Encontra parâmetros ótimos para criação de targets.
    
    Args:
        df_features: DataFrame com features
        config: Configuração do modelo
        base_model_params: Parâmetros base do modelo
        ticker: Símbolo do ticker
        
    Returns:
        Dicionário com parâmetros ótimos
    """
    strategy_grid = config['model_training']['triple_barrier_grid']
    atr_grid_cfg = config['model_training'].get('target_grid_atr', None)
    rules = config['model_training'].get('target_selection_rules', {})
    
    min_up_ratio = float(rules.get('min_up_ratio', 0.15))
    max_up_ratio = float(rules.get('max_up_ratio', 0.40))
    min_num_classes = int(rules.get('min_num_classes', 3))
    min_samples_per_class = int(rules.get('min_samples_per_class', 50))
    
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = reports_dir / f"target_grids_results.csv"
    
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "04_labeled" / "cache_targets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    best_score, best_params = -np.inf, None
    results = []
    
    if atr_grid_cfg:
        holding_days_grid = atr_grid_cfg.get('holding_days', [7,10])
        k_profit_grid = atr_grid_cfg.get('k_profit_atr_grid', [1.0, 1.5, 2.0])
        k_loss_grid = atr_grid_cfg.get('k_loss_atr_grid', [0.8, 1.0])
        
        print(f"Testando grid ATR para {ticker}: {len(holding_days_grid)}x{len(k_profit_grid)}x{len(k_loss_grid)} combinações")
        
        for h, kpa, kla in product(holding_days_grid, k_profit_grid, k_loss_grid):
            cache_csv = cache_dir / f"{ticker}_ATR_h{h}_kp{kpa}_kl{kla}.csv"
            
            if cache_csv.exists():
                df_labeled = pd.read_csv(cache_csv, index_col='Date', parse_dates=True)
            else: 
                try:
                    df_labeled = create_dynamic_triple_barrier_target(
                        df_features.copy(), 
                        config['model_training']['target_column'], 
                        kpa, kla, h
                    )
                    df_labeled.to_csv(cache_csv)
                except Exception as e:
                    _append_log_row(log_path, {
                        'ticker': ticker, 'holding_days': h, 'k_profit_atr': kpa, 'k_loss_atr': kla, 
                        'status': 'rejected', 'label_error': str(e)
                    })
                    continue
            
            x_train, y_train, x_val, y_val, _, _ = split_data(
                df_labeled, 
                config['model_training']['train_final_date'], 
                config['model_training']['validation_start_date'], 
                config['model_training']['validation_end_date'], 
                config['model_training']['test_start_date'], 
                config['model_training']['test_end_date'], 
                target_column_name=config['model_training']['target_column']
            )
            
            y_val_arr = y_val.values
            up_ratio_val = float((y_val_arr == 2).mean()) if len(y_val_arr) > 0 else 0.0
            unique_classes = np.unique(y_val_arr)
            class_counts = pd.Series(y_val_arr).value_counts()
            min_class_count = int(class_counts.min()) if not class_counts.empty else 0
            
            if (up_ratio_val < min_up_ratio or up_ratio_val > max_up_ratio or 
                len(unique_classes) < min_num_classes or 
                (min_samples_per_class and min_class_count < min_samples_per_class)):
                print(f"  Rejeitado: h={h}, kp={kpa}, kl={kla} - up_ratio={up_ratio_val:.3f}")
                _append_log_row(log_path, {
                    'ticker': ticker, 'holding_days': h, 'k_profit_atr': kpa, 'k_loss_atr': kla,
                    'up_ratio_val': up_ratio_val, 'num_classes_val': len(unique_classes), 'min_class_count_val': min_class_count,
                    'status': 'rejected', 'reason': 'early_filters'
                })
                continue
            
            model = _train_xgb_quick(x_train, y_train, x_val, y_val, base_model_params)
            
            dval = xgb.DMatrix(x_val)
            prob_val = model.predict(dval)
            
            auc_pr = _compute_metrics(y_val, prob_val)
            y_val_bin = (y_val == 2).astype(int)
            p_up_val = prob_val[:, 2] if prob_val.ndim == 2 and prob_val.shape[1] >= 3 else prob_val
            preds_bin = (p_up_val >= 0.5).astype(int)
            f1_up = float(f1_score(y_val_bin, preds_bin)) if len(np.unique(y_val_bin)) > 1 else float('nan')

            print(f"  Aceito: h={h}, kp={kpa}, kl={kla} - up_ratio={up_ratio_val:.3f}, auc_pr={auc_pr:.3f}, f1={f1_up:.3f}")
            row = {
                'ticker': ticker, 'holding_days': h, 'k_profit_atr': kpa, 'k_loss_atr': kla,
                'up_ratio_val': up_ratio_val, 'num_classes_val': len(unique_classes),
                'auc_pr_up': auc_pr, 'f1_up': f1_up, 'status': 'accepted'
            }
            _append_log_row(log_path, row)

            metric_order = config['model_training'].get('target_selection_metric', ["auc_pr_up", "f1_up"])
            score = 0.0
            for idx, m in enumerate(metric_order):
                val = row.get(m)
                if pd.notna(val):
                    weight = 1.0 / (idx + 1)
                    score += float(val) * weight

            results.append({'params': {'holding_days': h, 'k_profit_atr': kpa, 'k_loss_atr': kla},
                            'score': score, 'auc_pr_up': auc_pr, 'f1_up': f1_up, 'up_ratio_val': up_ratio_val})

            if score > best_score:
                best_score = score
                best_params = {'holding_days': h, 'k_profit_atr': kpa, 'k_loss_atr': kla}

    # Fallback: grid percentual
    if not best_params and strategy_grid:
        print(f"Testando {len(strategy_grid)} combinações de parâmetros...")
        for i, params in enumerate(strategy_grid):
            try:
                df_temp = create_FIXED_triple_barrier_target(
                    df_features.copy(),
                    config['model_training']['target_column'],
                    **params
                )

                x_train, y_train, x_val, y_val, _, _ = split_data(
                    df_temp,
                    config['model_training']['train_final_date'],
                    config['model_training']['validation_start_date'],
                    config['model_training']['validation_end_date'],
                    config['model_training']['test_start_date'],
                    config['model_training']['test_end_date'],
                    target_column_name=config['model_training']['target_column']
                )

                class_dist = y_train.value_counts(normalize=True)
                up_class_ratio = class_dist.get(2, 0.0)
                unique_classes = y_train.unique()
                if len(unique_classes) < 2 or up_class_ratio < 0.10:
                    continue

                dtrain = xgb.DMatrix(x_train, label=y_train)
                model = xgb.train(base_model_params, dtrain, num_boost_round=base_model_params['n_estimators'])

                val_df = df_features[df_features.index.isin(x_val.index)]
                dval = xgb.DMatrix(x_val, label=y_val)
                probabilities = model.predict(dval)
                _, sharpe = find_optimal_score_thresholds(val_df, probabilities)

                score = sharpe + (up_class_ratio * 0.3) + (1 - abs(up_class_ratio - 0.2) * 2)
                if score > best_score:
                    best_score, best_params = score, params

            except Exception as e:
                print(f"  Parâmetros {i+1}: Erro - {str(e)}")
                continue

    return best_params


def _append_log_row(log_csv, row_dict):
    """Anexa linha ao log CSV."""
    import csv
    header = list(row_dict.keys())
    with open(log_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not log_csv.exists():
            w.writeheader()
        w.writerow(row_dict)


def _train_xgb_quick(x_train, y_train, x_val, y_val, base_model_params):
    """Treina modelo XGBoost rapidamente para validação."""
    params = dict(base_model_params)
    params['n_estimators'] = 100
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    return model


def _compute_metrics(y_true, proba_matrix):
    """Computa métricas de avaliação."""
    from sklearn.metrics import average_precision_score
    
    y_bin = (y_true == 2).astype(int)
    if proba_matrix.ndim == 1:
        p_up = proba_matrix
    else: 
        p_up = proba_matrix[:, 2] if proba_matrix.shape[1] >= 3 else proba_matrix[:, -1]
    
    if len(np.unique(y_bin)) < 2:
        return float('nan')
    return float(average_precision_score(y_bin, p_up))


def analyze_feature_importance(model, feature_names, ticker, top_n=15):
    """
    Analisa importância das features do modelo.
    
    Args:
        model: Modelo XGBoost treinado
        feature_names: Lista com nomes das features
        ticker: Nome do ticker
        top_n: Número de features top para mostrar
        
    Returns:
        DataFrame com importância das features
    """
    try:
        importance_dict = model.get_score(importance_type='gain')
        
        importance_data = []
        for feature in feature_names:
            importance_value = importance_dict.get(feature, 0.0)
            importance_data.append({
                'feature': feature,
                'importance_gain': importance_value,
                'ticker': ticker
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('importance_gain', ascending=False)
        
        total_importance = importance_df['importance_gain'].sum()
        if total_importance > 0:
            importance_df['importance_pct'] = (importance_df['importance_gain'] / total_importance) * 100
        else:
            importance_df['importance_pct'] = 0.0
        
        print(f"\n=== FEATURE IMPORTANCE - {ticker} ===")
        print(f"Total features: {len(feature_names)}")
        print(f"Features com importância > 0: {(importance_df['importance_gain'] > 0).sum()}")
        print(f"\nTop {top_n} features mais importantes:")
        print(importance_df.head(top_n)[['feature', 'importance_gain', 'importance_pct']].to_string(index=False))
        
        return importance_df
        
    except Exception as e:
        print(f"Erro ao analisar feature importance para {ticker}: {e}")
        return pd.DataFrame()


def objective(trial, x_train, y_train, x_val, y_val, objective_type='f1_macro'):
    """
    Função objetivo para otimização com Optuna.
    
    Args:
        trial: Trial do Optuna
        x_train: Features de treino
        y_train: Labels de treino
        x_val: Features de validação
        y_val: Labels de validação
        objective_type: Tipo de métrica objetivo
        
    Returns:
        Score do trial
    """
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return -np.inf
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'aucpr',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 256, 2048),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'seed': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    
    pruning_callback = XGBoostPruningCallback(trial, 'val-aucpr')
    
    if 'max_delta_step' not in params:
        params['max_delta_step'] = 1.0

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False,
        callbacks=[pruning_callback]
    )
    
    try:
        score = model.best_score
        if score is None or np.isnan(score):
            return -np.inf
        return float(score)
    except Exception as e:
        print(f"Erro no trial: {e}")
        return -np.inf


def main():
    """Função principal para treinamento dos modelos."""
    labeled_dir = Path(__file__).resolve().parents[2] / "data" / "04_labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    feature_data_path = config["data"]["features_data_path"]
    model_training_config = config["model_training"]
    
    from ..data.essential_columns_manager import EssentialColumnsManager
    essential_manager = EssentialColumnsManager()
    
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

    tickers = config["data"].get("tickers", [])
    for ticker in tickers:
        ticker_file = f"{ticker}.csv"
        labeled_file_path = labeled_dir / f"{ticker_file}"

        feature_csv_path = Path(feature_data_path) / ticker_file
        if not feature_csv_path.exists():
            print(f"Arquivo de features não encontrado para {ticker}: {feature_csv_path}. Pulando...")
            continue

        print(f"\n{'='*60}\nProcessando Ticker: {ticker}\n{'='*60}")
        df_features = pd.read_csv(str(feature_csv_path), index_col=0, parse_dates=True)
        
        print(f"Separando features do modelo das colunas essenciais para {ticker}...")
        
        df_essential = essential_manager.load_essential_columns(ticker)
        if df_essential is None:
            original_features_path = Path(__file__).resolve().parents[2] / "data" / "03_features" / ticker_file
            if original_features_path.exists():
                df_original = pd.read_csv(original_features_path, index_col=0, parse_dates=True)
                df_essential = essential_manager.extract_essential_columns(df_original, ticker)
            else:
                print(f"Não foi possível obter colunas essenciais para {ticker}")
                df_essential = pd.DataFrame()
        
        if not df_essential.empty:
            df_features_clean = df_features.drop(columns=df_essential.columns, errors='ignore')
            df_for_targets = pd.concat([df_features_clean, df_essential], axis=1)
            print(f"Usando {len(df_essential.columns)} colunas essenciais para criar targets")
        else:
            df_for_targets = df_features.copy()
            print(f"Usando apenas features selecionadas para criar targets")

        best_target_params = find_optimal_target_params(df_for_targets, config, base_params, ticker)
        
        if best_target_params is None:
            print(f"Não foi possível determinar parâmetros de target para {ticker}. Pulando...")
            continue

        if 'k_profit_atr' in best_target_params:
            df_final_labels = create_dynamic_triple_barrier_target(
                df_for_targets.copy(),
                model_training_config["target_column"],
                profit_multiplier=best_target_params['k_profit_atr'],
                loss_multiplier=best_target_params['k_loss_atr'],
                holding_days=best_target_params['holding_days']
            )
        else:
            df_final_labels = create_FIXED_triple_barrier_target(
                df_for_targets.copy(),
                model_training_config["target_column"],
                **best_target_params
            )
        
        essential_cols_to_exclude = ['Open', 'High', 'Low', 'Close']
        model_features = [col for col in df_features.columns if col not in essential_cols_to_exclude]
        df_final_labels = df_final_labels[model_features + [model_training_config["target_column"]]]
        
        print(f"Modelo treinará com {len(model_features)} features (sem data leakage)")
        print(f"Targets criados usando colunas essenciais: {df_essential.columns.tolist()}")
                        
        df_final_labels.to_csv(labeled_file_path)
        
        meta = {
            "ticker": ticker,
            "target_column": model_training_config["target_column"],
            "target_params": best_target_params,
            "train_final_date": config['model_training']['train_final_date'],
            "validation_start_date": config['model_training']['validation_start_date'],
            "validation_end_date": config['model_training']['validation_end_date'],
            "test_start_date": config['model_training']['test_start_date'],
            "test_end_date": config['model_training']['test_end_date']
        }
        (labeled_dir / f"{ticker}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            
        x_train, y_train, x_val, y_val, _, _ = split_data(
            df_final_labels, 
            config['model_training']['train_final_date'], 
            config['model_training']['validation_start_date'], 
            config['model_training']['validation_end_date'], 
            config['model_training']['test_start_date'], 
            config['model_training']['test_end_date'], 
            target_column_name=model_training_config['target_column']
        )
            
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val), n_trials=200)
            
        if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
            print(f"Nenhum trial completo para {ticker}. Usando parâmetros padrão.")
            final_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'aucpr',
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_lambda': 1.0,
                'reg_alpha': 0.0,
                'n_jobs': -1,
                'tree_method': 'hist',
                'seed': 42
            }
        else:
            final_params = {
                **study.best_params,    
                'objective': 'multi:softprob', 
                'num_class': 3, 
                'eval_metric': 'aucpr',
                'seed': 42,
                'n_jobs': -1
            }
            
        x_train_full, y_train_full = pd.concat([x_train, x_val]), pd.concat([y_train, y_val])
            
        dtrain_full = xgb.DMatrix(x_train_full, label=y_train_full)
        dval_final = xgb.DMatrix(x_val, label=y_val)
            
        booster = xgb.train(
            final_params,
            dtrain_full,
            num_boost_round=final_params['n_estimators'],
            evals=[(dval_final, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
            
        final_model = booster

        feature_names = x_train.columns.tolist()
        importance_df = analyze_feature_importance(final_model, feature_names, ticker, top_n=15)
            
        importance_output_path = Path(__file__).resolve().parents[2] / "reports" / "feature_importance" / f"feature_importance_{ticker}.csv"
        if not importance_df.empty:
            importance_df.to_csv(importance_output_path, index=False)
            print(f"Feature importance salva em: {importance_output_path}")

        model_path = Path(__file__).resolve().parents[2] / "models" / "01_xgboost" / "01_original" / f"{ticker.replace('.csv', '')}.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(model_path)
            
        print(f"Modelo para {ticker} salvo com sucesso.")


if __name__ == "__main__":
    main()