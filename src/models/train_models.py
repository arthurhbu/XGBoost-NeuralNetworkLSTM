import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
import optuna


def create_dynamic_triple_barrier_target(df, target_column, profit_multiplier, loss_multiplier, holding_days=7):
    if 'ATR' not in df.columns:
        raise ValueError('A coluna ATR não foi encontrada no DataFrame')
    
    target = np.full(len(df), np.nan)
    
    for i in range(len(df) - holding_days):
        entry_price = df['Open'].iloc[i]
        atr_value = df['ATR'].iloc[i-1] if i > 0 else df['ATR'].iloc[i]
        
        if pd.isna(atr_value) or atr_value == 0:
            continue
            
        profit_barrier = entry_price + (profit_multiplier * atr_value)
        loss_barrier = entry_price - (loss_multiplier * atr_value)
        outcome = np.nan
        
        for j in range(1, holding_days + 1):
            if i + j >= len(df):
                break
                
            day_high, day_low = df['High'].iloc[i+j], df['Low'].iloc[i+j]
            
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


def split_data(df, train_final_date, validation_start_date, validation_end_date, 
               test_start_date, test_end_date, target_column_name):
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
    initial_capital, transaction_cost = 100000.0, 0.001
    cash, stocks_held = initial_capital, 0.0
    portfolio_history = []
    scores = probabilities[:, 2] - probabilities[:, 0]
    
    # Verificar se temos dados suficientes
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
    # Grid mais amplo e realista para thresholds
    buy_grid = np.arange(0.05, 0.8, 0.05)  # De 0.05 a 0.75
    sell_grid = np.arange(-0.8, 0.1, 0.05)  # De -0.75 a 0.05
    best_sharpe, best_thresholds = -np.inf, (0.5, -0.5)
    
    for th_buy in buy_grid:
        for th_sell in sell_grid:
            if th_buy <= th_sell:
                continue
                
            sharpe = run_optimization_backtest(validation_df, probabilities, th_buy, th_sell)
            
            if sharpe > best_sharpe:
                best_sharpe, best_thresholds = sharpe, (th_buy, th_sell)
    
    return best_thresholds, best_sharpe


def find_optimal_target_params(df_features, config, base_model_params):
    
    strategy_grid = config['model_training']['triple_barrier_grid']
    best_score, best_params = -np.inf, None
    results = []
    
    print(f"Testando {len(strategy_grid)} combinações de parâmetros...")
    
    for i, params in enumerate(strategy_grid):
        try:
            df_temp = create_dynamic_triple_barrier_target(
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
            
            # Verificar distribuição de classes
            class_dist = y_train.value_counts(normalize=True)
            up_class_ratio = class_dist.get(2, 0.0)
            down_class_ratio = class_dist.get(0, 0.0)
            flat_class_ratio = class_dist.get(1, 0.0)
            
            # Critério mais flexível - aceitar estratégias com pelo menos 5% de classe Up
            if up_class_ratio < 0.05:
                print(f"  Parâmetros {i+1}: Rejeitado - Classe Up muito baixa ({up_class_ratio:.3f})")
                continue
                
            model = xgb.XGBClassifier(**base_model_params).fit(x_train, y_train)
            
            val_df = df_features[df_features.index.isin(x_val.index)]
            probabilities = model.predict_proba(x_val)
            
            _, sharpe = find_optimal_score_thresholds(val_df, probabilities)
            
            # Score melhorado: combinar Sharpe ratio com distribuição de classes
            score = sharpe + (up_class_ratio * 0.3) + (1 - abs(up_class_ratio - 0.2) * 2)  # Penalizar muito desbalanceado
            
            results.append({
                'params': params,
                'sharpe': sharpe,
                'up_ratio': up_class_ratio,
                'score': score,
                'class_dist': class_dist
            })
            
            print(f"  Parâmetros {i+1}: Sharpe={sharpe:.3f}, Up={up_class_ratio:.3f}, Score={score:.3f}")
            
            if score > best_score:
                best_score, best_params = score, params
                
        except Exception as e:
            print(f"  Parâmetros {i+1}: Erro - {str(e)}")
            continue
    
    if best_params:
        print(f"\n✅ Melhores parâmetros encontrados: {best_params}")
        print(f"   Score: {best_score:.3f}")
        # Mostrar top 3 resultados
        results.sort(key=lambda x: x['score'], reverse=True)
        print("\nTop 3 estratégias:")
        for i, result in enumerate(results[:3]):
            print(f"  {i+1}. {result['params']} - Score: {result['score']:.3f}, Sharpe: {result['sharpe']:.3f}")
    else:
        print("\n❌ Nenhuma estratégia de target viável encontrada.")
    
    return best_params


def calculate_class_weights(y_train):
    """
    Calcula pesos para balanceamento de classes usando sklearn.
    
    Args:
        y_train: Array com labels de treinamento
    
    Returns:
        dict: Dicionário com pesos para cada classe
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))
    
    print(f"Pesos das classes calculados: {weight_dict}")
    return weight_dict

def objective(trial, x_train, y_train, x_val, y_val):
    # Calcular pesos das classes
    class_weights = calculate_class_weights(y_train)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'seed': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'scale_pos_weight': class_weights.get(2, 1.0)  # Peso para classe Up (2)
    }
    
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

    try:
        return float(model.best_score)
    except Exception:
        proba = model.predict(dval)
        eps = 1e-12
        y_true = y_val.astype(int)
        log_probs = -np.log(np.clip(proba[np.arange(len(y_true)), y_true], eps, 1.0))
        return float(np.mean(log_probs))


def main():
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    feature_data_path = config["data"]["features_data_path"]
    model_training_config = config["model_training"]
    base_params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'n_estimators': 500,
        'max_depth': 5,
        'n_jobs': -1
    }

    for ticker_file in os.listdir(feature_data_path):
        if ticker_file.endswith('.csv'):
            ticker = ticker_file.replace('.csv', '')
            print(f"\n{'='*60}\nProcessando Ticker: {ticker}\n{'='*60}")
            
            df_features = pd.read_csv(f'{feature_data_path}/{ticker_file}', index_col=0, parse_dates=True)
            
            # best_target_params = find_optimal_target_params(df_features, config, base_params)
            
            # Usar os parâmetros otimizados encontrados
            df_final_labels = create_dynamic_triple_barrier_target(
                df_features, 
                model_training_config["target_column"], 
                profit_multiplier=2.0,
                loss_multiplier=1.5,
                holding_days=7
            )
            
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                df_final_labels, 
                config['model_training']['train_final_date'], 
                config['model_training']['validation_start_date'], 
                config['model_training']['validation_end_date'], 
                config['model_training']['test_start_date'], 
                config['model_training']['test_end_date'], 
                target_column_name=model_training_config['target_column']
            )
            
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val), n_trials=50)
            
            best_model_params = study.best_params
            
            # Calcular pesos das classes para o modelo final
            class_weights = calculate_class_weights(y_train)
            
            final_params = {
                **best_model_params, 
                'objective': 'multi:softprob', 
                'num_class': 3, 
                'eval_metric': 'mlogloss',
                'scale_pos_weight': class_weights.get(2, 1.0)  # Peso para classe Up (2)
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
            
            final_model = xgb.XGBClassifier(**final_params)
            final_model._Booster = booster
            final_model._le = None

            model_path = Path(__file__).resolve().parents[2] / "models" / "01_xgboost" / f"{ticker.replace('.csv', '')}.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(model_path)
            
            print(f"\nModelo para {ticker} salvos com sucesso.")


if __name__ == "__main__":
    main()