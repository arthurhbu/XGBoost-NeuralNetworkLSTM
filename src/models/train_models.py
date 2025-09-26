import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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


def create_FIXED_triple_barrier_target(
    df: pd.DataFrame, 
    target_column_name: str, 
    holding_days: int, 
    profit_threshold: float, # ex: 0.03 para 3%
    loss_threshold: float    # ex: -0.015 para -1.5%
) -> pd.DataFrame:
    """
    Cria um target ternário [-1, 0, 1] usando o método Triple Barrier com
    barreiras de lucro e perda FIXAS (em porcentagem).
    """
    target = np.full(len(df), np.nan) 

    for i in range(len(df) - holding_days):
        entry_price = df['Open'].iloc[i]
        
        # Barreiras fixas
        profit_barrier = entry_price * (1 + profit_threshold)
        loss_barrier = entry_price * (1 + loss_threshold) # loss_threshold já é negativo
        
        outcome = np.nan

        for j in range(1, holding_days + 1):
            if i + j >= len(df): break

            day_high = df['High'].iloc[i+j]
            day_low = df['Low'].iloc[i+j]

            if day_high >= profit_barrier:
                outcome = 1; break
            elif day_low <= loss_barrier:
                outcome = -1; break
        
        if pd.isna(outcome):
            outcome = 0
            
        target[i] = outcome

    df[target_column_name] = target
    
    label_map = {-1: 0, 0: 1, 1: 2}
    df[target_column_name] = df[target_column_name].map(label_map)
    
    return df.dropna(subset=[target_column_name])


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
            
            # Verificar distribuição de classes
            class_dist = y_train.value_counts(normalize=True)
            up_class_ratio = class_dist.get(2, 0.0)
            down_class_ratio = class_dist.get(0, 0.0)
            flat_class_ratio = class_dist.get(1, 0.0)
            
            # Critério mais flexível - aceitar estratégias com pelo menos 2% de classe Up
            # e verificar se há pelo menos 2 classes diferentes
            unique_classes = y_train.unique()
            if len(unique_classes) < 2:
                print(f"  Parâmetros {i+1}: Rejeitado - Apenas {len(unique_classes)} classe(s) encontrada(s)")
                continue
                
            if up_class_ratio < 0.02:
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

def calculate_sample_weights(y_train, class_weights):
    """
    Calcula sample_weights para cada amostra baseado nos pesos das classes.
    
    Args:
        y_train: Array com labels de treinamento
        class_weights: Dicionário com pesos para cada classe
    
    Returns:
        array: Array com pesos para cada amostra
    """
    sample_weights = np.array([class_weights[label] for label in y_train])
    print(f"Sample weights calculados - Min: {sample_weights.min():.3f}, Max: {sample_weights.max():.3f}, Mean: {sample_weights.mean():.3f}")
    return sample_weights

def apply_aggressive_class_balancing(x_train, y_train):
    """
    Aplica balanceamento mais agressivo para classes desbalanceadas.
    
    Args:
        x_train: Features de treinamento
        y_train: Labels de treinamento
    
    Returns:
        tuple: (x_train_balanced, y_train_balanced)
    """
    from sklearn.utils import resample
    
    # Verificar distribuição atual
    class_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"Distribuição original: {dict(class_counts)}")
    
    # Encontrar a classe majoritária
    majority_class = class_counts.idxmax()
    majority_count = class_counts.max()
    
    # Balancear para ter pelo menos 10% de cada classe minoritária
    target_minority_count = max(int(majority_count * 0.1), 10)
    
    x_train_balanced = [x_train]
    y_train_balanced = [y_train]
    
    for class_label in class_counts.index:
        if class_label == majority_class:
            continue
            
        class_mask = y_train == class_label
        x_class = x_train[class_mask]
        y_class = y_train[class_mask]
        
        if len(y_class) < target_minority_count:
            # Oversample da classe minoritária
            n_samples = target_minority_count - len(y_class)
            x_oversampled, y_oversampled = resample(
                x_class, y_class, 
                n_samples=n_samples, 
                random_state=42, 
                replace=True
            )
            x_train_balanced.append(x_oversampled)
            y_train_balanced.append(y_oversampled)
            print(f"Oversampled classe {class_label}: {len(y_class)} -> {len(y_class) + n_samples}")
    
    # Concatenar todos os dados
    x_final = pd.concat(x_train_balanced, ignore_index=True)
    y_final = pd.concat(y_train_balanced, ignore_index=True)
    
    # Verificar distribuição final
    final_counts = pd.Series(y_final).value_counts().sort_index()
    print(f"Distribuição após balanceamento: {dict(final_counts)}")
    
    return x_final, y_final

def calculate_objective_score(y_true, y_pred, probabilities, objective_type='f1_macro'):
    """
    Calcula diferentes tipos de scores para otimização.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições do modelo
        probabilities: Probabilidades do modelo
        objective_type: Tipo de objetivo ('f1_macro', 'f1_weighted', 'auc_ovr', 'precision_up', 'recall_up', 'f1_up', 'balanced_accuracy')
    
    Returns:
        float: Score calculado
    """
    try:
        if objective_type == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        elif objective_type == 'f1_weighted':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        elif objective_type == 'auc_ovr':
            # AUC One-vs-Rest para classe Up (2)
            y_binary = (y_true == 2).astype(int)
            if len(np.unique(y_binary)) > 1:  # Verificar se há ambas as classes
                return roc_auc_score(y_binary, probabilities[:, 2])
            else:
                return 0.0
        
        elif objective_type == 'precision_up':
            return precision_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)
        
        elif objective_type == 'recall_up':
            return recall_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)
        
        elif objective_type == 'f1_up':
            return f1_score(y_true, y_pred, labels=[2], average='macro', zero_division=0)
        
        elif objective_type == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        
        elif objective_type == 'custom_score':
            # Score customizado: combina F1 macro com penalização por desbalanceamento
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            up_ratio = np.mean(y_pred == 2)
            # Penalizar se não há predições Up ou se há muitas predições Up
            balance_penalty = 1.0 if 0.05 <= up_ratio <= 0.3 else 0.5
            return f1_macro * balance_penalty
        
        else:
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    except Exception as e:
        print(f"Erro ao calcular score {objective_type}: {e}")
        return 0.0

def objective(trial, x_train, y_train, x_val, y_val, objective_type='f1_macro'):
    # Verificar se há classes suficientes
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return -np.inf
    
    # Calcular pesos das classes com verificação
    class_weights = calculate_class_weights(y_train)
    sample_weights = calculate_sample_weights(y_train, class_weights)
    
    # Parâmetros do modelo
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
        'tree_method': 'hist'
    }
    
    dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(x_val, label=y_val)

    # Treinar modelo
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Fazer predições
    probabilities = model.predict(dval)
    y_pred = np.argmax(probabilities, axis=1)
    
    # Calcular score baseado no tipo de objetivo
    score = calculate_objective_score(y_val, y_pred, probabilities, objective_type)
    
    return score


def test_multiple_objectives(x_train, y_train, x_val, y_val, n_trials_per_objective=10):
    """
    Testa múltiplos objetivos de otimização e retorna o melhor.
    
    Args:
        x_train, y_train: Dados de treinamento
        x_val, y_val: Dados de validação
        n_trials_per_objective: Número de trials por objetivo
    
    Returns:
        tuple: (melhor_objectivo, melhor_score, melhor_params)
    """
    objectives_to_test = [
        ('f1_macro', 'F1-Score Macro'),
        ('f1_weighted', 'F1-Score Weighted'), 
        ('auc_ovr', 'AUC One-vs-Rest (Up)'),
        ('precision_up', 'Precision Classe Up'),
        ('recall_up', 'Recall Classe Up'),
        ('f1_up', 'F1-Score Classe Up'),
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('custom_score', 'Score Customizado')
    ]
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"TESTANDO MÚLTIPLOS OBJETIVOS DE OTIMIZAÇÃO")
    print(f"{'='*60}")
    
    for obj_type, obj_name in objectives_to_test:
        print(f"\n🎯 Testando: {obj_name} ({obj_type})")
        
        try:
            study = optuna.create_study(
                direction='maximize', 
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                lambda trial: objective(trial, x_train, y_train, x_val, y_val, obj_type), 
                n_trials=n_trials_per_objective
            )
            
            score = study.best_value
            params = study.best_params
            
            results.append({
                'objective': obj_type,
                'name': obj_name,
                'score': score,
                'params': params
            })
            
            print(f"  ✅ Score: {score:.4f}")
            
        except Exception as e:
            print(f"  ❌ Erro: {str(e)}")
            results.append({
                'objective': obj_type,
                'name': obj_name,
                'score': -np.inf,
                'params': None
            })
    
    # Encontrar o melhor resultado
    best_result = max(results, key=lambda x: x['score'])
    
    print(f"\n{'='*60}")
    print(f"RESULTADOS FINAIS")
    print(f"{'='*60}")
    
    # Ordenar por score
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    
    for i, result in enumerate(results_sorted[:5]):  # Top 5
        status = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"{status} {result['name']}: {result['score']:.4f}")
    
    print(f"\n🏆 MELHOR OBJETIVO: {best_result['name']}")
    print(f"   Score: {best_result['score']:.4f}")
    print(f"   Parâmetros: {best_result['params']}")
    
    return best_result['objective'], best_result['score'], best_result['params']

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
        if ticker_file != 'ITUB4.SA.csv':
            continue
        if ticker_file.endswith('.csv'):
            ticker = ticker_file.replace('.csv', '')
            print(f"\n{'='*60}\nProcessando Ticker: {ticker}\n{'='*60}")
            
            df_features = pd.read_csv(f'{feature_data_path}/{ticker_file}', index_col=0, parse_dates=True)
            
            # best_target_params = find_optimal_target_params(df_features, config, base_params)
            
            # Usar os parâmetros otimizados encontrados
            df_final_labels = create_FIXED_triple_barrier_target(
                df_features, 
                model_training_config["target_column"], 
                profit_threshold=0.05,
                loss_threshold=-0.025,
                holding_days=5
            )
            df_final_labels.to_csv(f'{feature_data_path}/final_labels_teste_para_descobrir_desbalanceamento.csv')
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                df_final_labels, 
                config['model_training']['train_final_date'], 
                config['model_training']['validation_start_date'], 
                config['model_training']['validation_end_date'], 
                config['model_training']['test_start_date'], 
                config['model_training']['test_end_date'], 
                target_column_name=model_training_config['target_column']
            )
            
            # Testar múltiplos objetivos de otimização
            best_objective, best_score, best_model_params = test_multiple_objectives(
                x_train, y_train, x_val, y_val, n_trials_per_objective=15
            )
            
            if best_model_params is None:
                print("❌ Nenhum objetivo funcionou, usando parâmetros padrão")
                best_model_params = {
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'n_estimators': 500,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'gamma': 0,
                    'reg_lambda': 1.0,
                    'reg_alpha': 0.0
                }
            
            # Aplicar balanceamento agressivo se necessário
            class_dist = y_train.value_counts(normalize=True)
            up_ratio = class_dist.get(2, 0.0)
            
            # if up_ratio < 0.05:  # Se classe Up for muito rara
            print(f"Aplicando balanceamento agressivo - Classe Up muito rara ({up_ratio:.3f})")
            x_train_balanced, y_train_balanced = apply_aggressive_class_balancing(x_train, y_train)
            # else:
                # x_train_balanced, y_train_balanced = x_train, y_train
            
            # Calcular pesos das classes para o modelo final
            class_weights = calculate_class_weights(y_train_balanced)
            sample_weights = calculate_sample_weights(y_train_balanced, class_weights)
            
            final_params = {
                **best_model_params, 
                'objective': 'multi:softprob', 
                'num_class': 3, 
                'eval_metric': 'mlogloss'
            }
            
            x_train_full, y_train_full = pd.concat([x_train_balanced, x_val]), pd.concat([y_train_balanced, y_val])
            
            # Calcular sample_weights para o conjunto completo de treino
            class_weights_full = calculate_class_weights(y_train_full)
            sample_weights_full = calculate_sample_weights(y_train_full, class_weights_full)
            
            dtrain_full = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
            dval_final = xgb.DMatrix(x_val, label=y_val)
            
            booster = xgb.train(
                final_params,
                dtrain_full,
                num_boost_round=final_params['n_estimators'],
                sample_weight=sample_weights,
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