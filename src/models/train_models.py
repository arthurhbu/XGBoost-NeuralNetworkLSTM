import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import optuna

def create_dynamic_triple_barrier_target(df, target_column, profit_multiplier, loss_multiplier, holding_days=7):
    """
    Cria um target ternário [0, 1, 2] usando o método Triple Barrier com
    barreiras dinâmicas baseadas no ATR (volatilidade).

    Esta é a implementação final que combina as abordagens avançadas discutidas.

    Args:
        df (pd.DataFrame): DataFrame de entrada. DEVE conter as colunas OHLC e 'ATR'.
        target_column_name (str): O nome da coluna de target a ser criada.
        holding_days (int): O número máximo de dias para manter a posição (barreira de tempo).
        profit_multiplier (float): Multiplicador do ATR para a barreira de lucro.
        loss_multiplier (float): Multiplicador do ATR para a barreira de perda.

    Returns:
        pd.DataFrame: O DataFrame original com a coluna de target adicionada e sem 
                    linhas que não puderam ser rotuladas.
    
    Mapeamento de Rótulos de Saída:
    - 0: A barreira de perda (Stop-Loss) foi atingida primeiro.
    - 1: A barreira de tempo (Timeout) foi atingida primeiro.
    - 2: A barreira de lucro (Take-Profit) foi atingida primeiro.
    
    """
    
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

            day_high = df['High'].iloc[i+j]
            day_low = df['Low'].iloc[i+j]

            if day_high > profit_barrier:
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


# def create_triple_barrier_target(df,target_column, holding_days=5, profit_threshold=0.01, loss_threshold=-0.009):
    
#     if loss_threshold > 1: 
#         loss_threshold = -loss_threshold

#     target = np.zeros(len(df), dtype=int)

#     for i in range(len(df) - holding_days):
            
#         entry_price = df['Open'].iloc[i]
#         profit_barrier = entry_price * (1 + profit_threshold)
#         loss_barrier = entry_price * (1 + loss_threshold)

#         for j in range(1, holding_days + 1):
#             if i + j >= len(df):
#                 break

#             day_high = df['High'].iloc[i+j]
#             day_low = df['Low'].iloc[i+j]

#             if day_high > profit_barrier:
#                 target[i] = 1
#                 break
            
#             elif day_low <= loss_barrier:
#                 target[i] = 0
#                 break

    
#     df[target_column] = target
#     return df

# def create_target_variable(df, target_column, holding_period=1 ,min_return_pct=0.0009):
#     """
#     Cria o target alinhado à execução na abertura do próximo dia, sem leakage.

#     Definição: target = 1 quando o retorno intradiário do próximo pregão
#     (Close(t+1)/Open(t+1) - 1) > min_return_pct; caso contrário 0.

#     Args:
#         df: DataFrame com dados OHLCV
#         target_column: Nome da coluna do alvo
#         holding_period: não utilizado aqui (mantido para compatibilidade)
#         min_return_pct: retorno mínimo líquido exigido no dia seguinte

#     Returns:
#         DataFrame com coluna target e sem NaN na última linha do alvo
#     """
#     next_open = df['Open'].shift(-1)
#     next_close = df['Close'].shift(-1)

#     next_day_intraday_return = (next_close / next_open) - 1
#     df[target_column] = (next_day_intraday_return > min_return_pct).astype(int)

#     df.dropna(subset=[target_column], inplace=True)

#     return df

def split_data(df, train_final_date, validation_start_date, validation_end_date, test_start_date, test_end_date, target_column_name):

    if isinstance(test_end_date, str):
        test_end_date = pd.to_datetime(test_end_date)   
    if isinstance(validation_start_date, str):
        validation_start_date = pd.to_datetime(validation_start_date)
    if isinstance(test_start_date, str):
        test_start_date = pd.to_datetime(test_start_date)
    if isinstance(validation_end_date, str):
        validation_end_date = pd.to_datetime(validation_end_date)
    if isinstance(train_final_date, str):
        train_final_date = pd.to_datetime(train_final_date)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    train_data = df[df.index <= train_final_date]
    val_data = df[(df.index >= validation_start_date) & (df.index < validation_end_date)]
    test_data = df[(df.index >= test_start_date) & (df.index <= test_end_date)]

    # Separar as features (x) do alvo (y)
    x_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]

    x_val = val_data.drop(columns=[target_column_name])
    y_val = val_data[target_column_name]
    
    x_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return x_train, y_train, x_val, y_val, x_test, y_test

    
def objective(trial, x_train, y_train, x_val, y_val, baseline):
    """
    Objetivo Optuna para XGBoost multiclasse com mlogloss.

    Teoria: Em janelas curtas, preferimos regularização e baixa variância.
    Otimizamos mlogloss; a seleção financeira (Sharpe) é feita depois via thresholds.
    """
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

    for ticker in os.listdir(feature_data_path):
        if ticker.endswith('.csv'):
            print(f"Processando {ticker}...")
            
            df = pd.read_csv(f'{feature_data_path}/{ticker}', index_col=0, parse_dates=True)
            
            # Usar parâmetros do triple barrier method do config
            triple_barrier_config = model_training_config.get("triple_barrier", {})
            holding_days = triple_barrier_config.get("holding_days", 7)
            profit_multiplier = triple_barrier_config.get("profit_multiplier", 2.0)
            loss_multiplier = triple_barrier_config.get("loss_multiplier", 1.5)
            
            df = create_dynamic_triple_barrier_target(
                df,
                model_training_config["target_column"],
                holding_days=holding_days,
                profit_multiplier=profit_multiplier,
                loss_multiplier=loss_multiplier,
            )


            x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                df, 
                model_training_config['train_final_date'], 
                model_training_config['validation_start_date'], 
                model_training_config['validation_end_date'],
                model_training_config['test_start_date'],
                model_training_config['test_end_date'],
                model_training_config['target_column']
                )

            # model = train_xgboost_model(x_train, y_train, x_val, y_val, model_training_config['xgboost_params'])

            baseline = 1.0

            # Otimização com o Optuna
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val, baseline), n_trials=50)

            best_params = study.best_params

            final_params = {**best_params}
            final_params['use_label_encoder'] = False
            final_params['objective'] = 'multi:softprob'
            final_params['num_class'] = 3
            final_params['eval_metric'] = 'mlogloss'

            x_train_full = pd.concat([x_train, x_val])
            y_train_full = pd.concat([y_train, y_val])

            # Usar API nativa para treinamento com early stopping
            dtrain_full = xgb.DMatrix(x_train_full, label=y_train_full)
            dval_final = xgb.DMatrix(x_val, label=y_val)
            
            # Treinar com early stopping
            booster = xgb.train(
                final_params,
                dtrain_full,
                num_boost_round=final_params['n_estimators'],
                evals=[(dval_final, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Converter para sklearn wrapper para compatibilidade
            final_model = xgb.XGBClassifier(**final_params)
            final_model._Booster = booster
            final_model._le = None  # Será definido automaticamente
            
            # Acurácia multiclasse
            train_proba = booster.predict(dtrain_full)
            train_pred = np.argmax(train_proba, axis=1)
            train_acc = accuracy_score(y_train_full, train_pred)
            print(f'Train accuracy (multi): {train_acc:.4f}')

            print("\n--- 3. Análise de Importância das Features ---")
            # Obter importância das features do booster
            importance_dict = booster.get_score(importance_type='weight')
            feature_names = x_test.columns.tolist()
            importance_values = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
            
            feature_importances = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)

            print("As 20 features mais importantes para o modelo:")

            # Plotar o gráfico
            plt.figure(figsize=(10, 8))
            plt.barh(feature_importances['feature'][:20], feature_importances['importance'][:20])
            plt.xlabel("Importância")
            plt.ylabel("Feature")
            plt.title("Importância das Features no Modelo XGBoost")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'reports/features_importance/{ticker}.png')
            plt.close()

            feature_importances.to_csv(f'reports/features_importance/{ticker}.csv', index=False)
            print(f"\nGráfico de importância das features salvo em: reports/features_importance/feature_importance_{ticker}.png")
        
            print("\n--- Avaliação no Conjunto de Teste ---")
            # Usar o booster para predições
            dtest = xgb.DMatrix(x_test)
            test_proba = booster.predict(dtest)
            predictions = np.argmax(test_proba, axis=1)
            
            unique_classes = np.unique(np.concatenate([y_test, predictions]))
            print(f"  Classes encontradas: {unique_classes}")
            print(f"  Distribuição y_test: {pd.Series(y_test).value_counts().to_dict()}")
            print(f"  Distribuição predictions: {pd.Series(predictions).value_counts().to_dict()}")
            
            print(classification_report(y_test, predictions, zero_division=0))

            model_path = Path(__file__).resolve().parents[2] / "models" / "01_xgboost" / f"{ticker.replace('.csv', '')}.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Salvar o booster (API nativa)
            booster.save_model(model_path)
            print('Modelo salvo em: ', model_path)

if __name__ == "__main__":
    main()
