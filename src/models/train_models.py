import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, average_precision_score
import matplotlib.pyplot as plt
import optuna

def create_target_variable(df, target_column, holding_period=1 ,min_return_pct=0.0009):
    """
    Cria o target alinhado à execução na abertura do próximo dia, sem leakage.

    Definição: target = 1 quando o retorno intradiário do próximo pregão
    (Close(t+1)/Open(t+1) - 1) > min_return_pct; caso contrário 0.

    Args:
        df: DataFrame com dados OHLCV
        target_column: Nome da coluna do alvo
        holding_period: não utilizado aqui (mantido para compatibilidade)
        min_return_pct: retorno mínimo líquido exigido no dia seguinte

    Returns:
        DataFrame com coluna target e sem NaN na última linha do alvo
    """
    next_open = df['Open'].shift(-1)
    next_close = df['Close'].shift(-1)

    next_day_intraday_return = (next_close / next_open) - 1
    df[target_column] = (next_day_intraday_return > min_return_pct).astype(int)

    df.dropna(subset=[target_column], inplace=True)

    return df

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

    scale_pos_weight = trial.suggest_float('scale_pos_weight', max(0.25 * baseline, 0.1),
    max(4.0 * baseline, 1.0), log=True)


    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'use_label_encoder': False,
        'scale_pos_weight': scale_pos_weight,
        'early_stopping_rounds': 50,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False
    )

    proba = model.predict_proba(x_val)[:, 1]
    score = average_precision_score(y_val, proba)
    return score

# def train_xgboost_model(x_train, y_train, x_val, y_val, params):

#     print('Iniciando treinamento do modelo XGBoost...')
    
#     # Calcular peso para balancear classes
#     class_counts = y_train.value_counts()
#     scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
#     params['scale_pos_weight'] = scale_pos_weight
#     print(f'  Distribuição das classes: {dict(class_counts)}')
#     print(f'  ⚖️  Scale pos weight: {scale_pos_weight:.2f}')
    
#     # Adicionar scale_pos_weight aos parâmetros
#     params_with_balance = params.copy()
#     params_with_balance['scale_pos_weight'] = scale_pos_weight

#     xgb_model = xgb.XGBClassifier(**params_with_balance, use_label_encoder=False)

#     xgb_model.fit(
#         x_train, 
#         y_train, 
#         eval_set=[(x_val, y_val)], 
#         verbose=True
#         )

#     print('Modelo treinado com sucesso!')

#     return xgb_model

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
            
            df = create_target_variable(df, model_training_config["target_column"])
            
            target_dist = df[model_training_config["target_column"]].value_counts()
            print(f"  Distribuição do target: {dict(target_dist)}")
            print(f"  Percentual de alta: {target_dist[1]/len(df)*100:.1f}%")
            
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

            neg = int(np.sum(y_train == 0))
            pos = int(np.sum(y_train == 1))
            baseline = (neg / pos) if pos > 0 else 1.0

            # Otimização com o Optuna
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: objective(trial, x_train, y_train, x_val, y_val, baseline), n_trials=50)

            best_params = study.best_params

            final_params = {**best_params}
            final_params['use_label_encoder'] = False
            final_params['objective'] = 'binary:logistic'
            final_params['eval_metric'] = 'aucpr'

            x_train_full = pd.concat([x_train, x_val])
            y_train_full = pd.concat([y_train, y_val])

            final_model = xgb.XGBClassifier(**final_params)
            final_model.fit(
                x_train_full,
                y_train_full,
                verbose=False
            )

            print("\n--- 3. Análise de Importância das Features ---")
            feature_importances = pd.DataFrame({
                'feature': x_test.columns,
                'importance': final_model.feature_importances_
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
            predictions = final_model.predict(x_test)
            
            unique_classes = np.unique(np.concatenate([y_test, predictions]))
            print(f"  Classes encontradas: {unique_classes}")
            print(f"  Distribuição y_test: {pd.Series(y_test).value_counts().to_dict()}")
            print(f"  Distribuição predictions: {pd.Series(predictions).value_counts().to_dict()}")
            
            if len(unique_classes) == 2:
                print(classification_report(y_test, predictions, target_names=['Baixa/Estável', 'Alta']))
            else:
                print("  AVISO: Apenas uma classe encontrada - não é possível gerar relatório completo")
                print(f"  Acurácia: {accuracy_score(y_test, predictions):.2%}")

            model_path = Path(__file__).resolve().parents[2] / "models" / "01_xgboost" / f"{ticker.replace('.csv', '')}.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            final_model.save_model(model_path)
            print('Modelo salvo em: ', model_path)

if __name__ == "__main__":
    main()
