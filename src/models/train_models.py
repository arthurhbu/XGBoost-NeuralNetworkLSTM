import os
from pathlib import Path
import yaml
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report

def create_target_variable(df, target_column):

    df[target_column] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(subset=[target_column], inplace=True)

    return df

def split_data(df, validation_start_date, test_start_date, train_final_date, target_column_name):

    if isinstance(validation_start_date, str):
        validation_start_date = pd.to_datetime(validation_start_date)
    if isinstance(test_start_date, str):
        test_start_date = pd.to_datetime(test_start_date)
    if isinstance(train_final_date, str):
        train_final_date = pd.to_datetime(train_final_date)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    train_data = df[df.index <= train_final_date]
    val_data = df[(df.index >= validation_start_date) & (df.index < test_start_date)]
    test_data = df[(df.index >= test_start_date)]

    # Separar as features (x) do alvo (y)
    x_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]

    x_val = val_data.drop(columns=[target_column_name])
    y_val = val_data[target_column_name]
    
    x_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return x_train, y_train, x_val, y_val, x_test, y_test

def train_xgboost_model(x_train, y_train, x_val, y_val, params):

    print('Iniciando treinamento do modelo XGBoost...')

    xgb_model = xgb.XGBClassifier(**params, use_label_encoder=False)

    xgb_model.fit(
        x_train, 
        y_train, 
        eval_set=[(x_val, y_val)], 
        verbose=True
        )

    print('Modelo treinado com sucesso!')

    return xgb_model

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
            
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                df, 
                model_training_config['validation_start_date'], 
                model_training_config['test_start_date'],
                model_training_config['train_final_date'],
                model_training_config['target_column']
                )

            model = train_xgboost_model(x_train, y_train, x_val, y_val, model_training_config['xgboost_params'])

            print("\n--- Avaliação no Conjunto de Teste ---")
            predictions = model.predict(x_test)
            print(classification_report(y_test, predictions, target_names=['Baixa/Estável', 'Alta']))

            model_path = Path(__file__).resolve().parents[2] / "models" / "01_xgboost" / f"{ticker.replace('.csv', '')}.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)

            model.save_model(model_path)

            print('Modelo salvo em: ', model_path)

if __name__ == "__main__":
    main()
