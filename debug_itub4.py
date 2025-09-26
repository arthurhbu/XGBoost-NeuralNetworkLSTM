#!/usr/bin/env python3
"""
Script de diagnóstico para o problema de desbalanceamento no ITUB4.SA
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import xgboost as xgb
from src.models.train_models import create_FIXED_triple_barrier_target, split_data, calculate_class_weights

def diagnose_itub4():
    """Diagnóstico completo do problema no ITUB4.SA"""
    
    # Carregar configuração
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Carregar dados
    features_path = Path("data/03_features/ITUB4.SA.csv")
    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    print("="*60)
    print("DIAGNÓSTICO ITUB4.SA - PROBLEMA DE DESBALANCEAMENTO")
    print("="*60)
    
    print(f"\n1. DADOS CARREGADOS:")
    print(f"   Total de registros: {len(df_features)}")
    print(f"   Período: {df_features.index[0].date()} até {df_features.index[-1].date()}")
    print(f"   Colunas disponíveis: {list(df_features.columns)}")
    
    # Testar diferentes parâmetros de triple barrier
    print(f"\n2. TESTANDO DIFERENTES PARÂMETROS DE TRIPLE BARRIER:")
    
    test_params = [
        {"holding_days": 7, "profit_threshold": 0.02, "loss_threshold": -0.01},
        {"holding_days": 7, "profit_threshold": 0.03, "loss_threshold": -0.015},
        {"holding_days": 7, "profit_threshold": 0.05, "loss_threshold": -0.025},
        {"holding_days": 5, "profit_threshold": 0.05, "loss_threshold": -0.025},
        {"holding_days": 10, "profit_threshold": 0.04, "loss_threshold": -0.02},
    ]
    
    for i, params in enumerate(test_params):
        print(f"\n   Teste {i+1}: {params}")
        try:
            df_test = create_FIXED_triple_barrier_target(
                df_features.copy(), 
                "target", 
                **params
            )
            
            # Verificar distribuição de classes
            class_dist = df_test['target'].value_counts(normalize=True).sort_index()
            print(f"      Classes: {dict(class_dist)}")
            print(f"      Total com target: {len(df_test)}")
            
            # Verificar se há dados suficientes para split
            if len(df_test) > 100:
                x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                    df_test, 
                    config['model_training']['train_final_date'], 
                    config['model_training']['validation_start_date'], 
                    config['model_training']['validation_end_date'], 
                    config['model_training']['test_start_date'], 
                    config['model_training']['test_end_date'], 
                    target_column_name="target"
                )
                
                print(f"      Split - Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
                
                if len(y_train) > 0:
                    train_dist = y_train.value_counts(normalize=True).sort_index()
                    print(f"      Distribuição treino: {dict(train_dist)}")
                    
                    # Calcular pesos das classes
                    class_weights = calculate_class_weights(y_train)
                    print(f"      Pesos das classes: {class_weights}")
            else:
                print(f"      AVISO: Poucos dados para split ({len(df_test)})")
                
        except Exception as e:
            print(f"      ERRO: {str(e)}")
    
    # Verificar se o modelo existe e carregar
    print(f"\n3. VERIFICANDO MODELO SALVO:")
    model_path = Path("models/01_xgboost/ITUB4.SA.json")
    if model_path.exists():
        print(f"   Modelo encontrado: {model_path}")
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            print(f"   Modelo carregado com sucesso")
            
            # Testar predições em dados de validação
            print(f"\n4. TESTANDO PREDIÇÕES DO MODELO:")
            
            # Usar o melhor parâmetro encontrado
            best_params = {"holding_days": 7, "profit_threshold": 0.03, "loss_threshold": -0.015}
            df_final = create_FIXED_triple_barrier_target(
                df_features.copy(), 
                "target", 
                **best_params
            )
            
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                df_final, 
                config['model_training']['train_final_date'], 
                config['model_training']['validation_start_date'], 
                config['model_training']['validation_end_date'], 
                config['model_training']['test_start_date'], 
                config['model_training']['test_end_date'], 
                target_column_name="target"
            )
            
            # Fazer predições
            probabilities = model.predict_proba(x_val)
            predictions = model.predict(x_val)
            
            print(f"   Probabilidades shape: {probabilities.shape}")
            print(f"   Predições shape: {predictions.shape}")
            
            # Analisar distribuição de probabilidades
            prob_stats = {
                'P(down)': [probabilities[:, 0].min(), probabilities[:, 0].mean(), probabilities[:, 0].max()],
                'P(flat)': [probabilities[:, 1].min(), probabilities[:, 1].mean(), probabilities[:, 1].max()],
                'P(up)': [probabilities[:, 2].min(), probabilities[:, 2].mean(), probabilities[:, 2].max()]
            }
            
            print(f"\n   Estatísticas das probabilidades:")
            for class_name, stats in prob_stats.items():
                print(f"     {class_name}: min={stats[0]:.4f}, mean={stats[1]:.4f}, max={stats[2]:.4f}")
            
            # Calcular score P(up) - P(down)
            score = probabilities[:, 2] - probabilities[:, 0]
            print(f"\n   Score P(up) - P(down):")
            print(f"     Min: {score.min():.4f}")
            print(f"     Mean: {score.mean():.4f}")
            print(f"     Max: {score.max():.4f}")
            print(f"     Std: {score.std():.4f}")
            
            # Verificar quantos scores são positivos (indicando compra)
            positive_scores = np.sum(score > 0)
            print(f"     Scores positivos (compra): {positive_scores}/{len(score)} ({positive_scores/len(score):.2%})")
            
            # Verificar distribuição de predições
            pred_dist = pd.Series(predictions).value_counts().sort_index()
            print(f"\n   Distribuição de predições: {dict(pred_dist)}")
            
            # Verificar se há predições da classe "up" (2)
            up_predictions = np.sum(predictions == 2)
            print(f"   Predições 'up' (classe 2): {up_predictions}/{len(predictions)} ({up_predictions/len(predictions):.2%})")
            
        except Exception as e:
            print(f"   ERRO ao carregar modelo: {str(e)}")
    else:
        print(f"   Modelo não encontrado: {model_path}")
    
    print(f"\n" + "="*60)
    print("FIM DO DIAGNÓSTICO")
    print("="*60)

if __name__ == "__main__":
    diagnose_itub4()
