#!/usr/bin/env python3
"""
Script de teste para verificar se as correções do desbalanceamento funcionam
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import xgboost as xgb
from src.models.train_models import create_FIXED_triple_barrier_target, split_data, apply_aggressive_class_balancing

def test_itub4_fixes():
    """Testa as correções implementadas para o ITUB4.SA"""
    
    print("="*60)
    print("TESTE DAS CORREÇÕES - ITUB4.SA")
    print("="*60)
    
    # Carregar configuração
    config_path = Path("config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Carregar dados
    features_path = Path("data/03_features/ITUB4.SA.csv")
    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    print(f"\n1. DADOS CARREGADOS:")
    print(f"   Total de registros: {len(df_features)}")
    print(f"   Período: {df_features.index[0].date()} até {df_features.index[-1].date()}")
    
    # Testar com parâmetros mais conservadores
    test_params = {
        "holding_days": 7, 
        "profit_threshold": 0.02, 
        "loss_threshold": -0.01
    }
    
    print(f"\n2. TESTANDO CRIAÇÃO DE TARGET:")
    print(f"   Parâmetros: {test_params}")
    
    try:
        df_with_target = create_FIXED_triple_barrier_target(
            df_features.copy(), 
            "target", 
            **test_params
        )
        
        print(f"   ✅ Target criado com sucesso")
        print(f"   Registros com target: {len(df_with_target)}")
        
        # Verificar distribuição de classes
        class_dist = df_with_target['target'].value_counts(normalize=True).sort_index()
        print(f"   Distribuição de classes: {dict(class_dist)}")
        
        # Verificar se há dados suficientes para split
        if len(df_with_target) > 100:
            print(f"\n3. TESTANDO SPLIT DOS DADOS:")
            
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(
                df_with_target, 
                config['model_training']['train_final_date'], 
                config['model_training']['validation_start_date'], 
                config['model_training']['validation_end_date'], 
                config['model_training']['test_start_date'], 
                config['model_training']['test_end_date'], 
                target_column_name="target"
            )
            
            print(f"   ✅ Split realizado com sucesso")
            print(f"   Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
            
            # Verificar distribuição no treino
            train_dist = y_train.value_counts(normalize=True).sort_index()
            print(f"   Distribuição treino: {dict(train_dist)}")
            
            # Testar balanceamento agressivo se necessário
            up_ratio = train_dist.get(2, 0.0)
            if up_ratio < 0.05:
                print(f"\n4. APLICANDO BALANCEAMENTO AGRESSIVO:")
                print(f"   Classe Up muito rara ({up_ratio:.3f}), aplicando balanceamento...")
                
                x_train_balanced, y_train_balanced = apply_aggressive_class_balancing(x_train, y_train)
                
                # Verificar distribuição após balanceamento
                balanced_dist = y_train_balanced.value_counts(normalize=True).sort_index()
                print(f"   Distribuição após balanceamento: {dict(balanced_dist)}")
                
                # Testar treinamento de modelo simples
                print(f"\n5. TESTANDO TREINAMENTO DE MODELO:")
                
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model.fit(x_train_balanced, y_train_balanced)
                
                # Fazer predições
                val_proba = model.predict_proba(x_val)
                val_pred = model.predict(x_val)
                
                print(f"   ✅ Modelo treinado com sucesso")
                print(f"   Probabilidades shape: {val_proba.shape}")
                print(f"   Predições shape: {val_pred.shape}")
                
                # Verificar distribuição de predições
                pred_dist = pd.Series(val_pred).value_counts().sort_index()
                print(f"   Distribuição de predições: {dict(pred_dist)}")
                
                # Verificar se há predições da classe "up" (2)
                up_predictions = np.sum(val_pred == 2)
                print(f"   Predições 'up' (classe 2): {up_predictions}/{len(val_pred)} ({up_predictions/len(val_pred):.2%})")
                
                # Verificar distribuição de probabilidades
                prob_stats = {
                    'P(down)': [val_proba[:, 0].min(), val_proba[:, 0].mean(), val_proba[:, 0].max()],
                    'P(flat)': [val_proba[:, 1].min(), val_proba[:, 1].mean(), val_proba[:, 1].max()],
                    'P(up)': [val_proba[:, 2].min(), val_proba[:, 2].mean(), val_proba[:, 2].max()]
                }
                
                print(f"   Estatísticas das probabilidades:")
                for class_name, stats in prob_stats.items():
                    print(f"     {class_name}: min={stats[0]:.4f}, mean={stats[1]:.4f}, max={stats[2]:.4f}")
                
                # Calcular score P(up) - P(down)
                score = val_proba[:, 2] - val_proba[:, 0]
                print(f"   Score P(up) - P(down):")
                print(f"     Min: {score.min():.4f}, Mean: {score.mean():.4f}, Max: {score.max():.4f}")
                print(f"     Scores positivos: {np.sum(score > 0)}/{len(score)} ({np.sum(score > 0)/len(score):.2%})")
                
                if up_predictions > 0:
                    print(f"\n   ✅ SUCESSO: Modelo agora está predizendo a classe 'Up'!")
                else:
                    print(f"\n   ⚠️  AVISO: Modelo ainda não está predizendo a classe 'Up'")
                    
            else:
                print(f"\n4. BALANCEAMENTO NÃO NECESSÁRIO:")
                print(f"   Classe Up tem proporção adequada ({up_ratio:.3f})")
        else:
            print(f"   ❌ ERRO: Poucos dados para split ({len(df_with_target)})")
            
    except Exception as e:
        print(f"   ❌ ERRO: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("FIM DO TESTE")
    print("="*60)

if __name__ == "__main__":
    test_itub4_fixes()
