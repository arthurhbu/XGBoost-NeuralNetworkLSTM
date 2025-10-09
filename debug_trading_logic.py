#!/usr/bin/env python3
"""
Script para debugar por que não há trades sendo gerados
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def debug_ticker(ticker):
    print(f"\n🔍 DEBUGGING: {ticker}")
    print("=" * 50)
    
    # Carregar dados
    features_path = f"data/03_features_selected/{ticker}.csv"
    labeled_path = f"data/04_labeled/{ticker}.csv"
    
    try:
        features_df = pd.read_csv(features_path)
        labeled_df = pd.read_csv(labeled_path)
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return
    
    print(f"📊 Features shape: {features_df.shape}")
    print(f"📊 Labeled shape: {labeled_df.shape}")
    
    # Verificar features selecionadas
    print(f"\n🔧 FEATURES SELECIONADAS:")
    print(features_df.columns.tolist())
    
    # Verificar targets
    print(f"\n🎯 DISTRIBUIÇÃO DE TARGETS:")
    target_counts = labeled_df['target'].value_counts()
    print(target_counts)
    
    # Simular geração de scores (como no backtest)
    print(f"\n🎲 SIMULAÇÃO DE SCORES:")
    
    # Usar a primeira feature numérica para simular score
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        feature_col = numeric_cols[0]
        feature_values = features_df[feature_col].values
        
        # Normalizar feature para 0-1
        feature_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
        
        # Simular P(up) e P(down) baseado na feature
        p_up = 0.3 + 0.4 * feature_norm  # 0.3 a 0.7
        p_down = 0.3 - 0.2 * feature_norm  # 0.1 a 0.5
        p_flat = 1 - p_up - p_down
        p_flat = np.clip(p_flat, 0.1, 0.9)
        
        # Normalizar
        total_prob = p_up + p_down + p_flat
        p_up = p_up / total_prob
        p_down = p_down / total_prob
        p_flat = p_flat / total_prob
        
        # Calcular score s = P(up) - P(down)
        score = p_up - p_down
        
        print(f"Usando feature: {feature_col}")
        print(f"Score médio: {score.mean():.4f}")
        print(f"Score std: {score.std():.4f}")
        print(f"Score min: {score.min():.4f}")
        print(f"Score max: {score.max():.4f}")
        
        # Testar thresholds
        print(f"\n🎯 TESTE DE THRESHOLDS:")
        for buy_thresh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for sell_thresh in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2]:
                buy_signals = (score > buy_thresh).sum()
                sell_signals = (score < sell_thresh).sum()
                if buy_signals > 0 or sell_signals > 0:
                    print(f"  Buy>{buy_thresh:.1f}, Sell<{sell_thresh:.1f}: {buy_signals} compras, {sell_signals} vendas")
        
        # Verificar se há valores NaN ou infinitos
        print(f"\n⚠️ VERIFICAÇÃO DE DADOS:")
        print(f"Valores NaN em score: {np.isnan(score).sum()}")
        print(f"Valores infinitos em score: {np.isinf(score).sum()}")
        print(f"Valores NaN em features: {features_df.isnull().sum().sum()}")
        print(f"Valores infinitos em features: {np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()}")

def main():
    print("🔍 DEBUGGING TRADING LOGIC")
    print("=" * 50)
    
    # Debug tickers problemáticos
    problem_tickers = ["ITUB4.SA", "VIVT3.SA", "BBAS3.SA"]
    
    for ticker in problem_tickers:
        debug_ticker(ticker)

if __name__ == "__main__":
    main()
