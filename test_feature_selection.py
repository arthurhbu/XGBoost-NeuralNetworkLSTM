#!/usr/bin/env python3
"""
Teste da Seleção de Features
============================

Script para testar se a seleção de features está funcionando corretamente.
"""

import pandas as pd
from pathlib import Path

def test_feature_selection():
    """Testa se a seleção de features foi aplicada corretamente."""
    
    print("🔍 TESTANDO SELEÇÃO DE FEATURES")
    print("=" * 40)
    
    # Diretórios
    original_dir = Path("data/03_features")
    selected_dir = Path("data/03_features_selected")
    
    # Verificar se os diretórios existem
    if not original_dir.exists():
        print("❌ Diretório original não encontrado!")
        return
    
    if not selected_dir.exists():
        print("❌ Diretório de features selecionadas não encontrado!")
        return
    
    # Listar arquivos
    original_files = list(original_dir.glob("*.csv"))
    selected_files = list(selected_dir.glob("*.csv"))
    
    print(f"📁 Arquivos originais: {len(original_files)}")
    print(f"📁 Arquivos selecionados: {len(selected_files)}")
    
    # Testar um ticker específico
    ticker = "PETR4.SA"
    original_file = original_dir / f"{ticker}.csv"
    selected_file = selected_dir / f"{ticker}.csv"
    
    if not original_file.exists() or not selected_file.exists():
        print(f"❌ Arquivos para {ticker} não encontrados!")
        return
    
    # Carregar dados
    df_original = pd.read_csv(original_file, index_col=0, parse_dates=True)
    df_selected = pd.read_csv(selected_file, index_col=0, parse_dates=True)
    
    print(f"\n📊 ANÁLISE DO TICKER: {ticker}")
    print("-" * 30)
    print(f"Features originais: {len(df_original.columns)}")
    print(f"Features selecionadas: {len(df_selected.columns)}")
    print(f"Redução: {len(df_original.columns) - len(df_selected.columns)} features")
    print(f"Redução %: {((len(df_original.columns) - len(df_selected.columns)) / len(df_original.columns)) * 100:.1f}%")
    
    # Mostrar features removidas
    original_features = set(df_original.columns)
    selected_features = set(df_selected.columns)
    removed_features = original_features - selected_features
    
    print(f"\n🗑️  FEATURES REMOVIDAS:")
    for feature in sorted(removed_features):
        print(f"  - {feature}")
    
    # Mostrar features mantidas
    print(f"\n✅ FEATURES MANTIDAS:")
    for feature in sorted(selected_features):
        print(f"  - {feature}")
    
    # Verificar se as features esperadas foram removidas
    expected_removed = ['Volume', 'MACD_signal', 'MACD_hist', 'ADX', 'MFI']
    actually_removed = list(removed_features)
    
    print(f"\n🎯 VERIFICAÇÃO:")
    for feature in expected_removed:
        if feature in actually_removed:
            print(f"  ✅ {feature} - Removida corretamente")
        else:
            print(f"  ❌ {feature} - NÃO foi removida!")
    
    # Verificar se as features importantes foram mantidas
    expected_kept = ['wavelet_cD', 'wavelet_cA', 'EMA_long', 'EMA_short', 'BB_upper', 'BB_lower', 'BB_middle']
    print(f"\n🔍 VERIFICAÇÃO DE FEATURES IMPORTANTES:")
    for feature in expected_kept:
        if feature in selected_features:
            print(f"  ✅ {feature} - Mantida corretamente")
        else:
            print(f"  ❌ {feature} - NÃO foi mantida!")
    
    print(f"\n✅ Teste concluído!")

if __name__ == "__main__":
    test_feature_selection()
