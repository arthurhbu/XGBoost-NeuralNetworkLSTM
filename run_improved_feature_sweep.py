#!/usr/bin/env python3
"""
Script para executar o Feature Sweep Melhorado
==============================================

Este script executa a versão corrigida do feature sweep e compara
com os resultados originais para demonstrar as melhorias.

Uso:
    python run_improved_feature_sweep.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.analysis.feature_importance_sweep_improved import run_improved_sweep


def compare_results():
    """
    Compara resultados originais vs melhorados.
    """
    project_root = Path(__file__).parent
    
    # Carregar resultados originais
    original_summary = project_root / 'reports' / 'feature_sweep' / 'sweep_summary.csv'
    improved_summary = project_root / 'reports' / 'feature_sweep_improved' / 'sweep_summary_improved.csv'
    
    if not original_summary.exists():
        print("❌ Resultados originais não encontrados!")
        return
    
    if not improved_summary.exists():
        print("❌ Resultados melhorados não encontrados!")
        return
    
    # Carregar dados
    df_orig = pd.read_csv(original_summary)
    df_improved = pd.read_csv(improved_summary)
    
    # Comparar
    print("\n📊 COMPARAÇÃO: ORIGINAL vs MELHORADO")
    print("=" * 80)
    print(f"{'Ticker':<10} {'Original K':<12} {'Original Sharpe':<15} {'Melhorado K':<12} {'Melhorado Sharpe':<15} {'Melhoria':<10}")
    print("-" * 80)
    
    for ticker in df_orig['ticker'].unique():
        orig_row = df_orig[df_orig['ticker'] == ticker].iloc[0]
        improved_row = df_improved[df_improved['ticker'] == ticker].iloc[0]
        
        orig_sharpe = orig_row['sharpe_val']
        improved_sharpe = improved_row['adjusted_sharpe']
        improvement = ((improved_sharpe - orig_sharpe) / abs(orig_sharpe)) * 100 if orig_sharpe != 0 else 0
        
        print(f"{ticker:<10} {orig_row['k']:<12} {orig_sharpe:<15.4f} {improved_row['k']:<12} {improved_sharpe:<15.4f} {improvement:>+8.1f}%")
    
    print("=" * 80)
    
    # Estatísticas gerais
    avg_orig = df_orig['sharpe_val'].mean()
    avg_improved = df_improved['adjusted_sharpe'].mean()
    overall_improvement = ((avg_improved - avg_orig) / abs(avg_orig)) * 100 if avg_orig != 0 else 0
    
    print(f"\n📈 ESTATÍSTICAS GERAIS:")
    print(f"Sharpe médio original: {avg_orig:.4f}")
    print(f"Sharpe médio melhorado: {avg_improved:.4f}")
    print(f"Melhoria geral: {overall_improvement:+.1f}%")
    
    # Análise de estabilidade
    orig_std = df_orig['sharpe_val'].std()
    improved_std = df_improved['adjusted_sharpe'].std()
    
    print(f"\n📊 ESTABILIDADE:")
    print(f"Desvio padrão original: {orig_std:.4f}")
    print(f"Desvio padrão melhorado: {improved_std:.4f}")
    print(f"Melhoria na estabilidade: {((orig_std - improved_std) / orig_std) * 100:+.1f}%")


def main():
    """
    Função principal.
    """
    print("🚀 EXECUTANDO FEATURE SWEEP MELHORADO")
    print("=" * 60)
    
    try:
        # Executar sweep melhorado
        run_improved_sweep()
        
        # Comparar resultados
        compare_results()
        
        print("\n✅ Feature sweep melhorado concluído com sucesso!")
        print("📁 Verifique os resultados em: reports/feature_sweep_improved/")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
