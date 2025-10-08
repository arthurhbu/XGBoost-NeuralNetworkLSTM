#!/usr/bin/env python3
"""
Script para Extrair Colunas Essenciais
=====================================

Este script extrai as colunas essenciais (ATR, Open, High, Low, Close) 
dos arquivos de features originais antes que sejam processados pelo 
feature selection.

Uso:
    python extract_essential_columns.py
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.essential_columns_manager import main as extract_essential_columns


def main():
    """
    Função principal para extrair colunas essenciais.
    """
    print("🔧 EXTRAINDO COLUNAS ESSENCIAIS")
    print("=" * 50)
    print("Este script extrai as colunas necessárias para o train_models:")
    print("- ATR: Para triple barrier method dinâmico")
    print("- Open: Para preço de entrada")
    print("- High: Para verificar barreira de lucro")
    print("- Low: Para verificar barreira de perda")
    print("- Close: Para cálculo do portfólio")
    print("=" * 50)
    
    try:
        extract_essential_columns()
        print("\n✅ Extração de colunas essenciais concluída com sucesso!")
        print("📁 Colunas essenciais salvas em: data/04_essential_columns/")
        
    except Exception as e:
        print(f"\n❌ Erro durante extração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
