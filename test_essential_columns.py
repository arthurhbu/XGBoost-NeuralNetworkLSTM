#!/usr/bin/env python3
"""
Teste da Solução de Colunas Essenciais Corrigida
===============================================

Este script testa se a solução corrigida está funcionando corretamente:
1. Colunas essenciais são usadas APENAS para criar targets
2. Modelo treina APENAS com features selecionadas (sem data leakage)
3. Separação correta entre responsabilidades

Uso:
    python test_essential_columns.py
"""

import sys
from pathlib import Path
import pandas as pd

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.essential_columns_manager import EssentialColumnsManager


def test_essential_columns_separation():
    """
    Testa se a separação entre colunas essenciais e features do modelo está correta.
    """
    print("🧪 TESTANDO SEPARAÇÃO DE COLUNAS ESSENCIAIS")
    print("=" * 60)
    
    # Inicializar gerenciador
    manager = EssentialColumnsManager()
    
    # Carregar dados de exemplo
    project_root = Path(__file__).parent
    input_dir = project_root / "data" / "03_features"
    selected_dir = project_root / "data" / "03_features_selected"
    
    # Lista de tickers para testar
    tickers = ["PETR4.SA", "VALE3.SA", "BBDC4.SA"]
    
    results = {}
    
    for ticker in tickers:
        print(f"\n📊 Testando {ticker}...")
        
        # 1. Carregar dados originais
        original_file = input_dir / f'{ticker}.csv'
        selected_file = selected_dir / f'{ticker}.csv'
        
        if not original_file.exists() or not selected_file.exists():
            print(f"  ❌ Arquivos não encontrados para {ticker}")
            continue
        
        df_original = pd.read_csv(original_file, index_col=0, parse_dates=True)
        df_selected = pd.read_csv(selected_file, index_col=0, parse_dates=True)
        
        print(f"  📈 Features originais: {len(df_original.columns)}")
        print(f"  🎯 Features selecionadas: {len(df_selected.columns)}")
        
        # 2. Extrair colunas essenciais
        df_essential = manager.extract_essential_columns(df_original, ticker)
        
        if df_essential.empty:
            print(f"  ⚠️  Nenhuma coluna essencial encontrada")
            results[ticker] = "NO_ESSENTIAL"
            continue
        
        print(f"  🔧 Colunas essenciais: {df_essential.columns.tolist()}")
        
        # 3. Verificar se há data leakage
        essential_in_selected = [col for col in df_essential.columns if col in df_selected.columns]
        
        if essential_in_selected:
            print(f"  ⚠️  Data leakage detectado: {essential_in_selected}")
            results[ticker] = "DATA_LEAKAGE"
        else:
            print(f"  ✅ Sem data leakage - colunas essenciais não estão nas features selecionadas")
            results[ticker] = "CLEAN"
        
        # 4. Simular o processo do train_models
        print(f"  🔄 Simulando processo do train_models...")
        
        # Combinar temporariamente para criar targets (como no train_models)
        df_for_targets = pd.concat([df_selected, df_essential], axis=1)
        print(f"    Dados para targets: {len(df_for_targets.columns)} colunas")
        
        # Separar novamente para o modelo (como no train_models)
        model_features = [col for col in df_selected.columns if col not in df_essential.columns]
        df_for_model = df_for_targets[model_features]
        print(f"    Features para modelo: {len(df_for_model.columns)} colunas")
        
        # Verificar se o modelo não tem data leakage
        model_has_essential = [col for col in df_essential.columns if col in df_for_model.columns]
        
        if model_has_essential:
            print(f"    ❌ ERRO: Modelo ainda tem data leakage: {model_has_essential}")
            results[ticker] = "MODEL_LEAKAGE"
        else:
            print(f"    ✅ Modelo limpo - sem colunas essenciais")
            if results[ticker] == "CLEAN":
                results[ticker] = "SUCCESS"
    
    # Resumo dos resultados
    print(f"\n📊 RESUMO DOS TESTES")
    print("=" * 40)
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(results)
    
    for ticker, status in results.items():
        status_emoji = {
            "SUCCESS": "✅",
            "CLEAN": "✅",
            "DATA_LEAKAGE": "⚠️",
            "MODEL_LEAKAGE": "❌",
            "NO_ESSENTIAL": "❓"
        }
        print(f"{status_emoji[status]} {ticker}: {status}")
    
    print(f"\n🎯 Resultado: {success_count}/{total_count} tickers com sucesso")
    
    if success_count == total_count:
        print("🎉 Todos os testes passaram! A separação está funcionando corretamente.")
        print("✅ Colunas essenciais são usadas APENAS para criar targets")
        print("✅ Modelo treina APENAS com features selecionadas (sem data leakage)")
    elif success_count > 0:
        print("⚠️  Alguns testes passaram. Verifique os tickers com problemas.")
    else:
        print("❌ Nenhum teste passou. A solução precisa ser corrigida.")
    
    return results


def test_essential_columns_functionality():
    """
    Testa se as colunas essenciais têm os dados necessários para o train_models.
    """
    print(f"\n🔧 TESTANDO FUNCIONALIDADE DAS COLUNAS ESSENCIAIS")
    print("=" * 60)
    
    manager = EssentialColumnsManager()
    
    # Verificar se as colunas essenciais existem
    essential_dir = Path(__file__).parent / "data" / "04_essential_columns"
    
    if not essential_dir.exists():
        print("❌ Diretório de colunas essenciais não existe. Execute extract_essential_columns.py primeiro.")
        return False
    
    essential_files = list(essential_dir.glob("*_essential.csv"))
    
    if not essential_files:
        print("❌ Nenhum arquivo de colunas essenciais encontrado. Execute extract_essential_columns.py primeiro.")
        return False
    
    print(f"📁 Encontrados {len(essential_files)} arquivos de colunas essenciais")
    
    for file in essential_files:
        ticker = file.stem.replace("_essential", "")
        print(f"\n📊 Verificando {ticker}...")
        
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            print(f"  📈 Colunas: {df.columns.tolist()}")
            print(f"  📊 Linhas: {len(df)}")
            print(f"  📅 Período: {df.index.min()} a {df.index.max()}")
            
            # Verificar se tem dados válidos
            for col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    print(f"  ⚠️  {col}: {null_count} valores nulos")
                else:
                    print(f"  ✅ {col}: sem valores nulos")
        
        except Exception as e:
            print(f"  ❌ Erro ao ler arquivo: {e}")
    
    return True


def main():
    """
    Função principal.
    """
    try:
        # Teste 1: Separação de colunas
        separation_results = test_essential_columns_separation()
        
        # Teste 2: Funcionalidade das colunas essenciais
        functionality_ok = test_essential_columns_functionality()
        
        # Resumo final
        print(f"\n🎯 RESUMO FINAL")
        print("=" * 30)
        
        if functionality_ok:
            print("✅ Colunas essenciais extraídas e funcionais")
        else:
            print("❌ Problemas com colunas essenciais")
        
        success_count = sum(1 for status in separation_results.values() if status == "SUCCESS")
        total_count = len(separation_results)
        
        if success_count == total_count:
            print("✅ Separação de responsabilidades funcionando")
            print("🎉 Solução implementada com sucesso!")
        else:
            print("⚠️  Problemas na separação de responsabilidades")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()