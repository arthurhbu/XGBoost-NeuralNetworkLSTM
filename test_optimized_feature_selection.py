#!/usr/bin/env python3
"""
Teste da Seleção de Features Otimizada
======================================

Este script testa se a integração das melhores features do sweep
está funcionando corretamente no sistema de feature selection.

Uso:
    python test_optimized_feature_selection.py
"""

import sys
from pathlib import Path
import pandas as pd

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.features.feature_selection import GlobalFeatureSelector


def test_optimized_feature_selection():
    """
    Testa a seleção de features otimizada para cada ticker.
    """
    print("🧪 TESTANDO SELEÇÃO DE FEATURES OTIMIZADA")
    print("=" * 60)
    
    # Inicializar seletor
    selector = GlobalFeatureSelector()
    
    # Carregar dados de exemplo
    project_root = Path(__file__).parent
    input_dir = project_root / "data" / "03_features"
    
    # Lista de tickers para testar
    tickers = ["PETR4.SA", "VALE3.SA", "BBDC4.SA", "ITUB4.SA", 
               "BBAS3.SA", "B3SA3.SA", "ABEV3.SA", "VIVT3.SA"]
    
    results = {}
    
    for ticker in tickers:
        print(f"\n📊 Testando {ticker}...")
        
        feature_file = input_dir / f'{ticker}.csv'
        if not feature_file.exists():
            print(f"  ❌ Arquivo não encontrado: {feature_file}")
            continue
        
        # Carregar dados
        df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        print(f"  📈 Features originais: {len(df.columns)}")
        
        # Aplicar seleção otimizada
        try:
            df_optimized = selector.apply_global_feature_selection(
                df, ticker, strategy='optimized'
            )
            
            selected_features = df_optimized.columns.tolist()
            print(f"  ✅ Features selecionadas: {len(selected_features)}")
            print(f"  🎯 Features: {selected_features}")
            
            # Verificar se as features estão corretas
            if ticker in selector.per_ticker_cfg:
                config = selector.per_ticker_cfg[ticker]
                if 'best_features' in config:
                    expected_features = config['best_features']
                    optimal_k = config.get('optimal_k', len(expected_features))
                    sharpe_score = config.get('sharpe_score', 0.0)
                    
                    print(f"  📋 Configuração esperada:")
                    print(f"    K ótimo: {optimal_k}")
                    print(f"    Sharpe: {sharpe_score:.4f}")
                    print(f"    Features esperadas: {expected_features}")
                    
                    # Verificar se as features selecionadas estão corretas
                    missing_features = [f for f in expected_features if f not in selected_features]
                    extra_features = [f for f in selected_features if f not in expected_features]
                    
                    if not missing_features and not extra_features:
                        print(f"  ✅ Features corretas!")
                        results[ticker] = "SUCCESS"
                    else:
                        print(f"  ⚠️  Features diferentes:")
                        if missing_features:
                            print(f"    Faltando: {missing_features}")
                        if extra_features:
                            print(f"    Extras: {extra_features}")
                        results[ticker] = "PARTIAL"
                else:
                    print(f"  ⚠️  Configuração 'best_features' não encontrada")
                    results[ticker] = "NO_CONFIG"
            else:
                print(f"  ⚠️  Configuração para {ticker} não encontrada")
                results[ticker] = "NO_CONFIG"
                
        except Exception as e:
            print(f"  ❌ Erro: {e}")
            results[ticker] = "ERROR"
    
    # Resumo dos resultados
    print(f"\n📊 RESUMO DOS TESTES")
    print("=" * 40)
    
    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    total_count = len(results)
    
    for ticker, status in results.items():
        status_emoji = {
            "SUCCESS": "✅",
            "PARTIAL": "⚠️",
            "NO_CONFIG": "❓",
            "ERROR": "❌"
        }
        print(f"{status_emoji[status]} {ticker}: {status}")
    
    print(f"\n🎯 Resultado: {success_count}/{total_count} tickers com sucesso")
    
    if success_count == total_count:
        print("🎉 Todos os testes passaram! A integração está funcionando perfeitamente.")
    elif success_count > 0:
        print("⚠️  Alguns testes passaram. Verifique os tickers com problemas.")
    else:
        print("❌ Nenhum teste passou. Verifique a configuração.")
    
    return results


def main():
    """
    Função principal.
    """
    try:
        results = test_optimized_feature_selection()
        
        # Verificar se podemos executar a seleção completa
        print(f"\n🚀 EXECUTANDO SELEÇÃO COMPLETA DE FEATURES")
        print("=" * 50)
        
        from src.features.feature_selection import main as run_feature_selection
        run_feature_selection()
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
