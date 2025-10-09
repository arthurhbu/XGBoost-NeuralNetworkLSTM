#!/usr/bin/env python3
"""
Manual Feature Optimizer - Abordagem Manual Baseada em Análise
==============================================================

Baseado na análise dos dados existentes, propõe melhorias manuais
nas configurações de features para maximizar lucros.

Uso:
    python manual_feature_optimizer.py
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

class ManualFeatureOptimizer:
    def __init__(self):
        self.config = self._load_config()
        self.analysis_results = {}
        
    def _load_config(self):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    
    def analyze_ticker_data(self, ticker):
        """Analisa dados de um ticker para propor melhorias."""
        print(f"\n📊 ANALISANDO {ticker}")
        print("-" * 40)
        
        try:
            # Carregar dados
            features_path = Path("data/03_features") / f"{ticker}.csv"
            labeled_path = Path("data/04_labeled") / f"{ticker}.csv"
            
            features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
            labeled_df = pd.read_csv(labeled_path, index_col=0, parse_dates=True)
            
            # Combinar dados
            df = features_df.join(labeled_df[['target']], how='inner')
            
            # Analisar distribuição de targets
            target_dist = df['target'].value_counts().sort_index()
            print(f"  Distribuição de targets: {dict(target_dist)}")
            
            # Calcular desbalanceamento
            class_counts = target_dist.values
            min_class = class_counts.min()
            max_class = class_counts.max()
            imbalance_ratio = max_class / min_class
            
            print(f"  Desbalanceamento: {imbalance_ratio:.2f}")
            
            # Analisar correlação entre features e target
            feature_correlations = {}
            for col in features_df.columns:
                if col != 'target':
                    corr = df[col].corr(df['target'])
                    feature_correlations[col] = abs(corr)
            
            # Ordenar por correlação
            sorted_features = sorted(feature_correlations.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            print(f"  Top 5 features por correlação:")
            for i, (feature, corr) in enumerate(sorted_features[:5]):
                print(f"    {i+1}. {feature}: {corr:.4f}")
            
            # Propor melhorias baseadas na análise
            improvements = self._propose_improvements(ticker, sorted_features, imbalance_ratio)
            
            return improvements
            
        except Exception as e:
            print(f"  ❌ Erro ao analisar {ticker}: {e}")
            return None
    
    def _propose_improvements(self, ticker, sorted_features, imbalance_ratio):
        """Propõe melhorias baseadas na análise."""
        improvements = {
            'ticker': ticker,
            'imbalance_ratio': imbalance_ratio,
            'recommended_features': [],
            'strategy': '',
            'reasoning': []
        }
        
        # Selecionar top features
        top_features = [f[0] for f in sorted_features[:6]]
        improvements['recommended_features'] = top_features
        
        # Determinar estratégia baseada no ticker e desbalanceamento
        if ticker in ['ITUB4.SA', 'VIVT3.SA'] and imbalance_ratio > 3:
            improvements['strategy'] = 'rebalance_focus'
            improvements['reasoning'] = [
                'Desbalanceamento severo detectado',
                'Focar em features com maior correlação',
                'Considerar undersampling da classe majoritária',
                'Usar thresholds mais agressivos'
            ]
        elif ticker in ['BBAS3.SA'] and imbalance_ratio < 2:
            improvements['strategy'] = 'balanced_optimization'
            improvements['reasoning'] = [
                'Distribuição balanceada',
                'Otimizar features existentes',
                'Focar em momentum e volume',
                'Ajustar thresholds moderadamente'
            ]
        else:
            improvements['strategy'] = 'standard_optimization'
            improvements['reasoning'] = [
                'Otimização padrão',
                'Manter features de alta correlação',
                'Ajustar thresholds baseado na performance'
            ]
        
        return improvements
    
    def analyze_all_tickers(self):
        """Analisa todos os tickers."""
        tickers = self.config['data']['tickers']
        
        print("🔍 MANUAL FEATURE OPTIMIZER")
        print("="*60)
        print("Análise manual baseada em dados existentes")
        print("="*60)
        
        for ticker in tickers:
            result = self.analyze_ticker_data(ticker)
            if result:
                self.analysis_results[ticker] = result
        
        # Gerar relatório
        self._generate_report()
    
    def _generate_report(self):
        """Gera relatório final."""
        print("\n" + "="*80)
        print("📋 RELATÓRIO DE ANÁLISE E RECOMENDAÇÕES")
        print("="*80)
        
        if not self.analysis_results:
            print("❌ Nenhum resultado encontrado")
            return
        
        print("\n🎯 ANÁLISE POR TICKER:")
        print("-" * 60)
        
        for ticker, result in self.analysis_results.items():
            print(f"\n{ticker}:")
            print(f"  Estratégia: {result['strategy']}")
            print(f"  Desbalanceamento: {result['imbalance_ratio']:.2f}")
            print(f"  Features recomendadas: {result['recommended_features']}")
            print(f"  Justificativa:")
            for reason in result['reasoning']:
                print(f"    - {reason}")
        
        # Gerar configuração otimizada
        self._generate_optimized_config()
    
    def _generate_optimized_config(self):
        """Gera configuração otimizada."""
        print("\n💾 GERANDO CONFIGURAÇÃO OTIMIZADA...")
        
        # Atualizar config
        config = self.config.copy()
        
        if 'feature_selection' not in config:
            config['feature_selection'] = {}
        
        if 'per_ticker' not in config['feature_selection']:
            config['feature_selection']['per_ticker'] = {}
        
        # Adicionar resultados
        for ticker, result in self.analysis_results.items():
            config['feature_selection']['per_ticker'][ticker] = {
                'optimal_k': len(result['recommended_features']),
                'best_features': result['recommended_features'],
                'strategy': result['strategy'],
                'imbalance_ratio': result['imbalance_ratio'],
                'reasoning': result['reasoning']
            }
        
        # Salvar configuração otimizada
        with open("config_manual_optimized.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print("✅ Configuração salva em: config_manual_optimized.yaml")
        
        # Gerar resumo das mudanças
        self._generate_change_summary()
    
    def _generate_change_summary(self):
        """Gera resumo das mudanças propostas."""
        print("\n📝 RESUMO DAS MUDANÇAS PROPOSTAS:")
        print("-" * 60)
        
        for ticker, result in self.analysis_results.items():
            print(f"\n{ticker}:")
            print(f"  Features atuais: {self._get_current_features(ticker)}")
            print(f"  Features propostas: {result['recommended_features']}")
            print(f"  Estratégia: {result['strategy']}")
    
    def _get_current_features(self, ticker):
        """Obtém features atuais do ticker."""
        current_config = self.config.get('feature_selection', {}).get('per_ticker', {}).get(ticker, {})
        return current_config.get('best_features', [])
    
    def generate_implementation_guide(self):
        """Gera guia de implementação."""
        print("\n" + "="*80)
        print("🚀 GUIA DE IMPLEMENTAÇÃO RÁPIDA")
        print("="*80)
        
        print("\n1. BACKUP DO CONFIG ATUAL:")
        print("   cp config.yaml config_backup.yaml")
        
        print("\n2. APLICAR CONFIGURAÇÃO OTIMIZADA:")
        print("   cp config_manual_optimized.yaml config.yaml")
        
        print("\n3. EXECUTAR BACKTESTING:")
        print("   python -m src.backtesting.backtest")
        
        print("\n4. COMPARAR RESULTADOS:")
        print("   - Verificar se os tickers problemáticos melhoraram")
        print("   - Analisar métricas de Sharpe e retorno")
        print("   - Ajustar thresholds se necessário")
        
        print("\n5. AJUSTES ADICIONAIS (se necessário):")
        print("   - Para tickers com desbalanceamento severo:")
        print("     * Reduzir thresholds (buy: 0.4, sell: 0.3)")
        print("     * Aplicar undersampling da classe majoritária")
        print("   - Para tickers balanceados:")
        print("     * Manter thresholds atuais")
        print("     * Focar em otimização de features")

def main():
    """Função principal."""
    optimizer = ManualFeatureOptimizer()
    optimizer.analyze_all_tickers()
    optimizer.generate_implementation_guide()

if __name__ == "__main__":
    main()
