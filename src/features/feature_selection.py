#!/usr/bin/env python3
"""
Sistema de Feature Selection Global
==================================

Este módulo implementa estratégias de seleção de features baseadas em análise
de importância para melhorar a performance de todos os tickers.

Uso:
    python -m src.features.feature_selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

class GlobalFeatureSelector:
    """
    Classe para seleção global de features baseada em análise de importância.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o seletor de features.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.config_path = config_path
        self.config = self._load_config()

        # Ler listas do config quando disponíveis; manter defaults como fallback
        fs_cfg = (self.config or {}).get('feature_selection', {})
        self.global_low_importance_features = fs_cfg.get('global_low_importance', [
            'Volume', 'MFI', 'MACD_hist', 'MACD_signal', 'ADX'
        ])
        self.global_high_importance_features = fs_cfg.get('global_high_importance', [
            'wavelet_cD', 'wavelet_cA', 'EMA_long', 'EMA_short',
            'BB_upper', 'BB_lower', 'BB_middle', 'ATR', 'OBV',
            'Close', 'Open', 'High', 'Low', 'RSI', 'MACD'
        ])
        self.per_ticker_cfg = fs_cfg.get('per_ticker', {})
    
    def _load_config(self) -> dict:
        """Carrega configuração do arquivo YAML."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def apply_global_feature_selection(self, df: pd.DataFrame, 
                                     ticker: str = None,
                                     strategy: str = 'conservative') -> pd.DataFrame:
        """
        Aplica seleção global de features.
        
        Args:
            df: DataFrame com features
            ticker: Nome do ticker (opcional, para logging)
            strategy: Estratégia de seleção ('conservative', 'aggressive', 'custom')
        
        Returns:
            DataFrame com features selecionadas
        """
        original_features = df.columns.tolist()
        
        if strategy == 'conservative':
            # Remover apenas features consistentemente menos importantes
            features_to_remove = self.global_low_importance_features
            features_to_keep = [col for col in original_features 
                              if col not in features_to_remove]
            
        elif strategy == 'aggressive':
            # Manter apenas features de alta importância
            features_to_keep = [col for col in original_features 
                              if col in self.global_high_importance_features]
            
        elif strategy == 'optimized':
            # NOVA: Usar as melhores features descobertas pelo feature sweep
            features_to_keep = self._get_custom_features(ticker, original_features)
            
        elif strategy == 'custom':
            # Estratégia personalizada por ticker (legado)
            features_to_keep = self._get_custom_features(ticker, original_features)
            
        else:
            raise ValueError(f"Estratégia '{strategy}' não reconhecida. Use: 'conservative', 'aggressive', 'optimized', ou 'custom'")
        
        # Aplicar seleção
        df_selected = df[features_to_keep].copy()
        
        # Log da seleção
        removed_features = [col for col in original_features 
                          if col not in features_to_keep]
        
        if ticker:
            print(f"  {ticker}: Removidas {len(removed_features)} features")
            print(f"    Features removidas: {removed_features}")
            print(f"    Features mantidas: {len(features_to_keep)}")
        
        return df_selected
    
    def _get_custom_features(self, ticker: str, original_features: List[str]) -> List[str]:
        """
        Retorna features customizadas para um ticker específico baseadas no feature sweep otimizado.
        
        Args:
            ticker: Nome do ticker
            original_features: Lista de features originais
        
        Returns:
            Lista de features customizadas (melhores features descobertas pelo sweep)
        """
        # Configurações específicas por ticker vindas do config.yaml
        if ticker in self.per_ticker_cfg:
            config = self.per_ticker_cfg[ticker]
            
            # NOVO: Usar as melhores features descobertas pelo feature sweep
            if 'best_features' in config:
                best_features = config['best_features']
                optimal_k = config.get('optimal_k', len(best_features))
                sharpe_score = config.get('sharpe_score', 0.0)
                
                print(f"  🎯 Usando features otimizadas para {ticker}:")
                print(f"    K ótimo: {optimal_k}")
                print(f"    Sharpe score: {sharpe_score:.4f}")
                print(f"    Features: {best_features}")
                
                # Retornar apenas as features que existem no dataset
                available_features = [f for f in best_features if f in original_features]
                return available_features
            
            # Fallback: usar configuração antiga se disponível
            elif 'remove' in config:
                features_to_remove = config.get('remove', [])
                return [col for col in original_features if col not in features_to_remove]
        
        # Para outros tickers, usar estratégia conservadora
        return [col for col in original_features 
               if col not in self.global_low_importance_features]
    
    def analyze_feature_impact(self, original_df: pd.DataFrame, 
                              selected_df: pd.DataFrame, 
                              ticker: str) -> Dict:
        """
        Analisa o impacto da seleção de features.
        
        Args:
            original_df: DataFrame original
            selected_df: DataFrame com features selecionadas
            ticker: Nome do ticker
        
        Returns:
            Dict com métricas de impacto
        """
        original_features = original_df.columns.tolist()
        selected_features = selected_df.columns.tolist()
        removed_features = [col for col in original_features 
                          if col not in selected_features]
        
        impact_analysis = {
            'ticker': ticker,
            'original_features': len(original_features),
            'selected_features': len(selected_features),
            'removed_features': len(removed_features),
            'reduction_pct': (len(removed_features) / len(original_features)) * 100,
            'removed_feature_list': removed_features,
            'selected_feature_list': selected_features
        }
        
        return impact_analysis


def apply_global_feature_selection_to_all_tickers(
    input_dir: Path, 
    output_dir: Path, 
    strategy: str = 'conservative'
) -> Dict[str, Dict]:
    """
    Aplica seleção global de features para todos os tickers.
    
    Args:
        input_dir: Diretório com dados originais
        output_dir: Diretório para salvar dados processados
        strategy: Estratégia de seleção
    
    Returns:
        Dict com análise de impacto por ticker
    """
    selector = GlobalFeatureSelector()
    impact_analysis = {}
    
    # Criar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"APLICANDO SELEÇÃO GLOBAL DE FEATURES (Estratégia: {strategy})")
    print("=" * 60)
    
    # Processar cada arquivo de features
    for feature_file in input_dir.glob("*.csv"):
        if feature_file.name.startswith('.'):
            continue
            
        ticker = feature_file.stem
        print(f"\nProcessando {ticker}...")
        
        try:
            # Carregar dados originais
            df_original = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            
            # Aplicar seleção de features
            df_selected = selector.apply_global_feature_selection(
                df_original, ticker, strategy
            )
            
            # Salvar dados processados
            output_file = output_dir / f"{ticker}.csv"
            df_selected.to_csv(output_file)
            print(f"  Salvo em: {output_file}")
            
            # Analisar impacto
            impact = selector.analyze_feature_impact(df_original, df_selected, ticker)
            impact_analysis[ticker] = impact
            
        except Exception as e:
            print(f"  Erro ao processar {ticker}: {e}")
            continue
    
    # Salvar relatório de impacto
    impact_df = pd.DataFrame.from_dict(impact_analysis, orient='index')
    impact_report = output_dir / "feature_selection_impact_report.csv"
    impact_df.to_csv(impact_report, index=False)
    print(f"\nRelatório de impacto salvo em: {impact_report}")
    
    # Mostrar resumo
    print(f"\nRESUMO DA SELEÇÃO DE FEATURES")
    print("=" * 40)
    print(f"Tickers processados: {len(impact_analysis)}")
    print(f"Redução média de features: {impact_df['reduction_pct'].mean():.1f}%")
    print(f"Features removidas em média: {impact_df['removed_features'].mean():.1f}")
    
    return impact_analysis


def main():
    """Função principal para aplicação de seleção de features."""
    project_root = Path(__file__).resolve().parents[2]
    
    # Diretórios
    input_dir = project_root / "data" / "03_features"
    output_dir = project_root / "data" / "03_features_selected"
    
    # Aplicar seleção otimizada (baseada no feature sweep melhorado)
    print("🚀 APLICANDO SELEÇÃO DE FEATURES OTIMIZADA")
    print("   Baseada nos melhores resultados do feature sweep melhorado")
    print("=" * 60)
    
    impact_analysis = apply_global_feature_selection_to_all_tickers(
        input_dir, output_dir, strategy='optimized'
    )
    
    print(f"\n✅ Seleção global de features otimizada concluída!")
    print(f"   Dados processados salvos em: {output_dir}")
    print(f"   Usando as melhores features descobertas pelo sweep para cada ticker")


if __name__ == "__main__":
    main()
