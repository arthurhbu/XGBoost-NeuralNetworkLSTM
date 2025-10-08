#!/usr/bin/env python3
"""
Gerenciador de Colunas Essenciais
=================================

Este módulo gerencia as colunas essenciais que são necessárias para o train_models
mas podem ser removidas pelo feature selection. Ele captura essas colunas antes
da seleção e as disponibiliza quando necessário.

Colunas essenciais identificadas:
- ATR: Para triple barrier method dinâmico
- Open: Para preço de entrada
- High: Para verificar barreira de lucro  
- Low: Para verificar barreira de perda
- Close: Para cálculo do portfólio

Uso:
    from src.data.essential_columns_manager import EssentialColumnsManager
    
    manager = EssentialColumnsManager()
    df_with_essentials = manager.add_essential_columns(df_features, df_selected)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import yaml


class EssentialColumnsManager:
    """
    Gerencia colunas essenciais que são necessárias para o train_models
    mas podem ser removidas pelo feature selection.
    """
    
    # Colunas essenciais que devem ser preservadas
    ESSENTIAL_COLUMNS = ['ATR', 'Open', 'High', 'Low', 'Close']
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa o gerenciador de colunas essenciais.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Diretório para salvar colunas essenciais
        self.essential_dir = Path(__file__).resolve().parents[2] / "data" / "04_essential_columns"
        self.essential_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> dict:
        """Carrega configuração do arquivo YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Erro ao carregar config: {e}")
            return {}
    
    def extract_essential_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Extrai apenas as colunas essenciais de um DataFrame.
        
        Args:
            df: DataFrame com todas as colunas
            ticker: Nome do ticker
            
        Returns:
            DataFrame apenas com colunas essenciais
        """
        available_essential = [col for col in self.ESSENTIAL_COLUMNS if col in df.columns]
        
        if not available_essential:
            print(f"⚠️  Nenhuma coluna essencial encontrada para {ticker}")
            return pd.DataFrame()
        
        essential_df = df[available_essential].copy()
        
        # Salvar colunas essenciais
        essential_file = self.essential_dir / f"{ticker}_essential.csv"
        essential_df.to_csv(essential_file)
        
        print(f"📋 Colunas essenciais extraídas para {ticker}: {available_essential}")
        print(f"   Salvo em: {essential_file}")
        
        return essential_df
    
    def load_essential_columns(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Carrega colunas essenciais salvas para um ticker.
        
        Args:
            ticker: Nome do ticker
            
        Returns:
            DataFrame com colunas essenciais ou None se não encontrado
        """
        essential_file = self.essential_dir / f"{ticker}_essential.csv"
        
        if not essential_file.exists():
            print(f"⚠️  Arquivo de colunas essenciais não encontrado para {ticker}")
            return None
        
        try:
            df = pd.read_csv(essential_file, index_col=0, parse_dates=True)
            print(f"📋 Colunas essenciais carregadas para {ticker}: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"❌ Erro ao carregar colunas essenciais para {ticker}: {e}")
            return None
    
    def add_essential_columns(self, df_features: pd.DataFrame, df_selected: pd.DataFrame, 
                            ticker: str) -> pd.DataFrame:
        """
        Adiciona colunas essenciais ao DataFrame com features selecionadas.
        
        Args:
            df_features: DataFrame original com todas as features
            df_selected: DataFrame com features selecionadas
            ticker: Nome do ticker
            
        Returns:
            DataFrame com features selecionadas + colunas essenciais
        """
        # Extrair colunas essenciais do DataFrame original
        essential_df = self.extract_essential_columns(df_features, ticker)
        
        if essential_df.empty:
            print(f"⚠️  Nenhuma coluna essencial disponível para {ticker}")
            return df_selected
        
        # Combinar features selecionadas com colunas essenciais
        # Remover duplicatas (caso alguma coluna essencial já esteja nas features selecionadas)
        essential_cols = [col for col in essential_df.columns if col not in df_selected.columns]
        
        if essential_cols:
            df_combined = pd.concat([df_selected, essential_df[essential_cols]], axis=1)
            print(f"✅ Adicionadas {len(essential_cols)} colunas essenciais para {ticker}: {essential_cols}")
        else:
            df_combined = df_selected.copy()
            print(f"ℹ️  Todas as colunas essenciais já estão nas features selecionadas para {ticker}")
        
        return df_combined
    
    def process_all_tickers(self, input_dir: Path, output_dir: Path) -> Dict[str, List[str]]:
        """
        Processa todos os tickers, extraindo colunas essenciais.
        
        Args:
            input_dir: Diretório com dados originais
            output_dir: Diretório para salvar dados processados
            
        Returns:
            Dict com colunas essenciais por ticker
        """
        tickers = self.config.get('data', {}).get('tickers', [])
        essential_summary = {}
        
        print("🔧 EXTRAINDO COLUNAS ESSENCIAIS PARA TODOS OS TICKERS")
        print("=" * 60)
        
        for ticker in tickers:
            print(f"\n📊 Processando {ticker}...")
            
            feature_file = input_dir / f'{ticker}.csv'
            if not feature_file.exists():
                print(f"  ❌ Arquivo não encontrado: {feature_file}")
                continue
            
            try:
                # Carregar dados originais
                df_original = pd.read_csv(feature_file, index_col=0, parse_dates=True)
                
                # Extrair colunas essenciais
                essential_df = self.extract_essential_columns(df_original, ticker)
                
                if not essential_df.empty:
                    essential_summary[ticker] = essential_df.columns.tolist()
                    print(f"  ✅ {len(essential_df.columns)} colunas essenciais extraídas")
                else:
                    essential_summary[ticker] = []
                    print(f"  ⚠️  Nenhuma coluna essencial encontrada")
                
            except Exception as e:
                print(f"  ❌ Erro ao processar {ticker}: {e}")
                essential_summary[ticker] = []
                continue
        
        # Salvar resumo
        summary_file = self.essential_dir / "essential_columns_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(essential_summary, f, indent=2)
        
        print(f"\n📋 Resumo salvo em: {summary_file}")
        print(f"✅ Processamento concluído para {len(tickers)} tickers")
        
        return essential_summary
    
    def get_essential_columns_info(self) -> Dict[str, str]:
        """
        Retorna informações sobre as colunas essenciais.
        
        Returns:
            Dict com descrição de cada coluna essencial
        """
        return {
            'ATR': 'Average True Range - usado para triple barrier method dinâmico',
            'Open': 'Preço de abertura - usado como preço de entrada',
            'High': 'Preço máximo - usado para verificar barreira de lucro',
            'Low': 'Preço mínimo - usado para verificar barreira de perda',
            'Close': 'Preço de fechamento - usado para cálculo do portfólio'
        }


def main():
    """
    Função principal para extrair colunas essenciais de todos os tickers.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    # Diretórios
    input_dir = project_root / "data" / "03_features"
    
    # Inicializar gerenciador
    manager = EssentialColumnsManager()
    
    # Processar todos os tickers
    essential_summary = manager.process_all_tickers(input_dir, None)
    
    # Mostrar resumo
    print(f"\n📊 RESUMO DAS COLUNAS ESSENCIAIS")
    print("=" * 50)
    
    for ticker, columns in essential_summary.items():
        if columns:
            print(f"✅ {ticker}: {columns}")
        else:
            print(f"❌ {ticker}: Nenhuma coluna essencial")
    
    print(f"\n💡 INFORMAÇÕES SOBRE AS COLUNAS ESSENCIAIS:")
    for col, desc in manager.get_essential_columns_info().items():
        print(f"  {col}: {desc}")


if __name__ == "__main__":
    main()
