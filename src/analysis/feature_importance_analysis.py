#!/usr/bin/env python3
"""
Análise de Feature Importance para Modelos XGBoost
==================================================

Este script analisa a importância das features para todos os tickers
e identifica padrões que podem explicar diferenças de performance.

Uso:
    python -m src.analysis.feature_importance_analysis
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_feature_importance_data(reports_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Carrega dados de feature importance de todos os tickers.
    
    Args:
        reports_dir: Diretório onde estão os arquivos de feature importance
    
    Returns:
        Dict com DataFrames de importância por ticker
    """
    importance_data = {}
    
    for file_path in reports_dir.glob("feature_importance/feature_importance_*.csv"):
        ticker = file_path.stem.replace("feature_importance_", "")
        try:
            df = pd.read_csv(file_path)
            importance_data[ticker] = df
            print(f"Carregado: {ticker} ({len(df)} features)")
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
    
    return importance_data


def analyze_ticker_performance(importance_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Analisa performance das features por ticker.
    
    Args:
        importance_data: Dados de importância por ticker
    
    Returns:
        DataFrame com análise de performance
    """
    analysis_results = []
    
    for ticker, df in importance_data.items():
        if df.empty:
            continue
            
        # Calcular métricas de qualidade das features
        total_features = len(df)
        features_with_importance = (df['importance_gain'] > 0).sum()
        top_10_importance = df.head(10)['importance_pct'].sum()
        top_5_importance = df.head(5)['importance_pct'].sum()
        
        # Calcular concentração de importância
        concentration_ratio = top_5_importance / 100 if top_5_importance > 0 else 0
        
        # Identificar features mais importantes
        top_features = df.head(3)['feature'].tolist()
        
        analysis_results.append({
            'ticker': ticker,
            'total_features': total_features,
            'features_with_importance': features_with_importance,
            'top_10_importance_pct': top_10_importance,
            'top_5_importance_pct': top_5_importance,
            'concentration_ratio': concentration_ratio,
            'top_3_features': ', '.join(top_features)
        })
    
    return pd.DataFrame(analysis_results)


def identify_problematic_features(importance_data: Dict[str, pd.DataFrame], 
                                problematic_tickers: List[str] = None) -> Dict:
    """
    Identifica features problemáticas para tickers com baixa performance.
    
    Args:
        importance_data: Dados de importância por ticker
        problematic_tickers: Lista de tickers problemáticos
    
    Returns:
        Dict com análise de features problemáticas
    """
    if problematic_tickers is None:
        problematic_tickers = ['B3SA3.SA', 'VIVT3.SA']
    
    analysis = {}
    
    for ticker in problematic_tickers:
        if ticker not in importance_data:
            continue
            
        df = importance_data[ticker]
        if df.empty:
            continue
        
        # Identificar features com baixa importância
        low_importance_features = df[df['importance_gain'] < df['importance_gain'].quantile(0.5)]['feature'].tolist()
        
        # Identificar features com alta variância (instáveis)
        high_variance_features = df[df['importance_gain'] > df['importance_gain'].quantile(0.9)]['feature'].tolist()
        
        # Features que podem estar causando overfitting
        overfitting_candidates = df[
            (df['importance_gain'] > df['importance_gain'].quantile(0.95)) & 
            (df['importance_pct'] > 20)
        ]['feature'].tolist()
        
        analysis[ticker] = {
            'low_importance_features': low_importance_features,
            'high_variance_features': high_variance_features,
            'overfitting_candidates': overfitting_candidates,
            'total_problematic': len(low_importance_features) + len(overfitting_candidates)
        }
    
    return analysis


def create_feature_importance_visualization(importance_data: Dict[str, pd.DataFrame], 
                                          output_dir: Path):
    """
    Cria visualizações da importância das features.
    
    Args:
        importance_data: Dados de importância por ticker
        output_dir: Diretório para salvar as visualizações
    """
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Heatmap de importância por ticker
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Análise de Feature Importance por Ticker', fontsize=16, fontweight='bold')
    
    # Preparar dados para heatmap
    all_features = set()
    for df in importance_data.values():
        all_features.update(df['feature'].tolist())
    
    all_features = sorted(list(all_features))
    tickers = list(importance_data.keys())
    
    # Matriz de importância
    importance_matrix = np.zeros((len(all_features), len(tickers)))
    
    for i, feature in enumerate(all_features):
        for j, ticker in enumerate(tickers):
            if ticker in importance_data:
                feature_data = importance_data[ticker]
                feature_row = feature_data[feature_data['feature'] == feature]
                if not feature_row.empty:
                    importance_matrix[i, j] = feature_row['importance_gain'].iloc[0]
    
    # Plot 1: Heatmap de importância
    ax1 = axes[0, 0]
    im = ax1.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(tickers)))
    ax1.set_xticklabels(tickers, rotation=45)
    ax1.set_yticks(range(0, len(all_features), max(1, len(all_features)//10)))
    ax1.set_yticklabels([all_features[i] for i in range(0, len(all_features), max(1, len(all_features)//10))])
    ax1.set_title('Heatmap de Importância das Features')
    ax1.set_xlabel('Ticker')
    ax1.set_ylabel('Features')
    plt.colorbar(im, ax=ax1, label='Importância (Gain)')
    
    # Plot 2: Top 10 features por ticker
    ax2 = axes[0, 1]
    for ticker in tickers:
        if ticker in importance_data:
            df = importance_data[ticker].head(10)
            ax2.plot(df['importance_pct'], label=ticker, marker='o')
    ax2.set_title('Top 10 Features por Ticker')
    ax2.set_xlabel('Ranking')
    ax2.set_ylabel('Importância (%)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribuição de importância
    ax3 = axes[1, 0]
    importance_values = []
    ticker_labels = []
    for ticker, df in importance_data.items():
        importance_values.extend(df['importance_gain'].tolist())
        ticker_labels.extend([ticker] * len(df))
    
    importance_df = pd.DataFrame({
        'importance': importance_values,
        'ticker': ticker_labels
    })
    
    sns.boxplot(data=importance_df, x='ticker', y='importance', ax=ax3)
    ax3.set_title('Distribuição de Importância por Ticker')
    ax3.set_xlabel('Ticker')
    ax3.set_ylabel('Importância (Gain)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Concentração de importância
    ax4 = axes[1, 1]
    concentration_data = []
    for ticker, df in importance_data.items():
        top_5_pct = df.head(5)['importance_pct'].sum()
        concentration_data.append({'ticker': ticker, 'top_5_concentration': top_5_pct})
    
    concentration_df = pd.DataFrame(concentration_data)
    bars = ax4.bar(concentration_df['ticker'], concentration_df['top_5_concentration'])
    ax4.set_title('Concentração de Importância (Top 5 Features)')
    ax4.set_xlabel('Ticker')
    ax4.set_ylabel('Importância Acumulada (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Colorir barras baseado na concentração
    for i, bar in enumerate(bars):
        if concentration_df.iloc[i]['top_5_concentration'] > 70:
            bar.set_color('red')  # Alta concentração (pode ser overfitting)
        elif concentration_df.iloc[i]['top_5_concentration'] > 50:
            bar.set_color('orange')  # Concentração moderada
        else:
            bar.set_color('green')  # Baixa concentração (bom)
    
    plt.tight_layout()
    
    # Salvar visualização
    output_file = output_dir / "feature_importance_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualização salva em: {output_file}")
    
    plt.close()


def generate_recommendations(analysis_results: pd.DataFrame, 
                           problematic_analysis: Dict) -> str:
    """
    Gera recomendações baseadas na análise de feature importance.
    
    Args:
        analysis_results: Resultados da análise de performance
        problematic_analysis: Análise de tickers problemáticos
    
    Returns:
        String com recomendações
    """
    recommendations = []
    recommendations.append("=" * 80)
    recommendations.append("RECOMENDAÇÕES BASEADAS NA ANÁLISE DE FEATURE IMPORTANCE")
    recommendations.append("=" * 80)
    
    # Análise geral
    recommendations.append("\n1. ANÁLISE GERAL:")
    recommendations.append(f"   - Total de tickers analisados: {len(analysis_results)}")
    recommendations.append(f"   - Tickers com alta concentração (>70%): {len(analysis_results[analysis_results['top_5_importance_pct'] > 70])}")
    recommendations.append(f"   - Tickers com baixa concentração (<50%): {len(analysis_results[analysis_results['top_5_importance_pct'] < 50])}")
    
    # Tickers problemáticos
    recommendations.append("\n2. TICKERS PROBLEMÁTICOS:")
    for ticker, analysis in problematic_analysis.items():
        recommendations.append(f"\n   {ticker}:")
        recommendations.append(f"   - Features com baixa importância: {len(analysis['low_importance_features'])}")
        recommendations.append(f"   - Features com alta variância: {len(analysis['high_variance_features'])}")
        recommendations.append(f"   - Candidatos a overfitting: {len(analysis['overfitting_candidates'])}")
        
        if analysis['overfitting_candidates']:
            recommendations.append(f"   - Features problemáticas: {', '.join(analysis['overfitting_candidates'][:3])}")
    
    # Recomendações específicas
    recommendations.append("\n3. RECOMENDAÇÕES ESPECÍFICAS:")
    
    # Para tickers com alta concentração
    high_concentration = analysis_results[analysis_results['top_5_importance_pct'] > 70]
    if not high_concentration.empty:
        recommendations.append("\n   Para tickers com alta concentração de importância:")
        for _, row in high_concentration.iterrows():
            recommendations.append(f"   - {row['ticker']}: Considerar remover features menos importantes")
            recommendations.append(f"     Top features: {row['top_3_features']}")
    
    # Para tickers com baixa concentração
    low_concentration = analysis_results[analysis_results['top_5_importance_pct'] < 50]
    if not low_concentration.empty:
        recommendations.append("\n   Para tickers com baixa concentração de importância:")
        for _, row in low_concentration.iterrows():
            recommendations.append(f"   - {row['ticker']}: Considerar adicionar features mais específicas")
    
    # Recomendações gerais
    recommendations.append("\n4. RECOMENDAÇÕES GERAIS:")
    recommendations.append("   - Implementar feature selection baseada em importância")
    recommendations.append("   - Testar remoção de features com baixa importância")
    recommendations.append("   - Considerar feature engineering específico por ticker")
    recommendations.append("   - Implementar regularização mais forte para tickers com overfitting")
    
    return "\n".join(recommendations)


def main():
    """Função principal para análise de feature importance."""
    print("INICIANDO ANÁLISE DE FEATURE IMPORTANCE")
    print("=" * 50)
    
    # Configurar diretórios
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "reports"
    output_dir = project_root / "reports" / "feature_importance"	
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(exist_ok=True)
    
    # 1. Carregar dados de feature importance
    print("\n1. Carregando dados de feature importance...")
    importance_data = load_feature_importance_data(reports_dir)
    
    if not importance_data:
        print("Nenhum arquivo de feature importance encontrado!")
        print("   Execute primeiro: python -m src.models.train_models")
        return
    
    # 2. Analisar performance das features
    print("\n2. Analisando performance das features...")
    analysis_results = analyze_ticker_performance(importance_data)
    
    # Salvar análise
    analysis_output = output_dir / "feature_importance_analysis.csv"
    analysis_results.to_csv(analysis_output, index=False)
    print(f"Análise salva em: {analysis_output}")
    
    # 3. Identificar features problemáticas
    print("\n3. Identificando features problemáticas...")
    problematic_tickers = ['B3SA3.SA', 'VIVT3.SA']
    problematic_analysis = identify_problematic_features(importance_data, problematic_tickers)
    
    # 4. Criar visualizações
    print("\n4. Criando visualizações...")
    create_feature_importance_visualization(importance_data, output_dir)
    
    # 5. Gerar recomendações
    print("\n5. Gerando recomendações...")
    recommendations = generate_recommendations(analysis_results, problematic_analysis)
    
    # Salvar recomendações
    recommendations_output = output_dir / "feature_importance_recommendations.txt"
    with open(recommendations_output, 'w', encoding='utf-8') as f:
        f.write(recommendations)
    print(f"Recomendações salvas em: {recommendations_output}")
    
    # 6. Mostrar resumo
    print("\n" + "=" * 50)
    print("RESUMO DA ANÁLISE")
    print("=" * 50)
    print(f"Tickers analisados: {len(importance_data)}")
    print(f"Features analisadas: {len(set().union(*[df['feature'].tolist() for df in importance_data.values()]))}")
    print(f"Arquivos gerados:")
    print(f"  - {analysis_output}")
    print(f"  - {output_dir / 'feature_importance_analysis.png'}")
    print(f"  - {recommendations_output}")
    
    print("\nAnálise de feature importance concluída!")


if __name__ == "__main__":
    main()
