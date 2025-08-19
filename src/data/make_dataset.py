import yfinance as yf
from pathlib import Path
import yaml
import logging
import shutil
import pandas as pd
import os
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_csv_file(file_path):
    """
    Valida se um arquivo CSV existe e tem dados válidos.
    
    Args:
        file_path: Caminho para o arquivo CSV
        
    Returns:
        bool: True se o arquivo é válido, False caso contrário
    """
    try:
        if not os.path.exists(file_path):
            return False
        
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"Arquivo {file_path} existe mas está vazio")
            return False
        
        # Verificar se tem as colunas básicas esperadas de dados financeiros
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in expected_columns):
            logger.warning(f"Arquivo {file_path} não tem todas as colunas esperadas")
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Erro ao validar arquivo {file_path}: {e}")
        return False

def download_data(ticker, start_date, end_date, raw_data_path, force_update=False):
    """
    Baixa os dados do Yahoo Finance para um determinado ticker e salva em um arquivo CSV.
    Implementa sistema de backup para evitar perda de dados em caso de erro.
    
    Args:
        ticker: String com o ticker do ativo
        start_date: String com a data de início no formato YYYY-MM-DD
        end_date: String com a data de fim no formato YYYY-MM-DD
        raw_data_path: String com o caminho para salvar os dados
        force_update: Bool para forçar atualização mesmo se arquivo existir
    """
    
    Path(raw_data_path).mkdir(parents=True, exist_ok=True)
    
    file_path = Path(raw_data_path) / f"{ticker}.csv"
    backup_path = Path(raw_data_path) / f"{ticker}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    temp_path = Path(raw_data_path) / f"{ticker}_temp.csv"
    
    # Verificar se arquivo já existe e é válido
    file_exists = validate_csv_file(file_path)
    
    if file_exists and not force_update:
        logger.info(f"Arquivo {ticker}.csv já existe e é válido. Pulando download.")
        return True
    
    try:
        logger.info(f"Iniciando download dos dados para {ticker}...")
        
        if file_exists:
            logger.info(f"Fazendo backup do arquivo existente: {ticker}.csv")
            shutil.copy2(file_path, backup_path)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"Nenhum dado foi retornado para o ticker {ticker}")
        
        data.to_csv(temp_path, index=True)
        
        if not validate_csv_file(temp_path):
            raise ValueError(f"Dados baixados para {ticker} são inválidos")
        
        if temp_path.exists():
            shutil.move(temp_path, file_path)
        
        if backup_path.exists():
            os.remove(backup_path)
        
        logger.info(f"Download concluído com sucesso para {ticker}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao baixar dados para {ticker}: {e}")
        
        if temp_path.exists():
            os.remove(temp_path)
        
        if backup_path.exists():
            if file_path.exists():
                os.remove(file_path)
            shutil.move(backup_path, file_path)
            logger.info(f"Backup restaurado para {ticker}")
        
        return False


def main(force_update=False):
    """
    Função principal que baixa todos os dados configurados no config.yaml
    
    Args:
        force_update: Bool para forçar atualização de todos os arquivos
    """
    
    try:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
        
        logger.info("Iniciando processo de download de dados...")
        logger.info(f"Carregando configurações de: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        data_config = config["data"]
        
        # Estatísticas do processo
        total_tickers = len(data_config["tickers"])
        successful_downloads = 0
        skipped_files = 0
        failed_downloads = 0
        
        logger.info(f"Processando {total_tickers} tickers...")
        logger.info(f"Período: {data_config['start_date']} até {data_config['end_date']}")
        logger.info(f"Diretório de destino: {data_config['raw_data_path']}")
        
        if force_update:
            logger.info("Modo force_update ativado - todos os arquivos serão atualizados")
        
        for i, ticker in enumerate(data_config["tickers"], 1):
            logger.info(f"Processando {ticker} ({i}/{total_tickers})...")
            
            result = download_data(
                ticker, 
                data_config["start_date"], 
                data_config["end_date"], 
                data_config["raw_data_path"],
                force_update=force_update
            )
            
            if result is True:
                # Verificar se foi download ou skip
                file_path = Path(data_config["raw_data_path"]) / f"{ticker}.csv"
                if validate_csv_file(file_path):
                    if force_update or not Path(file_path).exists():
                        successful_downloads += 1
                    else:
                        skipped_files += 1
            else:
                failed_downloads += 1
        
        # Relatório final
        logger.info("="*50)
        logger.info("RELATÓRIO FINAL:")
        logger.info(f"Total de tickers processados: {total_tickers}")
        logger.info(f"Downloads bem-sucedidos: {successful_downloads}")
        logger.info(f"Arquivos já existentes (pulados): {skipped_files}")
        logger.info(f"Downloads com falha: {failed_downloads}")
        logger.info("="*50)
        
        if failed_downloads > 0:
            logger.warning(f"Atenção: {failed_downloads} downloads falharam. Verifique os logs acima.")
            return False
        else:
            logger.info("Todos os dados foram processados com sucesso!")
            return True
            
    except Exception as e:
        logger.error(f"Erro crítico na função main: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download de dados financeiros do Yahoo Finance')
    parser.add_argument('--force-update', action='store_true', 
                       help='Forçar atualização de todos os arquivos, mesmo os que já existem')
    
    args = parser.parse_args()
    
    success = main(force_update=args.force_update)
    exit(0 if success else 1)





