import yfinance as yf
from pathlib import Path
import yaml

def download_data(ticker, start_date, end_date, raw_data_path):
    """
    Baixa os dados do Yahoo Finance para um determinado ticker e salva em um arquivo CSV.
    
    Args:
        ticker: String com o ticker do ativo
        start_date: String com a data de início no formato YYYY-MM-DD
        end_date: String com a data de fim no formato YYYY-MM-DD
        raw_data_path: String com o caminho para salvar os dados
    """
    
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"{raw_data_path}/{ticker}.csv", index=False)


def main():
    
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    
    for ticker in data_config["tickers"]:
        download_data(ticker, data_config["start_date"], data_config["end_date"], data_config["raw_data_path"])


if __name__ == "__main__":
    main()





