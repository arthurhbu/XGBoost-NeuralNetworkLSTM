import pandas as pd
from pathlib import Path
import yaml
import talib
import numpy as np
import pywt

def clean_and_convert_data(dataframe):
    """
    Limpa e converte os dados provenientes do CSV para float64 para ser possível manipular e utilizar as funções do talib e também remove as linhas com valores não numéricos.
    
    Args:
        dataframe: DataFrame com os dados provenientes do CSV
        
    Returns:
        DataFrame com os dados convertidos para float64 e sem valores não numéricos
    """
    dataframe = dataframe.copy()
    
    numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    for col in numeric_columns:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    
    dataframe = dataframe.dropna()
    
    for col in numeric_columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype('float64')
            
    return dataframe


def create_technical_indicators(dataframe, indicators_config):
    """
    Cria as features técnicas utilizando as funções do talib.
    
    Args:
        dataframe: DataFrame com os dados provenientes do CSV
        indicators_config: Dicionário com as configurações dos indicadores técnicos
        
    Returns:
        DataFrame com as features técnicas criadas
    """
    
    dataframe = clean_and_convert_data(dataframe)
    
    close_prices = dataframe['Close'].values
    high_prices = dataframe['High'].values
    low_prices = dataframe['Low'].values
    open_prices = dataframe['Open'].values
    
    dataframe.loc[:, 'RSI'] = talib.RSI(close_prices, timeperiod=indicators_config['rsi_length'])
    
    ema_short = talib.EMA(close_prices, timeperiod=indicators_config['ema_short'])
    ema_long = talib.EMA(close_prices, timeperiod=indicators_config['ema_long'])
    
    dataframe.loc[:, 'EMA_short'] = ema_short 
    dataframe.loc[:, 'EMA_long'] = ema_long
    
    macd, macdsignal, macdhist = talib.MACD(close_prices,fastperiod=indicators_config['macd_fast'], slowperiod=indicators_config['macd_slow'], signalperiod=indicators_config['macd_signal'])
    
    dataframe.loc[:, 'MACD'] = macd
    dataframe.loc[:, 'MACD_signal'] = macdsignal
    dataframe.loc[:, 'MACD_hist'] = macdhist
    
    upper, middle, lower = talib.BBANDS(close_prices, timeperiod=indicators_config['bbands_length'], nbdevup=indicators_config['bbands_std'], nbdevdn=indicators_config['bbands_std'], matype=0)
    
    dataframe.loc[:, 'BB_upper'] = upper    
    dataframe.loc[:, 'BB_middle'] = middle
    dataframe.loc[:, 'BB_lower'] = lower
    
    dataframe.loc[:, 'ADX'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=indicators_config['adx_length'])
    
    dataframe = dataframe.dropna()
    return dataframe

def create_wavelet_features(dataframe, wavelet_config):
    """
    Cria as features wavelet utilizando a biblioteca pywt.
    
    Args:
        dataframe: DataFrame com os dados provenientes do CSV
        wavelet_config: Dicionário com as configurações das features wavelet
        
    Returns:
        DataFrame com as features wavelet criadas
    """
    
    coeffs = pywt.wavedec(dataframe['Close'].values, wavelet=wavelet_config['family'], level=wavelet_config['level'], mode='symmetric')
    cA, cD = coeffs
    
    cA_series = pd.Series(np.nan, index=dataframe.index)
    cD_series = pd.Series(np.nan, index=dataframe.index)
    cA_series.iloc[-len(cA):] = cA
    cD_series.iloc[-len(cD):] = cD
    cA_series.bfill(inplace=True)
    cD_series.bfill(inplace=True)
    
    dataframe.loc[:, 'wavelet_cA'] = cA_series
    dataframe.loc[:, 'wavelet_cD'] = cD_series
    
    dataframe = dataframe.dropna()
    return dataframe

def main():
    
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    indicators_config = config["features"]["indicators"]
    wavelet_config = config["features"]["wavelet"]
    
    for ticker in data_config["tickers"]:
        print(f"Criando features para {ticker}...")
        
        data = pd.read_csv(f'{data_config["raw_data_path"]}/{ticker}.csv')
        
        data = create_technical_indicators(data, indicators_config)
        
        data = create_wavelet_features(data, wavelet_config)
        data.to_csv(f'{data_config["features_data_path"]}/{ticker}.csv', index=False)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    