import pandas as pd
from pathlib import Path
import yaml
import talib
import numpy as np
import pywt

def clean_and_convert_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza dados OHLCV para engenharia de features.

    Converte as colunas numéricas (``Close``, ``High``, ``Low``, ``Open``, ``Volume``)
    para valores numéricos, remove linhas inválidas e garante o dtype ``float64``.

    Args:
        dataframe: DataFrame de entrada contendo ao menos as colunas OHLCV.

    Returns:
        Um novo ``DataFrame`` com colunas numéricas convertidas para ``float64``
        e sem linhas com valores inválidos.

    Raises:
        TypeError: Se ``dataframe`` não for um ``pd.DataFrame``.
        ValueError: Se faltarem colunas obrigatórias do conjunto OHLCV.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    required_columns = {'Close', 'High', 'Low', 'Open', 'Volume'}
    missing_columns = required_columns.difference(set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = dataframe.copy()

    # Coerção numérica com NaN para valores inválidos
    for column_name in required_columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')

    # Remoção de linhas com NaN após coerção
    df = df.dropna(subset=list(required_columns))

    # Padronização do dtype
    for column_name in required_columns:
        df[column_name] = df[column_name].astype('float64')

    return df


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
    volume = dataframe['Volume'].values

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
    
    dataframe.loc[:, 'ATR'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=indicators_config['atr_length'])

    dataframe.loc[:, 'OBV'] = talib.OBV(close_prices, volume)

    dataframe.loc[:, 'MFI'] = talib.MFI(high_prices, low_prices, close_prices, volume, timeperiod=indicators_config['mfi_length'])
    
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
    close_prices = dataframe['Close'].values

    cA_list = []
    cD_list = []
    min_length = 2 ** (wavelet_config['level'] + 2)
    min_length = max(min_length, 30)
    
    for i in range(len(close_prices)):
        if i < min_length:
            cA_list.append(np.nan)
            cD_list.append(np.nan)
        else: 
            window_data = close_prices[:i+1]
            try:
                coeffs = pywt.wavedec(
                    window_data,
                    wavelet=wavelet_config['family'],
                    level=wavelet_config['level'],
                    mode='constant'
                )
                cA, cD = coeffs
                cA_list.append(cA[-1])
                cD_list.append(cD[-1])
            except Exception as e:
                print(e)
                cA_list.append(np.nan)
                cD_list.append(np.nan)

    cA_series = pd.Series(cA_list, index=dataframe.index)
    cD_series = pd.Series(cD_list, index=dataframe.index)

    dataframe.loc[:, 'wavelet_cA'] = cA_series
    dataframe.loc[:, 'wavelet_cD'] = cD_series

    return dataframe.dropna()

def main():
    
    config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    
    new_header = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    data_config = config["data"]
    indicators_config = config["features"]["indicators"]
    wavelet_config = config["features"]["wavelet"]
    
    for ticker in data_config["tickers"]:
        print(f"Criando features para {ticker}...")
        
        data = pd.read_csv(f'{data_config["raw_data_path"]}/{ticker}.csv', skiprows=3, names=new_header, index_col=0)
        
        data = clean_and_convert_data(data)
        
        data = create_technical_indicators(data, indicators_config)
        
        data = create_wavelet_features(data, wavelet_config)

        # Salvar o CSV com o índice (datas) incluído
        data.to_csv(f'{data_config["features_data_path"]}/{ticker}.csv')

if __name__ == "__main__":
    main()