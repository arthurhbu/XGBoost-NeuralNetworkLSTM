import pandas as pd
from bcb import sgs

def simulação_renda_fixa(initial_capital, start_date, end_date):

        cdi_series_code = 12

        try:
            cdi_daily = sgs.get({'cdi': cdi_series_code}, start=start_date, end=end_date)

            cdi_daily['cdi'] = cdi_daily['cdi'] / 100
        except Exception as e:
            print(f"Erro ao obter os dados do CDI: {e}")
            return None
        
        portfolio_value = initial_capital
        portfolio_value_history = [portfolio_value]

        


    