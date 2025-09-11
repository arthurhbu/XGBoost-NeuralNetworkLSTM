import pandas as pd
from bcb import sgs
import os
from datetime import datetime


def _load_or_fetch_cdi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Carrega ou busca os dados do CDI e já os converte para a taxa diária efetiva.
    """
    cdi_series_code = 12
    csv_path = 'src/renda_fixa_simulação/cdi_daily_rate.csv' # Renomeie para clareza

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if os.path.exists(csv_path):
        try:
            cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if not cached.empty and cached.index.min() <= start_dt and cached.index.max() >= end_dt:
                # Retorna a fatia do cache que já está no formato correto (taxa diária)
                return cached.loc[(cached.index >= start_dt) & (cached.index <= end_dt)].copy()
        except Exception:
            pass # Se o cache falhar, busca novamente

    # Se precisar buscar na API
    cdi_fetched = sgs.get({'cdi': cdi_series_code}, start=start_date, end=end_date)
    if cdi_fetched.empty:
        return cdi_fetched

    # Lógica de conversão centralizada
    median_val = float(cdi_fetched['cdi'].median())

    if median_val > 1.0:
        # Caso 1: API retorna taxa anualizada em % (ex: 11.5)
        # Converte para fração (0.115) e depois para taxa diária
        annual_rates = cdi_fetched['cdi'] / 100.0
        daily_rates = annual_rates.apply(_daily_from_annual)
    else:
        # Caso 2: API retorna taxa diária em % (ex: 0.043)
        # Apenas converte para fração (0.00043)
        daily_rates = cdi_fetched['cdi'] / 100.0
    
    cdi_daily_df = pd.DataFrame(daily_rates)
    
    # Salva o cache já com a taxa diária correta
    cdi_daily_df.to_csv(csv_path)

    return cdi_daily_df


def _daily_from_annual(annual_rate: float, business_days: int = 252) -> float:
	# Converte taxa anual efetiva para taxa diária efetiva
	return (1.0 + annual_rate) ** (1.0 / business_days) - 1.0


def _custody_or_admin_daily_from_annual(annual_rate: float) -> float:
	# Taxas pequenas: usar conversão efetiva diária para consistência
	return _daily_from_annual(annual_rate)


def _iof_rate_by_day(holding_days: int) -> float:
	# Tabela IOF regressiva (dias 1 a 30). Valores representam % do rendimento que vira IOF.
	# Fonte: Receita/BACEN (padrão de mercado). 30 dias => 0.
	iof_table = {
		1: 0.96, 2: 0.93, 3: 0.90, 4: 0.86, 5: 0.83,
		6: 0.80, 7: 0.76, 8: 0.73, 9: 0.70, 10: 0.66,
		11: 0.63, 12: 0.60, 13: 0.56, 14: 0.53, 15: 0.50,
		16: 0.46, 17: 0.43, 18: 0.40, 19: 0.36, 20: 0.33,
		21: 0.30, 22: 0.26, 23: 0.23, 24: 0.20, 25: 0.16,
		26: 0.13, 27: 0.10, 28: 0.06, 29: 0.03, 30: 0.00,
	}
	if holding_days <= 0:
		return 1.0
	if holding_days >= 30:
		return 0.0
	return iof_table.get(holding_days, 0.0)


def _ir_rate_by_day(holding_days: int) -> float:
	# IR regressivo sobre rendimentos
	if holding_days <= 180:
		return 0.225
	elif holding_days <= 360:
		return 0.20
	elif holding_days <= 720:
		return 0.175
	else:
		return 0.15


def simulação_renda_fixa(
    initial_capital: float,
    start_date: str,
    end_date: str,
    custody_rate_aa: float = 0.002,  # 0,20% a.a. B3 (Tesouro Direto)
    admin_rate_aa: float = 0.0,      # taxa de administração da corretora, se houver
) -> pd.DataFrame:

    # 1. Carrega os dados do CDI, que já virão como taxa diária efetiva e fracionária
    cdi_daily_df = _load_or_fetch_cdi(start_date, end_date)
    
    # A heurística de verificação foi removida, pois a função acima já garante o formato correto.
    gross_daily_rates = cdi_daily_df['cdi']

    # 2. Calcula taxas diárias de custódia e administração
    custody_daily_rate = _custody_or_admin_daily_from_annual(custody_rate_aa)
    admin_daily_rate = _custody_or_admin_daily_from_annual(admin_rate_aa)

    gross_values = []
    values_after_fees = []

    gross_value = initial_capital
    value_after_fees = initial_capital

    for i, daily_rate in enumerate(gross_daily_rates):
        gross_value = gross_value * (1.0 + daily_rate)
        
        # --- CÁLCULO DE TAXAS CORRIGIDO ---
        # Combina o rendimento com a dedução das taxas em um único fator diário.
        # Esta é a forma matematicamente mais precisa.
        net_daily_factor = (1.0 + daily_rate - custody_daily_rate - admin_daily_rate)
        value_after_fees *= net_daily_factor

        gross_values.append(gross_value)
        values_after_fees.append(value_after_fees)

    index = cdi_daily_df.index
    gross_series = pd.Series(gross_values, index=index, name='gross')
    fees_series = pd.Series(values_after_fees, index=index, name='after_fees')

    # 3. Série após todos os impostos (lógica de IR e IOF já estava correta)
    net_after_all = []
    for day_idx, date in enumerate(index):
        holding_days = day_idx + 1
        value_before_tax = fees_series.iloc[day_idx]
        profit = max(0.0, value_before_tax - initial_capital)
        
        if profit <= 0:
            net_after_all.append(value_before_tax)
            continue
            
        ir_rate = _ir_rate_by_day(holding_days)
        iof_rate = _iof_rate_by_day(holding_days)
        
        iof_tax = profit * iof_rate
        base_ir = max(0.0, profit - iof_tax)
        ir_tax = base_ir * ir_rate
        
        total_tax = iof_tax + ir_tax
        net_after_all.append(value_before_tax - total_tax)

    net_series = pd.Series(net_after_all, index=index, name='after_all_taxes')

    result = pd.concat([gross_series, fees_series, net_series], axis=1)
    return result

def main():
	initial_capital = 100000
	start_date = '2024-01-01'
	end_date = '2025-01-01'
	result = simulação_renda_fixa(initial_capital, start_date, end_date)
	# Exporta todas as séries
	result.to_csv('src/renda_fixa_simulação/simulação_renda_fixa.csv')


if __name__ == '__main__':
	main()
	 