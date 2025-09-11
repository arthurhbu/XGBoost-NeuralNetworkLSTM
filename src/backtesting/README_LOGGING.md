# Sistema de Logging para Backtesting

## 🎯 **OBJETIVO**
Substituir prints confusos no console por logs estruturados em arquivos para facilitar debugging de múltiplos modelos.

## 📁 **ESTRUTURA DOS LOGS**

### **Diretório de Logs**
```
reports/
├── logs/
│   ├── ABEV3.SA_20250829_162051.log
│   ├── B3SA3.SA_20250829_162052.log
│   ├── BBAS3.SA_20250829_162053.log
│   └── ...
├── results_simulated.csv
├── results_simulated.txt
└── results_simulated.json
```

### **Formato dos Logs**
```
2025-08-29 16:20:51,123 - INFO - ============================================================
2025-08-29 16:20:51,124 - INFO - INICIANDO BACKTESTING
2025-08-29 16:20:51,125 - INFO - ============================================================
2025-08-29 16:20:51,126 - INFO - Tamanho test_df: 252
2025-08-29 16:20:51,127 - INFO - Tamanho predictions: 252
2025-08-29 16:20:51,128 - INFO - Capital inicial: R$ 100,000.00
2025-08-29 16:20:51,129 - INFO - Custo de transação: 0.100%
2025-08-29 16:20:51,130 - INFO - DISTRIBUIÇÃO DAS PREDIÇÕES:
2025-08-29 16:20:51,131 - INFO -   - Total: 252
2025-08-29 16:20:51,132 - INFO -   - Predições de alta (1): 45
2025-08-29 16:20:51,133 - INFO -   - Predições de baixa (0): 207
2025-08-29 16:20:51,134 - INFO -   - Percentual de alta: 17.9%
2025-08-29 16:20:51,135 - INFO - EXECUÇÃO DOS TRADES:
2025-08-29 16:20:51,136 - INFO - PROGRESSO: Data 2024-01-02 | Pred: 0 | Cash: R$ 100,000.00 | Ações: 0
2025-08-29 16:20:51,137 - INFO - COMPRA: 2024-01-03 | Preço: R$ 15.20 | Ações: 6,578 | Custo: R$ 100,000.00
2025-08-29 16:20:51,138 - INFO - VENDA: 2024-01-04 | Preço: R$ 14.80 | Ações: 6,578 | Valor líquido: R$ 97,354.40
...
```

## 🔍 **COMO USAR**

### **1. Executar Backtesting**
```bash
cd src/backtesting
python backtest.py
```

### **2. Console Limpo**
```
=== CONFIGURAÇÃO DE BACKTESTING ===
Capital inicial: R$ 100,000.00
Custo de transação: 0.100%
Data inicial simulação: 2024-01-01
Data final simulação: 2025-01-01
Período de simulação: 2024-01-01 até 2025-01-01

PROCESSANDO: ABEV3.SA
  📊 Dados: 2017-01-03 até 2025-08-18 (2156 registros)
  ✅ Modelo carregado
  📈 Simulação: 252 registros
  🤖 Predições: 45 alta, 207 baixa
  📝 Log salvo em: ABEV3.SA_20250829_162051.log
  📊 Resultado: Modelo +12.45% | Buy&Hold -9.65%

PROCESSANDO: B3SA3.SA
  📊 Dados: 2017-01-03 até 2025-08-18 (2156 registros)
  ✅ Modelo carregado
  📈 Simulação: 252 registros
  🤖 Predições: 38 alta, 214 baixa
  📝 Log salvo em: B3SA3.SA_20250829_162052.log
  📊 Resultado: Modelo -8.23% | Buy&Hold -25.22%
```

### **3. Analisar Logs Detalhados**
```bash
# Ver log de um ticker específico
cat reports/logs/ABEV3.SA_20250829_162051.log

# Procurar por problemas específicos
grep "AVISO" reports/logs/ABEV3.SA_20250829_162051.log
grep "ERRO" reports/logs/ABEV3.SA_20250829_162051.log
grep "COMPRA" reports/logs/ABEV3.SA_20250829_162051.log
grep "VENDA" reports/logs/ABEV3.SA_20250829_162051.log
```

## 📊 **INFORMAÇÕES NOS LOGS**

### **Dados de Entrada**
- Tamanho dos DataFrames
- Distribuição das predições
- Configurações de capital e custos

### **Execução dos Trades**
- Progresso a cada 50 iterações
- Detalhes de cada compra/venda
- Preços, quantidades e custos

### **Resumo Final**
- Total de trades executados
- Compras vs vendas realizadas
- Capital inicial vs final
- Retorno total calculado

### **Alertas e Erros**
- Problemas de alinhamento de dados
- Compras/vendas rejeitadas
- Avisos sobre predições extremas

## 🚀 **VANTAGENS DO NOVO SISTEMA**

1. **Console Limpo**: Apenas informações essenciais
2. **Logs Estruturados**: Fácil de analisar e buscar
3. **Histórico Completo**: Cada execução gera log único
4. **Debugging Eficiente**: Problemas ficam registrados
5. **Análise Posterior**: Pode revisar logs antigos
6. **Múltiplos Modelos**: Cada ticker tem seu log

## 🔧 **CONFIGURAÇÃO**

### **Diretório de Logs**
Os logs são salvos automaticamente em:
```
{results_path}/logs/
```

### **Formato dos Nomes**
```
{ticker}_{YYYYMMDD}_{HHMMSS}.log
```

### **Níveis de Log**
- **INFO**: Informações gerais e progresso
- **WARNING**: Avisos e problemas menores
- **ERROR**: Erros críticos (se implementados)

## 📝 **EXEMPLO DE ANÁLISE**

### **Problema: Nenhum Trade Executado**
```bash
# Verificar se há predições de alta
grep "Predições de alta" reports/logs/*.log

# Verificar se há avisos
grep "AVISO" reports/logs/*.log

# Verificar distribuição das predições
grep "Distribuição" reports/logs/*.log
```

### **Problema: Performance Ruim**
```bash
# Verificar trades executados
grep "COMPRA\|VENDA" reports/logs/*.log

# Verificar resumo final
grep "RESUMO FINAL" reports/logs/*.log
```

### **Problema: Dados Insuficientes**
```bash
# Verificar tamanho dos DataFrames
grep "Tamanho test_df" reports/logs/*.log

# Verificar período de simulação
grep "Simulação:" reports/logs/*.log
```
