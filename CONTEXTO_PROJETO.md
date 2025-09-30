# Contexto Completo do Projeto - XGBoost Trading Model

## 📋 RESUMO EXECUTIVO

**Problema Principal**: Modelo XGBoost com baixos resultados financeiros e métricas ruins, especialmente poucos trades e retornos baixos. O modelo apresenta desbalanceamento severo de classes, calibração ruim de probabilidades e possíveis vazamentos de informação.

**Status Atual**: ETAPA 3 concluída - Ajuste de targets (triple barrier) e critério de classe "Up"

**Próximo Passo Crítico**: ETAPA 4 - Calibração de probabilidades (Platt/Isotônica)

**Performance Atual**: PETR4.SA +22.46% (melhor ticker), mas ainda abaixo do potencial identificado na ETAPA 1 (+28.70%)

---

## 🎯 OBJETIVOS ESTRATÉGICOS

### Objetivo Principal
Criar um modelo de trading que supere consistentemente o Buy&Hold em múltiplos tickers, mantendo Sharpe > 1.0 e drawdown < 15%.

### Objetivos Específicos
1. **Gerar trades consistentes**: Eliminar tickers com 0 trades (ITUB4.SA)
2. **Melhorar calibração**: Reduzir Brier score de 0.207 (PETR4) para < 0.1
3. **Aumentar precisão "Up"**: De 41.75% (PETR4) para > 60%
4. **Reduzir ruído**: Manter hold_min_days=5, evitar over-trading
5. **Escalabilidade**: Funcionar em pelo menos 6 dos 8 tickers testados

---

## 🔍 DIAGNÓSTICOS REALIZADOS - ANÁLISE DETALHADA

### 1. Distribuição de Rótulos (2024) - Análise por Trimestre

#### PETR4.SA (MELHOR TICKER)
- **2024 Total**: 31.9% "Up", 18.7% "Neutro", 49.4% "Down"
- **Q1 2024**: 31.1% "Up" - bom equilíbrio
- **Q2 2024**: 47.6% "Up" - excelente (melhor trimestre)
- **Q3 2024**: 22.7% "Up" - moderado
- **Q4 2024**: 26.2% "Up" - moderado
- **Conclusão**: Classes relativamente equilibradas, explica boa performance

#### VALE3.SA (PROBLEMÁTICO)
- **2024 Total**: 12.4% "Up", 8.8% "Neutro", 78.9% "Down"
- **Q1 2024**: 9.8% "Up" - muito baixo
- **Q2 2024**: 19.0% "Up" - melhor trimestre
- **Q3 2024**: 13.6% "Up" - baixo
- **Q4 2024**: 6.6% "Up" - crítico
- **Conclusão**: Desbalanceamento severo, regime de baixa persistente

#### ITUB4.SA (CRÍTICO)
- **2024 Total**: 6.0% "Up", 37.1% "Neutro", 57.0% "Down"
- **Q1 2024**: 11.5% "Up" - baixo
- **Q2 2024**: 1.6% "Up" - crítico
- **Q3 2024**: 10.6% "Up" - baixo
- **Q4 2024**: 0.0% "Up" - ZERO trades possíveis
- **Conclusão**: Target muito agressivo para o regime, necessita ajuste urgente

#### ABEV3.SA (VOLÁTIL)
- **2024 Total**: 13.1% "Up", 24.7% "Neutro", 62.2% "Down"
- **Q1 2024**: 0.0% "Up" - ZERO (crítico)
- **Q2 2024**: 0.0% "Up" - ZERO (crítico)
- **Q3 2024**: 31.8% "Up" - bom (recuperação)
- **Q4 2024**: 19.7% "Up" - moderado
- **Conclusão**: Sazonalidade extrema, Q1-Q2 impossíveis de operar

### 2. AUC-PR para classe "Up" - Análise de Qualidade do Sinal

#### Excelente Qualidade (AUC-PR > 0.9)
- **VALE3.SA**: 0.999 - Ranking perfeito, mas "Up" raríssimo
- **BBDC4.SA**: 0.988 - Excelente ranking, "Up" moderado
- **ITUB4.SA**: 0.999 - Ranking perfeito, mas "Up" inexistente

#### Boa Qualidade (AUC-PR 0.6-0.9)
- **BBAS3.SA**: 0.876 - Boa qualidade, aproveitável
- **PETR4.SA**: 0.688 - Moderada, mas classes equilibradas
- **ABEV3.SA**: 0.759 - Boa, mas sazonalidade extrema

#### Qualidade Ruim (AUC-PR < 0.6)
- **VIVT3.SA**: 0.451 - Baixa qualidade, sinais ruidosos
- **B3SA3.SA**: 0.742 - Moderada, mas calibração ruim

### 3. Calibração de Probabilidades (Brier Score) - Análise de Confiabilidade

#### Bem Calibrado (Brier < 0.1)
- **VALE3.SA**: 0.041 - Excelente calibração
- **ITUB4.SA**: 0.031 - Excelente calibração
- **BBDC4.SA**: 0.074 - Boa calibração

#### Mal Calibrado (Brier > 0.1)
- **PETR4.SA**: 0.207 - Calibração ruim (principal problema)
- **B3SA3.SA**: 0.194 - Calibração ruim
- **ABEV3.SA**: 0.108 - Calibração moderada
- **VIVT3.SA**: 0.104 - Calibração moderada

**Insight Crítico**: PETR4.SA tem boa AUC-PR (0.688) mas calibração ruim (0.207). Isso significa que o modelo rankeia bem, mas as probabilidades não são confiáveis para thresholding.

### 4. Alinhamento p_up vs Retornos - Análise de Leakage

#### Correlação Next Day (Esperada: 0.2-0.5)
- **VALE3.SA**: 0.283 - Boa correlação
- **BBDC4.SA**: 0.241 - Boa correlação
- **ITUB4.SA**: 0.249 - Boa correlação
- **BBAS3.SA**: 0.170 - Moderada
- **PETR4.SA**: 0.027 - Muito baixa (problema)
- **VIVT3.SA**: 0.021 - Muito baixa (problema)

#### Correlação Same Day (Esperada: < 0.1)
- **PETR4.SA**: 0.551 - ALTA (possível leakage)
- **BBAS3.SA**: 0.554 - ALTA (possível leakage)
- **B3SA3.SA**: 0.534 - ALTA (possível leakage)
- **VALE3.SA**: 0.456 - Moderada
- **BBDC4.SA**: 0.352 - Moderada
- **ITUB4.SA**: 0.210 - Baixa

**Insight Crítico**: PETR4.SA tem correlação alta com mesma barra (0.551), indicando possível vazamento de informação. Isso pode explicar por que funciona bem mas não generaliza.

---

## 🚀 ETAPAS EXECUTADAS - ANÁLISE DETALHADA

### ✅ ETAPA 1: Ampliar Grid de Thresholds (SUCESSO)
**Modificação Técnica**: 
- `coarse_buy_grid`: 0.20-0.60 → 0.10-0.80 (passo 0.10)
- `coarse_sell_grid`: -0.30-0.10 → -0.50-0.20 (passo 0.10)
- `fine_buy_grid`: ±0.08 → ±0.10 (refinamento)

**Resultados Financeiros**:
- **PETR4.SA**: +28.70% (vs +16.44% baseline) - **+12.26 p.p.**
- **Sharpe Ratio**: 2.097 (vs 1.047) - **+100%**
- **Max Drawdown**: 7.53% (vs 11.18%) - **-3.65 p.p.**
- **Volatilidade**: 12.56% (vs 15.96%) - **-3.40 p.p.**

**Análise Técnica**:
- Encontrou thresholds ótimos fora da faixa original
- Reduziu risco significativamente
- Confirmou que havia sinal aproveitável (AUC-PR alta se converteu em PnL)

**Lições Aprendidas**:
- Grid original (0.2-0.6) era muito restritivo
- Faixa ampliada capturou regimes diferentes por ticker
- Histerese natural (buy > sell) funcionou bem

### ❌ ETAPA 2: Reduzir hold_min_days (FALHA TOTAL)
**Modificação Técnica**: 
- `hold_min_days`: 5 → 1 (no config.yaml)

**Resultados Financeiros**:
- **PETR4.SA**: +3.40% (vs +28.70% ETAPA 1) - **-25.30 p.p.**
- **Sharpe Ratio**: 0.272 (vs 2.097) - **-87%**
- **Max Drawdown**: 17.69% (vs 7.53%) - **+10.16 p.p.**
- **Volatilidade**: 19.32% (vs 12.56%) - **+6.76 p.p.**
- **Win Rate**: 46.55% (vs 56.52%) - **-9.97 p.p.**

**Análise Técnica**:
- Over-trading severo: mais trades = mais custos
- Ruído de curto prazo dominou sinais de tendência
- Custos de transação (0.1%) acumularam rapidamente
- Histerese temporal é crítica para filtrar ruído

**Lições Aprendidas**:
- hold_min_days=1 é muito agressivo para regime atual
- Custos de transação são críticos em alta frequência
- Sinais de 1 dia capturam muito ruído, não tendências
- Histerese temporal (5 dias) é necessária

### ⚠️ ETAPA 3: Ajustar Targets (SUCESSO PARCIAL)
**Modificações Técnicas**:
- `up_class_ratio < 0.02` → `up_class_ratio < 0.10` (train_models.py:222)
- Adicionados 5 targets mais brandos no triple_barrier_grid:
  - `{holding_days: 7, profit_threshold: 0.015, loss_threshold: -0.01}`
  - `{holding_days: 10, profit_threshold: 0.02, loss_threshold: -0.015}`
  - `{holding_days: 5, profit_threshold: 0.025, loss_threshold: -0.02}`
  - `{holding_days: 7, profit_threshold: 0.03, loss_threshold: -0.025}`
  - `{holding_days: 10, profit_threshold: 0.035, loss_threshold: -0.02}`

**Resultados Financeiros**:
- **PETR4.SA**: +22.46% (vs +28.70% ETAPA 1) - **-6.24 p.p.**
- **Sharpe Ratio**: 1.049 (vs 2.097) - **-50%**
- **Max Drawdown**: 16.01% (vs 7.53%) - **+8.48 p.p.**
- **Volatilidade**: 21.83% (vs 12.56%) - **+9.27 p.p.**

**Análise Técnica**:
- **SUCESSO**: Mais predições "Up" (115→230 em PETR4, +100%)
- **PROBLEMA**: Qualidade dos sinais piorou (mais ruído)
- **CRÍTICO**: BBAS3/VALE3 perderam TODAS as predições "Up" (0 predições)
- **PERSISTENTE**: ITUB4.SA ainda 0 predições

**Lições Aprendidas**:
- Targets mais brandos geram mais "Up" mas pior qualidade
- Critério 0.10 permitiu mais estratégias, mas algumas ruins
- Qualidade > Quantidade: melhor ter menos sinais mas melhores
- Alguns tickers precisam de targets específicos por regime

**Status**: Parcialmente bem-sucedida - aumentou diversidade mas piorou qualidade

---

## 📊 RESULTADOS POR ETAPA - ANÁLISE COMPARATIVA

### Tabela de Performance Financeira

| Ticker | ETAPA 0 (Baseline) | ETAPA 1 (Grid) | ETAPA 2 (hold=1) | ETAPA 3 (Targets) | Melhor Etapa | Status |
|--------|-------------------|----------------|------------------|-------------------|--------------|---------|
| **PETR4.SA** | +16.44% | **+28.70%** | +3.40% | +22.46% | **ETAPA 1** | ✅ Melhor |
| **BBDC4.SA** | +3.44% | +3.44% | +3.44% | +3.44% | **Todas** | ➖ Estável |
| **BBAS3.SA** | -2.23% | -2.23% | -2.23% | -2.23% | **Todas** | ➖ Estável |
| **B3SA3.SA** | -0.54% | -0.54% | -0.54% | -0.54% | **Todas** | ➖ Estável |
| **VALE3.SA** | -10.77% | -10.77% | -10.77% | -11.78% | **ETAPA 0-2** | ❌ Piorou |
| **ITUB4.SA** | 0.00% | 0.00% | 0.00% | 0.00% | **Todas** | ❌ Crítico |
| **ABEV3.SA** | -0.55% | -0.55% | -0.55% | -0.55% | **Todas** | ➖ Estável |
| **VIVT3.SA** | +1.94% | +1.94% | +1.94% | +1.94% | **Todas** | ➖ Estável |

### Análise de Sharpe Ratio

| Ticker | ETAPA 0 | ETAPA 1 | ETAPA 2 | ETAPA 3 | Melhor | Status |
|--------|---------|---------|---------|---------|---------|---------|
| **PETR4.SA** | 1.047 | **2.097** | 0.272 | 1.049 | **ETAPA 1** | ✅ Melhor |
| **BBDC4.SA** | 0.269 | 0.269 | 0.269 | 0.269 | **Todas** | ➖ Estável |
| **BBAS3.SA** | -0.426 | -0.426 | -0.426 | -0.426 | **Todas** | ❌ Ruim |
| **B3SA3.SA** | 0.009 | 0.009 | 0.009 | 0.009 | **Todas** | ❌ Ruim |
| **VALE3.SA** | -0.959 | -0.959 | -0.959 | -1.035 | **ETAPA 0-2** | ❌ Piorou |
| **ITUB4.SA** | -inf | -inf | -inf | -inf | **Todas** | ❌ Crítico |
| **ABEV3.SA** | 0.007 | 0.007 | 0.007 | 0.007 | **Todas** | ❌ Ruim |
| **VIVT3.SA** | 0.769 | 0.769 | 0.769 | 0.769 | **Todas** | ✅ Bom |

### Análise de Predições "Up"

| Ticker | ETAPA 0 | ETAPA 1 | ETAPA 2 | ETAPA 3 | Mudança | Status |
|--------|---------|---------|---------|---------|---------|---------|
| **PETR4.SA** | 118 | 115 | 174 | 230 | **+95%** | ✅ Melhorou |
| **BBDC4.SA** | 51 | 100 | 147 | 136 | **+167%** | ✅ Melhorou |
| **BBAS3.SA** | 33 | 40 | 50 | 0 | **-100%** | ❌ Perdeu |
| **B3SA3.SA** | 43 | 64 | 39 | 0 | **-100%** | ❌ Perdeu |
| **VALE3.SA** | 38 | 26 | 18 | 0 | **-100%** | ❌ Perdeu |
| **ITUB4.SA** | 0 | 0 | 0 | 0 | **0%** | ❌ Crítico |
| **ABEV3.SA** | 34 | 40 | 50 | 0 | **-100%** | ❌ Perdeu |
| **VIVT3.SA** | 38 | 26 | 18 | 26 | **-32%** | ❌ Piorou |

### Insights Críticos

1. **PETR4.SA é o único ticker funcional**: Apenas ele responde consistentemente às melhorias
2. **ETAPA 1 foi o pico de performance**: +28.70% com Sharpe 2.097
3. **ETAPA 2 foi desastrosa**: hold_min_days=1 destruiu performance
4. **ETAPA 3 aumentou sinais mas piorou qualidade**: Mais "Up" mas pior Sharpe
5. **ITUB4.SA é intratável**: 0 trades em todas as etapas
6. **VALE3.SA/BBAS3.SA perderam sinais na ETAPA 3**: Targets muito brandos os prejudicaram

---

## 🎯 PRÓXIMAS ETAPAS RECOMENDADAS - ROADMAP DETALHADO

### 🚨 ETAPA 4: Calibração de Probabilidades (CRÍTICA)
**Objetivo**: Melhorar calibração das probabilidades para reduzir ruído e melhorar thresholding
**Prioridade**: ALTA - Principal gargalo identificado

**Problema Atual**:
- PETR4.SA: Brier 0.207 (ruim) vs AUC-PR 0.688 (boa)
- B3SA3.SA: Brier 0.194 (ruim) vs AUC-PR 0.742 (boa)
- Probabilidades mal calibradas = thresholds ineficazes

**Modificações Técnicas**:
1. **Implementar calibração Platt/Isotônica por ticker**:
   - Treinar calibrador na validação (2022-2023)
   - Aplicar no teste (2024)
   - Usar `sklearn.calibration.CalibratedClassifierCV`

2. **Integrar calibração no pipeline**:
   - Modificar `adaptive_threshold_optimization` para usar probabilidades calibradas
   - Reotimizar thresholds após calibração
   - Manter score original para simulação simples

3. **Avaliar melhoria**:
   - Brier score < 0.1 (atual: 0.207)
   - Melhor correlação p_up vs retornos next day
   - Thresholds mais estáveis entre validação/teste

**Resultado Esperado**: PETR4.SA retorna para +28% com Sharpe > 2.0

### 🔍 ETAPA 5: Checagem de Leakage (IMPORTANTE)
**Objetivo**: Verificar e corrigir vazamento de informação da mesma barra
**Prioridade**: MÉDIA-ALTA - Pode explicar overfitting

**Problema Atual**:
- PETR4.SA: correlação same day 0.551 (muito alta)
- BBAS3.SA: correlação same day 0.554 (muito alta)
- B3SA3.SA: correlação same day 0.534 (muito alta)

**Modificações Técnicas**:
1. **Auditoria de features** (`src/features/build_features.py`):
   - Verificar se todas as features usam dados ≤ t-1
   - Identificar features que podem incorporar Close_t
   - Revisar janelas deslizantes e agregações

2. **Teste de defasagem**:
   - Defasar todas as features em 1 barra
   - Comparar performance com/sem defasagem
   - Manter apenas features que não pioram performance

3. **Validação temporal**:
   - Treinar em 2017-2020, validar em 2021-2022, testar em 2024
   - Verificar se performance se mantém out-of-sample

**Resultado Esperado**: Correlação same day < 0.1, melhor generalização

### 🎯 ETAPA 6: Política por Ticker (ESTRATÉGICA)
**Objetivo**: Aplicar estratégias diferentes por ticker baseado em performance e características
**Prioridade**: MÉDIA - Otimização final

**Estratégias por Ticker**:

#### PETR4.SA (TICKER PRINCIPAL)
- **Status**: Funcional, melhor performance
- **Estratégia**: Operação normal com thresholds otimizados
- **Foco**: Manter performance atual, melhorar calibração

#### BBDC4.SA (TICKER SECUNDÁRIO)
- **Status**: Estável, boa AUC-PR (0.988)
- **Estratégia**: Operação normal, targets moderados
- **Foco**: Manter estabilidade, evitar over-trading

#### VIVT3.SA (TICKER CONSERVADOR)
- **Status**: Boa calibração, AUC-PR baixa (0.451)
- **Estratégia**: Operação conservadora, thresholds altos
- **Foco**: Reduzir ruído, priorizar precisão

#### VALE3.SA/BBAS3.SA/B3SA3.SA (TICKERS PROBLEMÁTICOS)
- **Status**: Performance ruim, perderam sinais na ETAPA 3
- **Estratégia**: Targets específicos por ticker, não operar se necessário
- **Foco**: Investigar causas raiz, ajustar targets individualmente

#### ITUB4.SA (TICKER CRÍTICO)
- **Status**: 0 trades em todas as etapas
- **Estratégia**: Targets extremamente brandos ou não operar
- **Foco**: Investigar se é viável operar este ticker

### 🔧 ETAPA 7: Otimização Final (OPCIONAL)
**Objetivo**: Fine-tuning de parâmetros e validação final
**Prioridade**: BAIXA - Após resolver problemas principais

**Modificações**:
1. **Cross-validation temporal** com múltiplos folds
2. **Ensemble de modelos** por ticker
3. **Otimização de position sizing** baseada em volatilidade
4. **Validação walk-forward** para robustez

---

## 🚨 PROBLEMAS CRÍTICOS IDENTIFICADOS

### 1. Calibração Ruim (CRÍTICO)
- **PETR4.SA**: Brier 0.207 vs AUC-PR 0.688
- **Causa**: Probabilidades não refletem confiança real
- **Impacto**: Thresholds ineficazes, performance subótima
- **Solução**: ETAPA 4 - Calibração Platt/Isotônica

### 2. Possível Leakage (IMPORTANTE)
- **PETR4.SA**: Correlação same day 0.551
- **Causa**: Features podem usar informação da mesma barra
- **Impacto**: Overfitting, performance não generaliza
- **Solução**: ETAPA 5 - Auditoria e defasagem de features

### 3. Desbalanceamento Severo (ESTRUTURAL)
- **ITUB4.SA**: 0% "Up" em Q4 2024
- **VALE3.SA**: 6.6% "Up" em Q4 2024
- **Causa**: Targets muito agressivos para regime 2024
- **Impacto**: Impossível gerar trades em alguns tickers
- **Solução**: Targets adaptativos por ticker/regime

### 4. Over-trading (RESOLVIDO)
- **ETAPA 2**: hold_min_days=1 destruiu performance
- **Causa**: Custos de transação dominaram benefícios
- **Impacto**: -25 p.p. de retorno em PETR4.SA
- **Solução**: Manter hold_min_days=5 (já implementado)

### 5. Qualidade vs Quantidade (PARCIALMENTE RESOLVIDO)
- **ETAPA 3**: Mais "Up" mas pior qualidade
- **Causa**: Targets muito brandos geraram ruído
- **Impacto**: Sharpe caiu de 2.097 para 1.049
- **Solução**: Calibração + targets intermediários

---

## 📁 ARQUIVOS IMPORTANTES - MAPA COMPLETO

### 🔧 Configurações e Código Principal
- **`config.yaml`**: Configurações principais (tickers, datas, parâmetros)
- **`src/backtesting/backtest.py`**: Lógica de backtest e otimização de thresholds
- **`src/models/train_models.py`**: Treinamento XGBoost e geração de targets
- **`src/features/build_features.py`**: Geração de features técnicas
- **`src/reports/diagnostics.py`**: Módulo de diagnósticos implementado
- **`__main__.py`**: Ponto de entrada do projeto

### 📊 Resultados por Etapa
- **`reports/ETAPA0-SIMULACOES.txt`**: Baseline original (hold_min_days=5, grid restrito)
- **`reports/ETAPA1_RESULTADOS.txt`**: Melhor resultado (grid ampliado 0.1-0.8)
- **`reports/ETAPA2_RESULTADOS.txt`**: Falha (hold_min_days=1, over-trading)
- **`reports/ETAPA3_RESULTADOS.txt`**: Resultado atual (targets ajustados, critério 0.10)

### 🔍 Diagnósticos Implementados
- **`reports/diagnostics_label_distribution_*.csv`**: Distribuição de rótulos por ticker/trimestre
- **`reports/diagnostics_pr_curves_*.csv`**: AUC-PR por ticker (qualidade do sinal)
- **`reports/diagnostics_calibration_brier_*.csv`**: Calibração/Brier por ticker
- **`reports/diagnostics_alignment_*.csv`**: Alinhamento p_up vs retornos (leakage check)

### 📈 Dados de Treinamento
- **`data/01_raw/`**: Dados OHLCV brutos por ticker
- **`data/03_features/`**: Features técnicas calculadas
- **`data/04_labeled/`**: Dados com targets gerados (triple barrier)
- **`models/01_xgboost/`**: Modelos XGBoost treinados por ticker

### 📋 Arquivos de Contexto
- **`CONTEXTO_PROJETO.md`**: Este arquivo - contexto completo do projeto
- **`requirements.txt`**: Dependências Python necessárias
- **`environment.yml`**: Ambiente conda (se usado)

---

## 🛠️ COMANDOS ÚTEIS

### Executar Pipeline Completo
```bash
# Treinar modelos e gerar targets
python -m src.models.train_models

# Executar backtest
python -m src.backtesting.backtest

# Executar diagnósticos
python -m src.reports.diagnostics
```

### Executar Apenas Backtest
```bash
python -m src.backtesting.backtest
```

### Executar Apenas Diagnósticos
```bash
python -m src.reports.diagnostics
```

---

## 📋 CHECKLIST DE CONTINUIDADE

### ✅ O que já foi implementado
- [x] Grid de thresholds ampliado (0.1-0.8)
- [x] Critério de classe "Up" ajustado (0.02 → 0.10)
- [x] Targets mais brandos no triple_barrier_grid
- [x] Módulo de diagnósticos completo
- [x] hold_min_days=5 (otimizado)

### 🔄 O que precisa ser feito (ETAPA 4)
- [ ] Implementar calibração Platt/Isotônica por ticker
- [ ] Integrar calibração no pipeline de otimização
- [ ] Reotimizar thresholds após calibração
- [ ] Validar melhoria com Brier score < 0.1

### 🔍 O que investigar (ETAPA 5)
- [ ] Auditoria de features para leakage
- [ ] Teste de defasagem de features
- [ ] Validação temporal out-of-sample

### 🎯 O que otimizar (ETAPA 6)
- [ ] Política por ticker baseada em performance
- [ ] Targets específicos para tickers problemáticos
- [ ] Estratégia para ITUB4.SA (0 trades)

---

## 🚀 PRÓXIMO PASSO IMEDIATO

**ETAPA 4: Calibração de Probabilidades**

1. **Implementar calibração** no `src/models/train_models.py`
2. **Modificar** `adaptive_threshold_optimization` para usar probabilidades calibradas
3. **Retreinar modelos** com calibração
4. **Executar backtest** e comparar com ETAPA 1
5. **Validar** melhoria do Brier score

**Arquivo principal para modificar**: `src/models/train_models.py`
**Função principal**: `adaptive_threshold_optimization` em `src/backtesting/backtest.py`

---

## 🔧 CÓDIGO ATUAL - IMPLEMENTAÇÕES

### Grid de Thresholds (ETAPA 1 implementada)
```python
# src/backtesting/backtest.py - adaptive_threshold_optimization
coarse_buy_grid = np.arange(0.10, 0.80, 0.10)  # 0.10, 0.20, ..., 0.70
coarse_sell_grid = np.arange(-0.50, 0.20, 0.10)  # -0.50, -0.40, ..., 0.10

# Grid fino para refinamento
fine_buy_grid = np.arange(max(0.05, best_buy_coarse - 0.10), 
                          min(0.85, best_buy_coarse + 0.10), 0.01)
fine_sell_grid = np.arange(max(-0.60, best_sell_coarse - 0.10), 
                           min(0.30, best_sell_coarse + 0.10), 0.01)
```

### Critério de Classe Up (ETAPA 3 implementada)
```python
# src/models/train_models.py - linha 222
if up_class_ratio < 0.10:  # Era 0.02
    print(f"  Parâmetros {i+1}: Rejeitado - Classe Up muito baixa ({up_class_ratio:.3f})")
    continue
```

### Targets Mais Brandos (ETAPA 3 implementada)
```yaml
# config.yaml - triple_barrier_grid (adicionados)
- {holding_days: 7, profit_threshold: 0.015, loss_threshold: -0.01}
- {holding_days: 10, profit_threshold: 0.02, loss_threshold: -0.015}
- {holding_days: 5, profit_threshold: 0.025, loss_threshold: -0.02}
- {holding_days: 7, profit_threshold: 0.03, loss_threshold: -0.025}
- {holding_days: 10, profit_threshold: 0.035, loss_threshold: -0.02}
```

### Hold Mínimo Otimizado
```yaml
# config.yaml - threshold_optimization
threshold_optimization:
  enabled: true
  metric: "sharpe"
  hold_min_days: 5  # Otimizado (1 foi desastroso)
```

---

## 🚨 PROBLEMAS IDENTIFICADOS - RESUMO EXECUTIVO

### 1. Calibração Ruim (CRÍTICO - ETAPA 4)
- **PETR4.SA**: Brier 0.207 (ruim) vs AUC-PR 0.688 (boa)
- **B3SA3.SA**: Brier 0.194 (ruim) vs AUC-PR 0.742 (boa)
- **Impacto**: Thresholds ineficazes, performance subótima
- **Solução**: Calibração Platt/Isotônica por ticker

### 2. Possível Leakage (IMPORTANTE - ETAPA 5)
- **PETR4.SA**: Correlação same day 0.551 (muito alta)
- **BBAS3.SA**: Correlação same day 0.554 (muito alta)
- **Impacto**: Overfitting, performance não generaliza
- **Solução**: Auditoria e defasagem de features

### 3. Desbalanceamento Severo (ESTRUTURAL)
- **ITUB4.SA**: 0% "Up" em Q4 2024 (impossível operar)
- **VALE3.SA**: 6.6% "Up" em Q4 2024 (muito baixo)
- **Impacto**: Impossível gerar trades em alguns tickers
- **Solução**: Targets adaptativos por ticker/regime

### 4. Over-trading (RESOLVIDO)
- **ETAPA 2**: hold_min_days=1 destruiu performance
- **Impacto**: -25 p.p. de retorno em PETR4.SA
- **Solução**: Manter hold_min_days=5 (implementado)

### 5. Qualidade vs Quantidade (PARCIALMENTE RESOLVIDO)
- **ETAPA 3**: Mais "Up" mas pior qualidade
- **Impacto**: Sharpe caiu de 2.097 para 1.049
- **Solução**: Calibração + targets intermediários

---

## 💡 INSIGHTS PRINCIPAIS - LIÇÕES APRENDIDAS

### ✅ O que funciona
1. **PETR4.SA é o melhor ticker**: Classes equilibradas (31.9% "Up"), boa AUC-PR (0.688)
2. **Grid ampliado funcionou**: Encontrou thresholds melhores fora da faixa 0.2-0.6
3. **Histerese temporal é crítica**: hold_min_days=5 filtra ruído vs 1 dia
4. **Histerese de threshold funciona**: buy > sell evita over-trading
5. **AUC-PR alta se converte em PnL**: Quando há sinal, dá para monetizar

### ❌ O que não funciona
1. **hold_min_days=1**: Over-trading, custos dominam benefícios
2. **Targets muito brandos**: Mais "Up" mas pior qualidade
3. **Critério 0.02 muito restritivo**: Perdeu oportunidades
4. **Probabilidades mal calibradas**: Thresholds ineficazes
5. **ITUB4.SA intratável**: 0 trades em todas as etapas

### 🔍 O que descobrimos
1. **Regime 2024 é adverso**: Buy&Hold negativo em vários tickers
2. **Sazonalidade extrema**: ABEV3.SA Q1-Q2 com 0% "Up"
3. **Correlação same day alta**: Possível leakage em features
4. **Calibração é crítica**: Brier ruim = thresholds ruins
5. **Qualidade > Quantidade**: Melhor ter menos sinais mas melhores

---

## 🎯 OBJETIVO FINAL - METAS ESPECÍFICAS

### Meta Principal
Criar um modelo de trading que supere consistentemente o Buy&Hold em múltiplos tickers, mantendo Sharpe > 1.0 e drawdown < 15%.

### Metas Específicas por Ticker
- **PETR4.SA**: Manter +28% retorno, Sharpe > 2.0, drawdown < 10%
- **BBDC4.SA**: Melhorar de +3.44% para +10%+, Sharpe > 1.0
- **VIVT3.SA**: Manter +1.94%, reduzir ruído, Sharpe > 1.0
- **VALE3.SA/BBAS3.SA/B3SA3.SA**: Gerar trades consistentes, retorno > 0%
- **ITUB4.SA**: Investigar viabilidade, gerar pelo menos alguns trades

### Metas Técnicas
- **Brier score < 0.1** para todos os tickers (atual: 0.207 PETR4)
- **Correlação same day < 0.1** (atual: 0.551 PETR4)
- **Precisão "Up" > 60%** (atual: 41.75% PETR4)
- **Número de trades > 0** para todos os tickers (atual: ITUB4 = 0)

---

## 📊 STATUS ATUAL - RESUMO EXECUTIVO

**Progresso**: 60% concluído
- ✅ **ETAPA 1**: Grid ampliado (SUCESSO)
- ❌ **ETAPA 2**: hold_min_days=1 (FALHA)
- ⚠️ **ETAPA 3**: Targets ajustados (SUCESSO PARCIAL)

**Próximo Passo Crítico**: ETAPA 4 - Calibração de probabilidades
**Arquivo Principal**: `src/models/train_models.py`
**Função Principal**: `adaptive_threshold_optimization` em `src/backtesting/backtest.py`

**Expectativa**: PETR4.SA retornar para +28% com Sharpe > 2.0 após calibração
