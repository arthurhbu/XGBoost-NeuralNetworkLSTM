Memorial de Melhorias para o TCC: Roteiro de Aprimoramento do Projeto
Este documento consolida as sugestões de melhorias e aprofundamentos técnicos discutidos, servindo como um guia para refinar o projeto após a finalização do pipeline principal.

Fase 1: Estrutura do Projeto e Boas Práticas
Melhoria 1: Gerenciamento de Dependências com requirements.txt
Descrição: Criar um arquivo que lista todas as bibliotecas Python e suas versões exatas utilizadas no projeto.

Por Que Fazer? Garante 100% de reprodutibilidade. Qualquer pessoa (ou você mesmo, em outro computador) pode recriar o ambiente de desenvolvimento idêntico com um único comando, evitando erros de versão.

Como Implementar: No terminal, com o ambiente virtual ativado, execute pip freeze > requirements.txt.

Melhoria 2: Uso de Logging em Vez de print()
Descrição: Substituir as chamadas print("Mensagem...") pelo módulo logging nativo do Python.

Por Que Fazer? Permite um controle muito mais profissional sobre as mensagens de execução. Você pode definir níveis (INFO, DEBUG, ERROR), formatar as mensagens com data/hora e salvar todo o histórico de execução em um arquivo de log para depuração futura.

Como Implementar: Importar o módulo logging no início dos scripts, configurar o formato desejado e usar logging.info("Mensagem...") em vez de print().

Fase 2: Pipeline de Dados e Features
Melhoria 3: Análise de Estacionariedade
Descrição: Verificar se a série temporal (ou suas transformações) é estacionária, ou seja, se suas propriedades estatísticas são constantes ao longo do tempo.

Por Que Fazer? É um conceito fundamental em econometria e análise de séries temporais. Discutir isso no TCC demonstra profundidade teórica. Embora o XGBoost seja robusto a dados não-estacionários, o LSTM pode se beneficiar de dados transformados (como os retornos diários, que são mais estacionários).

Como Implementar: Utilizar testes estatísticos como o Augmented Dickey-Fuller (ADF). Nos scripts, em vez de usar o preço de fechamento diretamente, você pode criar uma feature de retorno percentual diário com df['Close'].pct_change().

Melhoria 4: Escalonamento de Features (Feature Scaling)
Descrição: Normalizar todas as colunas de features para que fiquem em uma escala comum (ex: entre 0 e 1).

Por Que Fazer? Embora não seja crucial para o XGBoost, é obrigatório para o modelo LSTM. Implementar essa etapa de forma modular no seu pipeline facilitará a reutilização do código para o segundo modelo.

Como Implementar: Usar classes do Scikit-learn como MinMaxScaler ou StandardScaler após a criação de todas as features e antes da divisão dos dados de treino/teste.

Fase 3: Treinamento e Validação do Modelo
Melhoria 5: Otimização de Hiperparâmetros (Hyperparameter Tuning)
Descrição: Encontrar a melhor combinação de parâmetros para o modelo (ex: max_depth, learning_rate no XGBoost) de forma automática, em vez de usar valores fixos.

Por Que Fazer? Aumenta significativamente a performance e a robustez do modelo, mostrando que você extraiu o potencial máximo da arquitetura escolhida.

Como Implementar: Utilizar o conjunto de validação para testar diferentes configurações. Bibliotecas como GridSearchCV (testa todas as combinações) e RandomizedSearchCV (testa combinações aleatórias) do Scikit-learn, ou ferramentas mais avançadas como Optuna, são ideais para isso.

Melhoria 6: Análise de Importância das Features (Feature Importance)
Descrição: Extrair do modelo XGBoost treinado um ranking de quais features foram mais úteis para suas previsões.

Por Que Fazer? Adiciona interpretabilidade ao seu trabalho. Permite discutir o que o modelo aprendeu, transformando-o de uma "caixa-preta" para um sistema com lógica analisável. É um material riquíssimo para a escrita do TCC.

Como Implementar: Acessar o atributo model.feature_importances_ do modelo XGBoost treinado e usar o Matplotlib/Seaborn para criar um gráfico de barras com as features mais importantes.

Melhoria 7: Validação Cruzada para Séries Temporais (Walk-Forward Validation)
Descrição: Implementar uma forma de validação mais robusta que a simples divisão treino/validação/teste. A validação "walk-forward" simula o re-treinamento periódico do modelo ao longo do tempo.

Por Que Fazer? É considerado o padrão-ouro para avaliação de modelos de séries temporais financeiras, pois simula de forma mais fiel como um modelo seria operado na realidade. Aumenta a confiança de que o desempenho do modelo não é um acaso de um período de teste específico.

Como Implementar: Criar um loop que expande a janela de treinamento progressivamente. Ex: treina em [Ano 1], testa em [Ano 2]; depois treina em [Ano 1, Ano 2], testa em [Ano 3], e assim por diante.

Fase 4: Avaliação (Backtesting) e Expansão do Projeto
Melhoria 8: Métricas de Avaliação Financeira Avançadas
Descrição: Além do Retorno Total, calcular outras métricas padrão da indústria financeira.

Por Que Fazer? Fornece uma visão mais completa do perfil de risco-retorno da sua estratégia.

Como Implementar: Calcular o Drawdown Máximo (maior perda do pico ao vale), Sharpe Ratio (retorno ajustado ao risco pela volatilidade) e Sortino Ratio (similar ao Sharpe, mas foca apenas na volatilidade negativa).

Melhoria 9: Desenvolvimento de Interface Gráfica
Descrição: Criar uma interface web simples para visualizar os resultados ou até mesmo as predições do modelo.

Por Que Fazer? Torna o projeto muito mais acessível e visualmente atraente para apresentação.

Como Implementar: Utilizar bibliotecas como Streamlit ou Flask. O Streamlit é especialmente fácil e rápido para criar dashboards de dados interativos.

PROMPT para normalização de código: 

Aqui está um prompt reutilizável que reproduz exatamente o que fiz agora, pronto para você colar no futuro.

Prompt (cole exatamente este texto):
Quero padronizar meu código em Python com foco em qualidade e consistência, sem alterar a lógica de negócio. Siga exatamente estas instruções:

1) Escopo desta etapa:
- Escolha UMA função representativa no arquivo indicado (ou em src/ se não for indicado).
- Faça as melhorias apenas nessa função, e pare para pedir meu feedback antes de seguir para outras.

2) Tipagem e validações:
- Adicione type hints completos a parâmetros e retorno.
- Adote nomes de variáveis claros e consistentes, preservando compatibilidade externa.
- Valide entradas essenciais:
  - Se o parâmetro deveria ser um pandas DataFrame, lance TypeError se não for.
  - Se houver colunas obrigatórias (ex.: OHLCV: Close, High, Low, Open, Volume), valide e lance ValueError listando as faltantes.
  - Converta colunas numéricas com pd.to_numeric(errors='coerce') e remova linhas inválidas quando fizer sentido, mantendo a mesma lógica de saída.

3) Docstrings (em português, estilo Google):
- Escreva em PT-BR, com seções: Resumo em uma linha, descrições, Args, Returns, Raises.
- Use nomes dos parâmetros exatamente como na função.
- Seja específico e objetivo, evitando redundância.

4) Restrições:
- Não altere o comportamento funcional nem a assinatura externa além de adicionar tipos.
- Não introduza novas dependências.
- Preserve o estilo de indentação e formatação existente.
- Evite comentários triviais; foque em docstrings e validações significativas.

5) Pós-edição:
- Rode/verifique linter e ajuste apenas problemas triviais.
- Me retorne:
  - Caminho e nome da função alterada.
  - O trecho de código atualizado da função.
  - Uma frase confirmando que a lógica não foi alterada.

Se estiver claro, comece escolhendo a melhor função para este padrão e aplique as edições apenas nela.