# Predição do Movimento de Ações com IA: Uma Análise Comparativa

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)

Este repositório contém o código-fonte e os resultados do meu Trabalho de Conclusão de Curso (TCC) em Ciência da Computação. O projeto foca na predição da direção do movimento de preços de ações do mercado brasileiro, utilizando uma análise comparativa entre dois modelos de Inteligência Artificial.

---

## 📜 Índice

* [Sobre o Projeto](#-sobre-o-projeto)
* [Modelos Implementados](#-modelos-implementados)
* [Tecnologias Utilizadas](#-tecnologias-utilizadas)
* [Estrutura do Projeto](#-estrutura-do-projeto)
* [Instalação](#-instalação)
* [Como Usar](#-como-usar)
* [Resultados](#-resultados)
* [Trabalhos Futuros](#-trabalhos-futuros)
* [Autor](#-autor)

---

## 📖 Sobre o Projeto

O mercado financeiro é um ambiente complexo e dinâmico, tornando a predição de preços de ativos um dos problemas mais desafiadores em finanças quantitativas. Este projeto aborda esse desafio tratando-o como um problema de **classificação binária**: o objetivo não é prever o valor exato de uma ação, mas sim sua direção (se o preço irá **subir** ou **descer** no dia seguinte).

Para avaliar a eficácia dessa abordagem, dois modelos híbridos e de deep learning são desenvolvidos, treinados e comparados. A performance final é medida não apenas por métricas de classificação (Acurácia, Precisão, F1-Score), mas também através de um backtesting, simulando uma estratégia de investimento baseada nas predições dos modelos para calcular o retorno financeiro.

---

## 🤖 Modelos Implementados

1.  **Transformada Wavelet + XGBoost**: Uma abordagem híbrida que primeiro utiliza a Transformada Wavelet para decompor a série temporal do preço da ação, separando o sinal em diferentes componentes de frequência. Isso ajuda a reduzir o ruído. Em seguida, um modelo Gradient Boosting (XGBoost) é treinado com os componentes da wavelet e outros indicadores técnicos para realizar a classificação.

2.  **Redes Neurais Recorrentes LSTM**: Um modelo de deep learning projetado especificamente para aprender padrões a partir de dados sequenciais. A rede LSTM (Long Short-Term Memory) é capaz de capturar dependências de longo prazo na série temporal, sendo uma escolha natural para a predição de preços de ações.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10
* **Gerenciamento de Ambiente:** Conda
* **Manipulação de Dados:** Pandas, NumPy
* **Aquisição de Dados:** yfinance
* **Engenharia de Features:** TA-Lib, PyWavelets
* **Machine Learning / Deep Learning:**
    * XGBoost
    * Scikit-learn (para métricas de avaliação e pré-processamento)
* **Visualização de Dados:** Matplotlib
* **Jupyter Notebooks** para exploração e desenvolvimento
* **Configuração:** YAML para parâmetros do projeto

---

## 📂 Estrutura do Projeto

A estrutura de pastas está organizada seguindo as melhores práticas de projetos de ciência de dados:

```
/
├── config.yaml              # Configurações do projeto (tickers, datas, parâmetros)
├── environment.yml           # Ambiente Conda com todas as dependências
├── requirements.txt          # Dependências pip (complementar ao Conda)
├── data/                    # Datasets brutos e processados
│   ├── 01_raw/             # Dados brutos baixados do Yahoo Finance
│   ├── 02_processed/       # Dados processados e limpos
│   └── 03_features/        # Dados com features técnicas e wavelet
├── notebooks/               # Jupyter Notebooks para exploração e testes
│   ├── 01_XGBoostModel.ipynb
│   └── 02_DataCleaning_Example.ipynb
├── src/                     # Código fonte principal
│   ├── data/
│   │   └── make_dataset.py  # Scripts para baixar dados do Yahoo Finance
│   ├── features/
│   │   └── build_features.py # Scripts para criar indicadores técnicos e features wavelet
│   └── models/
│       └── train_models.py  # Script para treinar os modelos
├── models/                  # Modelos treinados salvos
├── reports/                 # Relatórios e resultados
└── README.md                # Este arquivo
```

---

## ⚙️ Instalação

Para configurar o ambiente e rodar o projeto localmente, siga os passos abaixo:

### Pré-requisitos
* **Conda** instalado no seu sistema (recomendado: Miniconda ou Anaconda)
* **Git** para clonar o repositório

### Passos de Instalação

1.  **Clone o repositório:**
    ```sh
    git clone https://github.com/arthurhbu/XGBoost-NeuralNetworkLSTM.git
    cd XGBoost-NeuralNetworkLSTM
    ```

2.  **Crie e ative o ambiente Conda:**
    ```sh
    # Criar o ambiente com todas as dependências
    conda env create -f environment.yml
    
    # Ativar o ambiente
    conda activate projeto-analise-financeira
    ```

3.  **Verifique a instalação:**
    ```sh
    # Verificar se o ambiente está ativo
    conda info --envs
    
    # Verificar se as dependências estão instaladas
    python -c "import pandas, numpy, yfinance, talib, pywt, xgboost; print('Todas as dependências instaladas com sucesso!')"
    ```

### Alternativa com pip (não recomendado)
Se preferir usar apenas pip (não recomendado devido a possíveis conflitos com TA-Lib):
```sh
pip install -r requirements.txt
pip install pandas numpy matplotlib scikit-learn jupyter ta-lib xgboost yfinance pywavelets
```

---

## 🚀 Como Usar

### 1. Configuração Inicial
O projeto utiliza um arquivo `config.yaml` para centralizar todas as configurações:
- **Tickers das ações** a serem analisadas
- **Período de dados** (data início/fim)
- **Parâmetros dos indicadores técnicos**
- **Configurações da transformada wavelet**

### 2. Fluxo de Trabalho

#### Download dos Dados
```sh
# Baixar dados das ações configuradas no config.yaml
python src/data/make_dataset.py
```

#### Criação de Features
```sh
# Criar indicadores técnicos e features wavelet
python src/features/build_features.py
```

#### Desenvolvimento e Testes
```sh
# Abrir Jupyter Notebook para desenvolvimento
jupyter notebook notebooks/
```

### 3. Notebooks Disponíveis
- **`01_XGBoostModel.ipynb`**: Desenvolvimento do modelo XGBoost com indicadores técnicos
- **`02_DataCleaning_Example.ipynb`**: Exemplos de limpeza e processamento de dados

### 4. Estrutura de Dados
O projeto trabalha com as seguintes ações brasileiras:
- PETR4.SA (Petrobras)
- VALE3.SA (Vale)
- BBDC4.SA (Bradesco)
- ITUB4.SA (Itaú)
- BBAS3.SA (Banco do Brasil)
- B3SA3.SA (B3)
- ABEV3.SA (Ambev)
- VIVT3.SA (Vivo)

---

## 📊 Resultados

Nesta seção serão apresentados os resultados finais do projeto.

*(Local para inserir gráficos de performance do portfólio, matrizes de confusão e tabelas comparativas com as métricas de Acurácia, Precisão, Recall, F1-Score e Retorno sobre o Investimento (ROI) para cada modelo.)*

### Comparativo de Performance

| Métrica | XGBoost + Wavelet | LSTM | Buy & Hold |
| :-------- | :---------------: | :--: | :--------: |
| Acurácia |       XX.X%       | XX.X%|     -    |
| F1-Score  |       0.XX        | 0.XX |     -    |
| ROI Final |       +XX.X%      | +XX.X%|   +XX.X%   |

---

## 🔮 Trabalhos Futuros

* Desenvolvimento de uma interface gráfica (usando Streamlit ou Flask) para facilitar a visualização das predições.
* Teste dos modelos com um portfólio de múltiplos ativos simultaneamente.
* Incorporação de dados alternativos, como análise de sentimento de notícias.
* Otimização avançada de hiperparâmetros com ferramentas como Optuna.
* Implementação completa do modelo LSTM (atualmente em desenvolvimento).
* Sistema de backtesting automatizado para avaliação de performance.

---

## ✍️ Autor

* **Arthur Henrique Bando Ueda**
* **Email:** arthurhbu@gmail.com
* **LinkedIn:** [linkedin.com/in/arthurhbu](https://linkedin.com/in/arthurhbu)
* **GitHub:** [github.com/arthurhbu](https://github.com/arthurhbu)