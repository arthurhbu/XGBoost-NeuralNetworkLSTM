# Predição do Movimento de Ações com IA: Uma Análise Comparativa

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

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

Para avaliar a eficácia dessa abordagem, dois modelos híbridos e de deep learning são desenvolvidos, treinados e comparados. A performance final é medida não apenas por métricas de classificação (Acurácia, Precisão, F1-Score), but também através de um backtesting, simulando uma estratégia de investimento baseada nas predições dos modelos para calcular o retorno financeiro.

---

## 🤖 Modelos Implementados

1.  **Transformada Wavelet + XGBoost**: Uma abordagem híbrida que primeiro utiliza a Transformada Wavelet para decompor a série temporal do preço da ação, separando o sinal em diferentes componentes de frequência. Isso ajuda a reduzir o ruído. Em seguida, um modelo Gradient Boosting (XGBoost) é treinado com os componentes da wavelet e outros indicadores técnicos para realizar a classificação.

2.  **Redes Neurais Recorrentes LSTM**: Um modelo de deep learning projetado especificamente para aprender padrões a partir de dados sequenciais. A rede LSTM (Long Short-Term Memory) é capaz de capturar dependências de longo prazo na série temporal, sendo uma escolha natural para a predição de preços de ações.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Manipulação de Dados:** Pandas, NumPy
* **Aquisição de Dados:** yfinance
* **Engenharia de Features:** pandas_ta, PyWavelets
* **Machine Learning / Deep Learning:**
    * XGBoost
    * TensorFlow / Keras (para o modelo LSTM)
    * Scikit-learn (para métricas de avaliação e pré-processamento)
* **Visualização de Dados:** Matplotlib, Seaborn

---

## 📂 Estrutura do Projeto

A estrutura de pastas está organizada para separar as diferentes responsabilidades do projeto:

```
/
├── data/                  # Datasets brutos e processados
│   ├── raw/
│   └── processed/
├── notebooks/             # Jupyter Notebooks para exploração e testes
│   └── 01_data_exploration.ipynb
├── src/                   # Código fonte principal
│   ├── data_processing.py # Scripts para baixar e processar os dados
│   ├── feature_engineering.py # Scripts para criar indicadores e features
│   ├── train_model.py     # Script para treinar os modelos (XGBoost e LSTM)
│   ├── evaluate.py        # Script para avaliar os modelos com métricas
│   └── backtesting.py     # Script para a simulação da estratégia
├── results/               # Gráficos, tabelas e resultados finais
│   └── portfolio_performance.png
├── .gitignore             # Arquivos a serem ignorados pelo Git
├── README.md              # Este arquivo
└── requirements.txt       # Lista de dependências do projeto
```

---

## ⚙️ Instalação

Para configurar o ambiente e rodar o projeto localmente, siga os passos abaixo:

1.  **Clone o repositório:**
    ```sh
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```sh
    # Para Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    (Certifique-se de ter criado um arquivo `requirements.txt` com todas as bibliotecas necessárias)
    ```sh
    pip install -r requirements.txt
    ```

---

## 🚀 Como Usar

As instruções detalhadas para executar cada etapa do projeto estarão nos scripts dentro da pasta `src/`. O fluxo de trabalho geral é:

1.  **Executar o processamento de dados:**
    ```sh
    python src/data_processing.py
    ```
2.  **Treinar um dos modelos:**
    ```sh
    # Para treinar o XGBoost
    python src/train_model.py --model xgboost

    # Para treinar o LSTM
    python src/train_model.py --model lstm
    ```
3.  **Avaliar e simular a estratégia:**
    ```sh
    python src/evaluate.py --model xgboost
    ```
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

---

## ✍️ Autor

* **[Seu Nome Completo]**
* **Email:** `[arthurhbu@gmail.com]`
* **LinkedIn:** `[linkedin.com/in/arthurhbu]`
* **GitHub:** `[github.com/arthurhbu]`