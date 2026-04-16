# 🧠 Sentiment Ensemble — Twitter NLP

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> Ensemble de modelos transformer para análise de sentimento em tweets,
> combinando RoBERTa-Twitter, BERTweet e DistilBERT via soft voting ponderado.

## 🎯 Resultado Principal
O ensemble alcança **F1-Macro de ~0.77** no benchmark tweet_eval,
superando o melhor modelo individual.

## 🏗️ Arquitetura

```
Tweet → Limpeza → [M1: RoBERTa] ─┐
                  [M2: BERTweet] ─┼─ Weighted Soft Voting → Sentimento
                  [M3: DistilBERT]─┘
```

## 🚀 Instalação

```bash
cd pln_analysis
poetry install
```

## 📊 Notebooks

| Notebook | Descrição |
|---|---|
| [01_EDA.ipynb](notebooks/01_EDA.ipynb) | Análise exploratória do tweet_eval |
| [02_baseline.ipynb](notebooks/02_baseline.ipynb) | Avaliação individual dos modelos |
| [03_ensemble.ipynb](notebooks/03_ensemble.ipynb) | Ensemble + análise de erros |

## 🤗 Modelo no HuggingFace Hub

[seu-usuario/sentiment-ensemble](https://huggingface.co/seu-usuario/sentiment-ensemble)

## 📱 Interface Streamlit

```bash
poetry run streamlit run app/main.py
```

## 📋 Scripts

```bash
# Inferência standalone
poetry run python scripts/run_inference.py "I love this!"

# Publicar no HuggingFace Hub
poetry run python scripts/push_to_hub.py seu-usuario/sentiment-ensemble
```

## ⚙️ Dependências

- `transformers` — Hugging Face Transformers
- `datasets` — Hugging Face Datasets
- `torch` — PyTorch
- `scikit-learn` — Métricas
- `streamlit` — Interface web
- `pandas`, `matplotlib`, `seaborn` — Análise de dados

## 🔧 Testes

```bash
poetry run pytest tests/
```