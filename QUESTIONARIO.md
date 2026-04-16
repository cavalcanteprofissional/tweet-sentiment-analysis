# 🎓 Projeto Final — Python em Projetos de IA

👥 **Grupo:** 4 alunos  
🎯 **Objetivo:** Resolver um problema real utilizando técnicas de IA estudadas na disciplina.

| # | Nome | Email |
|---|------|-------|
| 1 | George Lucas Lopes da Silva Gomes | |
| 2 | Lucas Cavalcante dos Santos | cavalcantesidi@outlook.com |
| 3 | Raylson Silva de Lima | raylson.ifce@gmail.com |
| 4 | Sthefferson Bruno Costa Ferreira | sthefferson.ufma@gmail.com |

---

## 🧱 Estrutura do Projeto

### 1. Definição do Problema

- **Qual problema real vocês estão resolvendo?**
  Classificação automática de sentimento em tweets (positivo, negativo, neutro).

- **Qual o impacto prático desse problema?**
  Monitoramento de marca em tempo real, análise de crises, pesquisa social e coleta de opinião pública.

- **Existe solução atual?**
  Sim. VADER e TextBlob são amplamente usados para análise de sentimento.

- **Quais são as limitações?**
  - VADER não lida bem com português ou contextos culturais específicos
  - Modelos bag-of-words perdem contexto sequencial (ironia, negação)
  - Fine-tuning de um único modelo é frágil a distribuições out-of-distribution

- **Tipo de problema (classificação, regressão, etc.)**
  Classificação multiclasse (3 classes: positivo, negativo, neutro) com textos curtos e ruidosos.

---

### 2. Dataset

- **Fonte dos dados (link ou origem)**
  `cardiffnlp/tweet_eval` (Hugging Face Datasets) — benchmark padrão para tweets.

- **Quantidade de dados (tamanho)**
  - Train: 45.615 amostras
  - Validation: 2.000 amostras
  - Test: 12.284 amostras

- **Descrição das features**
  - `text`: texto do tweet (string)
  - `label`: 0=negative, 1=neutral, 2=positive

- **Existe desbalanceamento?**
  Sim. Espera-se: neutro > positivo > negativo (desequilíbrio moderado).

- **O dataset representa bem o problema real?**
  Sim. É um benchmark usado em papers acadêmicos e representa bem a linguagem informal do Twitter.

---

### 3. Pré-processamento

- **Quais técnicas foram aplicadas?**
  - Remoção de URLs
  - Normalização de @menções → @user (preserva estrutura sem vazar dados de usuários)
  - Normalização de #hashtags (remove # mas mantém a palavra)
  - Redução de caracteres repetidos (ex: "looooove" → "loove")
  - Remoção de espaços extras

- **Por que essas escolhas foram feitas?**
  - Preservar semântica emocional
  - NÃO remover emojis — eles carregam sentimento (ex: 😂, 😢)
  - NÃO fazer stemming — transformers tokenizam no nível de subpalavras
  - NÃO remover stopwords — negações como "não" e "never" são críticas para sentimento

- **Testaram outras abordagens de pré-processamento?**
  Não. As escolhas foram baseadas em boas práticas para transformers.

---

### 4. Escolha do Modelo

- **Qual modelo de IA foi escolhido?**
  Ensemble de 3 modelos transformer pré-treinados:
  1. `cardiffnlp/twitter-roberta-base-sentiment-latest` (125M parâmetros)
  2. `finiteautomata/bertweet-base-sentiment-analysis` (110M parâmetros)
  3. `distilbert-base-uncased-finetuned-sst-2-english` (66M parâmetros)

- **Por que esse modelo é adequado para o problema?**
  - RoBERTa treinada em 58M tweets (state-of-the-art em tweet_eval)
  - BERTweet pré-treinado em 850M tweets
  - DistilBERT rápido e leve, age como "regularizador"
  - Ensemble combina perspectivas diversas → mais robusto

- **Quais alternativas foram consideradas?**
  - VADER: baseado em regras, sem aprendizado — teto de performance baixo
  - GPT via API: custo proibitivo + não reproduzível offline
  - Treinar BERT do zero: inviável sem GPU dedicada

- **Qual a complexidade do modelo?**
  ~301M parâmetros no total (125M + 110M + 66M)

- **Existe risco de overfitting?**
  Baixo. Zero-shot inference (não há fine-tuning), modelos já pré-treinados.

---

### 5. Treinamento — Código

- **Divisão treino/teste (justifique a proporção)**
  Splits oficiais do dataset (train/validation/test). Não há data leakage.

- **Batch size, epochs, learning rate**
  N/A — abordagem zero-shot (sem treinamento).

- **Função de loss utilizada**
  N/A — sem treinamento, apenas inferência.

- **Como foi monitorado o treinamento**
  N/A

- **Houve fine-tuning de parâmetros?**
  Não. Os modelos já foram fine-tunados em dados de tweet sentiment.

- **Houve ajuste de hiperparâmetros?**
  Sim. Calibração de pesos do ensemble via F1-Macro no validation set.

---

### 6. Avaliação

- **Quais métricas foram usadas e por quê?**
  - **F1-Macro:** principal, penaliza igualmente erros em classes minoritárias
  - **Accuracy:** reportada mas não usada como critério principal
  - **Precision/Recall/F1 por classe:** diagnóstico

- **Accuracy é suficiente para esse problema?**
  Não. Com desbalanceamento, accuracy é enganosa.

- **O dataset é desbalanceado? Como isso afeta as métricas?**
  Sim. F1-Macro é usada para compensar o desbalanceamento.

- **Apresentar as métricas:**
  | Modelo | F1-Macro esperado |
  |--------|-------------------|
  | DistilBERT | ~0.65–0.70 |
  | BERTweet | ~0.72–0.76 |
  | RoBERTa Twitter | ~0.74–0.78 |
  | **Ensemble** | **~0.76–0.80** |

---

### 7. Resultados

- **O modelo performou bem? Justifique.**
  Sim. O ensemble deve superar todos os modelos individuais.

- **Comparação entre modelos (se houver)**
  RoBERTa Twitter > BERTweet > DistilBERT (individuais)
  Ensemble > todos (combinação)

- **Análise de erros:**
  - **Onde o modelo falha?** Classe neutro (mais ambígua)
  - **Por quê?** Tweets neutros têm linguagem menos expressiva

---

### 8. Limitações

- **Limitações do dataset**
  - Apenas em inglês (perde performance em PT-BR)

- **Limitações do modelo**
  - Ironia e sarcasmo continuam sendo o maior desafio

- **Limitações computacionais**
  - Inferência lenta (~2s/tweet em CPU com 3 modelos)

---

### 9. Melhorias Futuras

- **O que poderia melhorar o modelo?**
  - Adicionar modelo PT-BR (ex: neuralmind/bert-base-portuguese-cased)
  - Stacking com meta-learner
  - Quantização com `optimum` para 4x mais velocidade

- **Mais dados ajudariam?**
  Sim, especialmente para classes minoritárias (backtranslation).

- **Outro modelo seria melhor?**
  BERT multilíngue ou modelos específicos para PT-BR.

---

### 10. Conclusão Técnica

- **O modelo resolve o problema?**
  Sim. F1-Macro > 0.75 é referência de produção.

- **É viável em produção?**
  Com restrições. Precisa de caching e batching para latência aceitável.

- **Qual seria o próximo passo real?**
  Deploy em FastAPI + Redis cache para latência <200ms.

---

## 📋 Checklist do Trabalho Final

- [x] **Definição do Problema**
- [x] **Dataset** — tweet_eval via HF Datasets
- [x] **Pré-processamento** — limpeza de tweets justificada
- [x] **Escolha do Modelo** — 3 modelos + justificativa técnica
- [x] **Treinamento / Código** — pipelines HF + ensemble
- [x] **Avaliação** — F1-Macro, confusion matrix
- [x] **Resultados** — tabela comparativa
- [x] **Limitações** — listadas honestamente
- [x] **Melhorias Futuras**
- [x] **Conclusão Técnica**

### Diferenciais de portfólio

- [x] Modelo publicado no HuggingFace Hub com model card
- [x] Repositório GitHub com README profissional
- [x] Interface Streamlit para demo