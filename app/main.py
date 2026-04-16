"""Streamlit App - Interface para demonstração do ensemble"""

import streamlit as st
from src.ensemble import ensemble_predict, DEFAULT_WEIGHTS
from src.models import load_pipelines
from src.preprocessor import clean_tweet

st.set_page_config(
    page_title="Sentiment Ensemble",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource
def get_pipelines():
    return load_pipelines()


def main():
    st.title("🧠 Sentiment Ensemble")
    st.markdown("""
    Ensemble de 3 modelos transformer para análise de sentimento em tweets.
    Combina **RoBERTa-Twitter**, **BERTweet** e **DistilBERT** via weighted soft voting.
    """)
    
    st.divider()
    
    text = st.text_area(
        "Digite um tweet para analisar:",
        height=4,
        placeholder="Ex: I love this product! Amazing quality 👏",
    )
    
    if st.button("Analisar Sentimento", type="primary"):
        if not text.strip():
            st.warning("Por favor, digite um texto para analisar.")
            return
        
        cleaned = clean_tweet(text)
        pipes = get_pipelines()
        
        with st.spinner("Analisando..."):
            pred_idx, confidence, model_scores = ensemble_predict(
                text, pipes, DEFAULT_WEIGHTS
            )
        
        labels = {"negative": "Negativo", "neutral": "Neutro", "positive": "Positivo"}
        pred_label = labels[["negative", "neutral", "positive"][pred_idx]]
        
        colors = {"negative": "#ff6b6b", "neutral": "#ced4da", "positive": "#51cf66"}
        
        st.success(f"**Sentimento: {pred_label}**")
        st.metric("Confidence", f"{confidence:.1%}")
        
        st.subheader("Scores por Modelo")
        
        cols = st.columns(3)
        model_names = {
            "roberta_twitter": "RoBERTa Twitter",
            "bertweet": "BERTweet",
            "distilbert": "DistilBERT",
        }
        
        for i, (name, scores) in enumerate(model_scores.items()):
            with cols[i]:
                st.markdown(f"**{model_names[name]}**")
                for label, score in scores.items():
                    color = colors[label]
                    st.progress(score, text=f"{label}: {score:.1%}")
        
        with st.expander("Ver texto limpo"):
            st.code(cleaned)
    
    st.divider()
    
    st.markdown("""
    ### Modelos do Ensemble
    
    | Modelo | Descrição |
    |--------|-----------|
    | RoBERTa Twitter | Treinado em 58M tweets, referência em TweetEval |
    | BERTweet | BERT pré-treinado em 850M tweets |
    | DistilBERT | Rápido e leve, treinado em resenhas de filmes |
    
    ### Como funciona
    O ensemble usa **weighted soft voting**: as probabilidades de cada modelo
    são multiplicadas pelos pesos e então combinadas para formar a decisão final.
    """)


if __name__ == "__main__":
    main()