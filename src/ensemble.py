"""Ensemble - Weighted Soft Voting para combinação dos 3 modelos"""

import numpy as np
from typing import Optional
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.models import LABEL_MAP, LABEL_NAMES, load_pipelines, predict_single
from src.preprocessor import clean_tweet


DEFAULT_WEIGHTS = {
    "roberta_twitter": 0.40,
    "bertweet": 0.35,
    "distilbert": 0.25,
}


def calibrate_weights(
    pipes: dict,
    texts: list[str],
    labels: list[int],
) -> dict:
    """
    Calcula pesos baseados no F1-Macro de cada modelo no validation set.
    
    Args:
        pipes: dict com name -> pipeline
        texts: lista de textos de validação
        labels: lista de labels correspondentes
    
    Returns:
        dict com name -> peso normalizado
    """
    weights = {}
    
    for name, pipe in pipes.items():
        preds = []
        for text in tqdm(texts, desc=f"Calibrando {name}"):
            pred, _ = predict_single(text, pipe, LABEL_MAP[name])
            preds.append(pred)
        
        f1 = f1_score(labels, preds, average="macro")
        weights[name] = f1
        print(f"  {name}: F1-Macro = {f1:.4f}")
    
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def ensemble_predict(
    text: str,
    pipes: dict,
    weights: Optional[dict] = None,
) -> tuple[int, float, dict]:
    """
    Faz predição usando weighted soft voting.
    
    Args:
        text: texto de entrada
        pipes: dict com name -> pipeline
        weights: dict com name -> peso. Se None, usa DEFAULT_WEIGHTS.
    
    Returns:
        (pred_idx, confidence, model_scores)
        - pred_idx: índice da classe predita (0=neg, 1=neu, 2=pos)
        - confidence: soma ponderada do vetor de probabilidades
        - model_scores: dict com scores de cada modelo
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    combined = np.zeros(3)
    model_scores = {}
    
    for name, pipe in pipes.items():
        result = pipe(text)[0]
        scores = np.zeros(3)
        
        for r in result:
            label = r["label"]
            score = r["score"]
            idx = LABEL_MAP[name].get(label)
            if idx is not None:
                scores[idx] = score
        
        if name == "distilbert":
            scores[1] = 1.0 - scores[0] - scores[2]
        
        model_scores[name] = {LABEL_NAMES[i]: scores[i] for i in range(3)}
        combined += weights[name] * scores
    
    pred_idx = int(np.argmax(combined))
    confidence = float(combined[pred_idx])
    
    return pred_idx, confidence, model_scores


def batch_predict(
    texts: list[str],
    pipes: dict,
    weights: Optional[dict] = None,
) -> list[tuple[int, float]]:
    """
    Faz predição em lote usando ensemble.
    
    Args:
        texts: lista de textos
        pipes: dict com name -> pipeline
        weights: dict com name -> peso
    
    Returns:
        lista de (pred_idx, confidence)
    """
    results = []
    for text in tqdm(texts, desc="Ensemble inference"):
        pred, conf, _ = ensemble_predict(text, pipes, weights)
        results.append((pred, conf))
    return results


def predict_sentiment(text: str) -> dict:
    """
    Função deconveniência para predição de sentimento.
    
    Args:
        text: texto do tweet
    
    Returns:
        dict com 'label' e 'confidence'
    """
    pipes = load_pipelines()
    pred_idx, confidence, model_scores = ensemble_predict(text, pipes)
    return {
        "label": LABEL_NAMES[pred_idx],
        "confidence": confidence,
        "model_scores": model_scores,
    }


if __name__ == "__main__":
    text = "I love this project! Great work 🎉"
    print(f"Texto: {text}")
    
    pipes = load_pipelines()
    pred, conf, scores = ensemble_predict(text, pipes)
    
    print(f"Predição: {LABEL_NAMES[pred]}")
    print(f"Confidence: {conf:.3f}")
    print(f"Scores por modelo: {scores}")