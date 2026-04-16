"""Models - Wrapper para os 3 modelos HF do ensemble"""

from transformers import pipeline
import torch
from typing import Optional


MODELS = {
    "roberta_twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "bertweet": "finiteautomata/bertweet-base-sentiment-analysis",
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
}

LABEL_MAP = {
    "roberta_twitter": {"negative": 0, "neutral": 1, "positive": 2},
    "bertweet": {"NEG": 0, "NEU": 1, "POS": 2},
    "distilbert": {"NEGATIVE": 0, "POSITIVE": 2},
}

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def get_device() -> int:
    """Retorna device: -1 para CPU, 0 para GPU se disponível."""
    return 0 if torch.cuda.is_available() else -1


def load_pipelines(device: Optional[int] = None) -> dict:
    """
    Carrega os 3 pipelines do Hugging Face.
    
    Args:
        device: -1 (CPU) ou 0 (GPU). Se None, detecta automaticamente.
    
    Returns:
        dict com name -> pipeline
    """
    if device is None:
        device = get_device()
    
    pipes = {}
    for name, model_id in MODELS.items():
        pipes[name] = pipeline(
            "text-classification",
            model=model_id,
            top_k=None,
            device=device,
            truncation=True,
            max_length=128,
        )
    return pipes


def predict_single(text: str, pipe, label_map: dict) -> tuple[int, float]:
    """
    Faz predição com um único modelo.
    
    Args:
        text: texto de entrada
        pipe: pipeline do modelo
        label_map: mapeamento de labels para índices
    
    Returns:
        (predição_idx, confidence)
    """
    result = pipe(text)[0]
    scores = {}
    for r in result:
        label = r["label"]
        score = r["score"]
        idx = label_map.get(label)
        if idx is not None:
            scores[idx] = score
    
    pred = max(scores, key=scores.get)
    confidence = scores[pred]
    return pred, confidence


def predict_with_all_models(text: str, pipes: dict) -> dict:
    """
    Faz predição com todos os 3 modelos.
    
    Args:
        text: texto de entrada
        pipes: dict com name -> pipeline
    
    Returns:
        dict com name -> (pred_idx, confidence)
    """
    results = {}
    for name, pipe in pipes.items():
        pred, conf = predict_single(text, pipe, LABEL_MAP[name])
        results[name] = (pred, conf)
    return results


if __name__ == "__main__":
    print("Carregando pipelines...")
    pipes = load_pipelines()
    print("Modelos carregados!")
    
    test_texts = [
        "I love this product! Amazing quality 👏",
        "This is terrible, I hate it 😡",
        "The meeting is at 3pm",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        results = predict_with_all_models(text, pipes)
        for name, (pred, conf) in results.items():
            print(f"  {name}: {LABEL_NAMES[pred]} ({conf:.3f})")