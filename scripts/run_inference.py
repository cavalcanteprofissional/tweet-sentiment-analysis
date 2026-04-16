"""Script para rodar inferência standalone"""

import argparse
from src.ensemble import ensemble_predict, DEFAULT_WEIGHTS
from src.models import load_pipelines


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Ensemble - Inference CLI"
    )
    parser.add_argument("text", help="Texto para analisar")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        default=None,
        help="Pesos do ensemble (roberta, bertweet, distilbert)",
    )
    args = parser.parse_args()
    
    pipes = load_pipelines()
    
    if args.weights:
        weights = {
            "roberta_twitter": args.weights[0],
            "bertweet": args.weights[1],
            "distilbert": args.weights[2],
        }
    else:
        weights = DEFAULT_WEIGHTS
    
    pred_idx, confidence, model_scores = ensemble_predict(
        args.text, pipes, weights
    )
    
    labels = ["negative", "neutral", "positive"]
    label_names = ["Negativo", "Neutro", "Positivo"]
    
    print(f"\nTexto: {args.text}")
    print(f"Predição: {label_names[pred_idx]} (confidence: {confidence:.3f})")
    print("\nScores por modelo:")
    for name, scores in model_scores.items():
        print(f"  {name}:")
        for label, score in scores.items():
            print(f"    {label}: {score:.3f}")


if __name__ == "__main__":
    main()