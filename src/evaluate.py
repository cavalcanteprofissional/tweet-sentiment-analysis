"""Evaluate - Métricas e relatórios de avaliação"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import numpy as np


LABEL_NAMES = ["negative", "neutral", "positive"]


def full_report(
    y_true: list[int],
    y_pred: list[int],
    model_name: str = "Ensemble",
    save_path: Optional[str] = None,
) -> dict:
    """
    Gera relatório completo de avaliação.
    
    Args:
        y_true: labels verdadeiros
        y_pred: labels preditos
        model_name: nome do modelo
        save_path: caminho para salvar figura da confusion matrix
    
    Returns:
        dict com métricas
    """
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()
    
    return {
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(
    texts: list[str],
    y_true: list[int],
    predict_fn,
    model_name: str = "Model",
) -> dict:
    """
    Avaliaum modelo em um dataset.
    
    Args:
        texts: lista de textos
        y_true: labels verdadeiros
        predict_fn: função que recebe texto e retorna predição
        model_name: nome do modelo
    
    Returns:
        dict com métricas
    """
    y_pred = [predict_fn(text) for text in texts]
    return full_report(y_true, y_pred, model_name)


def compare_models(results: dict[str, dict]) -> None:
    """
    Imprime tabela comparativa de resultados de múltiplos modelos.
    
    Args:
        results: dict com nome -> dict de métricas
    """
    print("\n" + "="*60)
    print("  Comparativo de Modelos")
    print("="*60)
    print(f"{'Modelo':<20} {'F1-Macro':<12} {'Accuracy':<12}")
    print("-"*60)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['f1_macro']:<12.4f} {metrics['accuracy']:<12.4f}")
    
    best = max(results.items(), key=lambda x: x[1]["f1_macro"])
    print("-"*60)
    print(f"Mejor: {best[0]} (F1-Macro = {best[1]['f1_macro']:.4f})")


def plot_class_distribution(
    labels: list[int],
    save_path: Optional[str] = None,
) -> None:
    """
    Plota distribuição de classes.
    
    Args:
        labels: lista de labels
        save_path: caminho para salvar figura
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([LABEL_NAMES[u] for u in unique], counts, color=["red", "gray", "green"])
    ax.set_title("Distribuição de Classes")
    ax.set_ylabel("Quantidade")
    for i, (u, c) in enumerate(zip(unique, counts)):
        ax.text(i, c + 500, str(c), ha="center")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    y_true = [0, 1, 2, 0, 2, 1, 0]
    y_pred = [0, 1, 2, 0, 1, 1, 0]
    metrics = full_report(y_true, y_pred, "Test Model")
    print(f"\nMétricas: {metrics}")