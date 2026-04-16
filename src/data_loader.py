"""Data Loader - Carregamento do dataset tweet_eval via Hugging Face Datasets"""

from datasets import load_dataset
from typing import Optional
import pandas as pd


def load_tweet_eval(split: str = "train") -> pd.DataFrame:
    """
    Carrega o dataset tweet_eval do Hugging Face Hub.
    
    Args:
        split: one of 'train', 'validation', or 'test'
    
    Returns:
        pd.DataFrame com colunas 'text' e 'label'
    """
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    df = pd.DataFrame(ds)
    return df


def load_all_splits() -> dict[str, pd.DataFrame]:
    """
    Carrega todos os splits do dataset tweet_eval.
    
    Returns:
        dict com chaves 'train', 'validation', 'test'
    """
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    return {split: pd.DataFrame(ds[split]) for split in ds}


def get_label_names() -> dict[int, str]:
    """Mapeamento de labels numéricos para nomes."""
    return {0: "negative", 1: "neutral", 2: "positive"}


if __name__ == "__main__":
    print("Carregando tweet_eval...")
    splits = load_all_splits()
    for name, df in splits.items():
        print(f"{name}: {len(df)} samples")
        print(df["label"].value_counts().sort_index())