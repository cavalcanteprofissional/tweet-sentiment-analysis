"""Preprocessor - Limpeza específica para tweets"""

import re


def clean_tweet(text: str) -> str:
    """
    Limpeza focada em preservar semântica emocional.
    NÃO remove emojis — eles carregam sentimento.
    
    Args:
        text: texto original do tweet
    
    Returns:
        texto limpo
    """
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_batch(texts: list[str]) -> list[str]:
    """
    Aplica clean_tweet a uma lista de textos.
    
    Args:
        texts: lista de textos originais
    
    Returns:
        lista de textos limpos
    """
    return [clean_tweet(text) for text in texts]


if __name__ == "__main__":
    test_tweets = [
        "Check out this https://t.co/example amazing product! 🎉",
        "@user1 @user2 love this so much!!! #happy",
        "This is soooooo cooooolLLL",
    ]
    for tweet in test_tweets:
        print(f"Original: {tweet}")
        print(f"Cleaned:  {clean_tweet(tweet)}")
        print()