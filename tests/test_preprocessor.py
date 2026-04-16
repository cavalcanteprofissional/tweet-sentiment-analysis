"""Testes unitários para o preprocessor"""

import pytest
from src.preprocessor import clean_tweet


def test_remove_url():
    text = "Check this https://t.co/example great link"
    assert "http" not in clean_tweet(text)


def test_normalize_mention():
    text = "@user1 hello"
    cleaned = clean_tweet(text)
    assert "@user1" not in cleaned
    assert "@user" in cleaned


def test_normalize_hashtag():
    text = "#happy birthday"
    cleaned = clean_tweet(text)
    assert "happy" in cleaned
    assert "#" not in cleaned


def test_reduce_repeated_chars():
    text = "sooooo coooool"
    cleaned = clean_tweet(text)
    assert "oo" not in cleaned or len(cleaned) < len(text)


def test_preprocess_batch():
    texts = ["hello", "test @user"]
    results = clean_tweet(texts[0]), clean_tweet(texts[1])
    assert all(isinstance(r, str) for r in results)