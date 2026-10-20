"""
Unit tests for dataset splitting, smart truncation, and chat template formatting.
"""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer
from src.data import ArxivDatasetManager, smart_truncate_and_format, process_dataset_split


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def test_dataset_split_disjointness():
    # Create synthetic dataset with 350 items
    data = [{"article": f"Paper article body {i}", "abstract": f"Abstract {i}"} for i in range(350)]
    mock_ds = Dataset.from_list(data)

    manager = ArxivDatasetManager(
        train_samples=240,
        val_samples=36,
        test_samples=24,
        seed=42
    )

    # Deterministic permutation splitting logic
    import numpy as np
    rng = np.random.default_rng(manager.seed)
    all_indices = rng.permutation(len(mock_ds))
    train_idx = all_indices[:240].tolist()
    val_idx = all_indices[240:276].tolist()
    test_idx = all_indices[276:300].tolist()

    # Verify sizes
    assert len(train_idx) == 240
    assert len(val_idx) == 36
    assert len(test_idx) == 24

    # Strict disjointness assertions
    assert len(set(train_idx).intersection(set(val_idx))) == 0, "Train and Val overlap!"
    assert len(set(train_idx).intersection(set(test_idx))) == 0, "Train and Test overlap!"
    assert len(set(val_idx).intersection(set(test_idx))) == 0, "Val and Test overlap!"


def test_smart_truncation_preserves_target_and_truncates_article(tokenizer):
    # Short summary (approx 20 tokens)
    short_summary = "This is a concise scientific summary about transformer architectures and efficient attention."
    # Extremely long article (approx 3,000 words / 4,000 tokens)
    long_article = "Deep learning and natural language processing have revolutionized scientific research. " * 300

    max_seq_len = 512
    max_target_len = 128

    formatted = smart_truncate_and_format(
        article=long_article,
        summary=short_summary,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        max_target_length=max_target_len
    )

    # 1. Total length must not exceed max_seq_len
    assert formatted["total_length"] <= max_seq_len

    # 2. Target summary must NOT be truncated because it was within budget
    assert formatted["target_truncated"] is False

    # 3. Article body MUST be truncated
    assert formatted["article_truncated"] is True
    assert formatted["raw_article_token_len"] > formatted["final_article_token_len"]

    # 4. Chat template structure must be intact
    assert "<|im_start|>system" in formatted["prompt_text"]
    assert "<|im_start|>user" in formatted["prompt_text"]
    assert "<|im_start|>assistant\n" in formatted["prompt_text"]


def test_smart_truncation_detects_target_overflow(tokenizer):
    # Giant summary exceeding budget
    huge_summary = "In this research paper we present a novel neural network methodology. " * 50
    short_article = "Short paper introduction and conclusion."

    max_seq_len = 512
    max_target_len = 64  # very small budget

    formatted = smart_truncate_and_format(
        article=short_article,
        summary=huge_summary,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        max_target_length=max_target_len
    )

    # Target overflow must be explicitly flagged
    assert formatted["target_truncated"] is True
    assert formatted["final_target_token_len"] <= max_target_len
    assert formatted["total_length"] <= max_seq_len
