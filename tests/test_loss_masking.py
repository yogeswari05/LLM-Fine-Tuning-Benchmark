"""
Unit tests for loss masking, causal LM label shifting, and data collation.
"""

import pytest
import torch
from transformers import AutoTokenizer
from src.loss_masking import build_masked_inputs, CausalLMDataCollator, verify_loss_shifting


@pytest.fixture(scope="module")
def tokenizer():
    # Use fast tokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def test_prompt_masking_with_minus_100(tokenizer):
    prompt_text = "<|im_start|>system\nYou are a summarizer.<|im_end|>\n<|im_start|>user\nSummarize this paper.<|im_end|>\n<|im_start|>assistant\n"
    target_text = "This paper presents a new model.<|im_end|>\n"
    
    result = build_masked_inputs(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        max_seq_length=512
    )

    prompt_len = result["prompt_length"]
    labels = result["labels"]
    input_ids = result["input_ids"]

    # 1. Prompt tokens MUST have label -100
    assert len(labels) == len(input_ids)
    assert all(lbl == -100 for lbl in labels[:prompt_len]), "Prompt tokens were not masked with -100!"

    # 2. Target tokens MUST have exact matching IDs
    target_labels = labels[prompt_len:]
    target_ids = input_ids[prompt_len:]
    assert len(target_labels) > 0, "No target tokens supervised!"
    assert target_labels == target_ids, "Target labels do not match input IDs!"

    # 3. EOS token MUST be supervised
    eos_ids = [tokenizer.eos_token_id]
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_ids.extend(im_end_ids)
    has_eos_supervised = any(t in eos_ids for t in target_labels)
    assert has_eos_supervised, "End-of-sequence token was not supervised in target labels!"


def test_causal_lm_data_collator(tokenizer):
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    item1 = {
        "input_ids": [10, 20, 30],
        "attention_mask": [1, 1, 1],
        "labels": [-100, -100, 30]
    }
    item2 = {
        "input_ids": [10, 20, 30, 40, 50],
        "attention_mask": [1, 1, 1, 1, 1],
        "labels": [-100, -100, 30, 40, 50]
    }

    batch = collator([item1, item2])

    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)

    # Padding in item1
    pad_id = tokenizer.pad_token_id
    assert batch["input_ids"][0, 3].item() == pad_id
    assert batch["input_ids"][0, 4].item() == pad_id

    # Attention mask for padding must be 0
    assert batch["attention_mask"][0, 3].item() == 0
    assert batch["attention_mask"][0, 4].item() == 0

    # Labels for padding must be -100
    assert batch["labels"][0, 3].item() == -100
    assert batch["labels"][0, 4].item() == -100


def test_causal_loss_shifting_mathematical_identity():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 10
    vocab_size = 50

    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    # Mask first 4 tokens as prompt (-100) and last 2 tokens as padding (-100)
    labels = torch.tensor([
        [-100, -100, -100, -100, 15, 22, 35, 42, -100, -100],
        [-100, -100, -100, -100, 10, 18, 29, -100, -100, -100]
    ], dtype=torch.long)

    verification = verify_loss_shifting(logits, labels)

    assert verification["is_numerically_identical"], (
        f"Loss calculation discrepancy! HF Loss: {verification['hf_loss']}, "
        f"Manual Loss: {verification['manual_loss']}, Diff: {verification['abs_diff']}"
    )
    assert verification["num_supervised_tokens"] > 0
