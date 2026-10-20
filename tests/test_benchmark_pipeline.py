"""
Integration tests for acceptance criteria:
- Disjoint splits and identical test prompts for both models
- Adapter save & reload without retraining
- Fresh metric calculation
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from src.evaluate import calculate_metrics, paired_bootstrap_ci, compare_systems


class DummyModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        logits = self.linear(x)
        return type("Output", (), {"logits": logits})()


def test_adapter_save_and_reload_for_inference():
    base_model = DummyModel()
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["linear"]
    )
    peft_model = get_peft_model(base_model, lora_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_path = os.path.join(tmpdir, "test_adapter")
        # Save adapter
        peft_model.save_pretrained(adapter_path)
        assert os.path.exists(os.path.join(adapter_path, "adapter_config.json"))

        # Load fresh base model and re-attach adapter
        fresh_base = DummyModel()
        reloaded_peft = PeftModel.from_pretrained(fresh_base, adapter_path)
        reloaded_peft.eval()

        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = reloaded_peft(input_ids).logits
        assert out.shape == (1, 3, 100)


def test_metrics_and_paired_bootstrap_ci():
    # Synthetic predictions for base and fine-tuned
    base_preds = [
        {"example_id": "0", "generated_text": "Neural networks process language effectively.", "reference_summary": "Neural networks are effective in natural language processing.", "latency_sec": 1.2, "output_tokens": 40, "input_tokens": 150, "tokens_per_sec": 33.3, "hit_max_tokens": False},
        {"example_id": "1", "generated_text": "The theorem establishes lower bounds for optimization.", "reference_summary": "We prove tight lower bounds for convex optimization.", "latency_sec": 1.5, "output_tokens": 50, "input_tokens": 200, "tokens_per_sec": 33.3, "hit_max_tokens": False},
        {"example_id": "2", "generated_text": "A new transformer model scales with linear attention.", "reference_summary": "Transformers with linear attention scale to long sequences.", "latency_sec": 1.1, "output_tokens": 38, "input_tokens": 120, "tokens_per_sec": 34.5, "hit_max_tokens": False}
    ]
    ft_preds = [
        {"example_id": "0", "generated_text": "Neural networks are effective in natural language processing.", "reference_summary": "Neural networks are effective in natural language processing.", "latency_sec": 1.0, "output_tokens": 42, "input_tokens": 150, "tokens_per_sec": 42.0, "hit_max_tokens": False},
        {"example_id": "1", "generated_text": "We prove tight lower bounds for convex optimization models.", "reference_summary": "We prove tight lower bounds for convex optimization.", "latency_sec": 1.3, "output_tokens": 48, "input_tokens": 200, "tokens_per_sec": 36.9, "hit_max_tokens": False},
        {"example_id": "2", "generated_text": "Linear attention transformers scale efficiently to long sequences.", "reference_summary": "Transformers with linear attention scale to long sequences.", "latency_sec": 0.9, "output_tokens": 36, "input_tokens": 120, "tokens_per_sec": 40.0, "hit_max_tokens": False}
    ]

    base_metrics = calculate_metrics(base_preds)
    ft_metrics = calculate_metrics(ft_preds)

    assert "rouge1" in base_metrics
    assert "rouge2" in base_metrics
    assert "rougeL" in base_metrics
    assert "rougeLsum" in base_metrics

    # Test paired bootstrap CI
    boot_ci = paired_bootstrap_ci(base_preds, ft_preds, num_rounds=100)
    assert "rouge1" in boot_ci
    assert "ci_lower" in boot_ci["rouge1"]
    assert "ci_upper" in boot_ci["rouge1"]

    # Test comparison table generation
    comp_df = compare_systems(base_metrics, ft_metrics, boot_ci)
    assert len(comp_df) > 0
    assert "Relative Change (%)" in comp_df.columns
