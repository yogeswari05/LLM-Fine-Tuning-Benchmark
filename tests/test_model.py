"""
Unit tests for PEFT/LoRA parameter accounting, frozen weights, and adapter toggling.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from src.utils import calculate_parameter_statistics


class SimpleCausalLM(nn.Module):
    """Minimal model for fast CPU testing of LoRA mechanics."""
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def can_generate(self):
        return True

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        q = self.q_proj(x)
        v = self.v_proj(x)
        logits = self.out_proj(q + v)
        return type("Output", (), {"logits": logits, "loss": torch.tensor(1.0, requires_grad=True)})()


def test_lora_parameter_accounting_and_frozen_base():
    base_model = SimpleCausalLM()
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"]
    )

    peft_model = get_peft_model(base_model, lora_config)
    stats = calculate_parameter_statistics(peft_model)

    # 1. Total trainable parameters must be strictly greater than 0
    assert stats["trainable_params"] > 0
    assert stats["total_params"] > stats["trainable_params"]

    # 2. Check that base parameters are frozen
    base_frozen_count = 0
    lora_trainable_count = 0
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA parameter {name} should require grad!"
            lora_trainable_count += param.numel()
        else:
            assert param.requires_grad is False, f"Base parameter {name} should be frozen!"
            base_frozen_count += param.numel()

    assert stats["trainable_params"] == lora_trainable_count
    assert stats["frozen_params"] == base_frozen_count


def test_adapter_disable_context():
    base_model = SimpleCausalLM()
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"]
    )
    peft_model = get_peft_model(base_model, lora_config)

    input_ids = torch.tensor([[1, 2, 3, 4]])

    # Inference with adapter active
    peft_model.eval()
    with torch.no_grad():
        out_active = peft_model(input_ids).logits

    # Inference with adapter disabled (baseline mode)
    with torch.no_grad():
        with peft_model.disable_adapter():
            out_disabled = peft_model(input_ids).logits

    # Initially LoRA weights (lora_B) are initialized to 0, so out_active == out_disabled
    assert torch.allclose(out_active, out_disabled, atol=1e-5)

    # Manually modify lora_B to simulate trained adapter
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            param.data.fill_(1.0)

    # Now with adapter active vs disabled MUST be different
    with torch.no_grad():
        out_trained_adapter = peft_model(input_ids).logits
    with torch.no_grad():
        with peft_model.disable_adapter():
            out_base_untouched = peft_model(input_ids).logits

    diff = torch.abs(out_trained_adapter - out_base_untouched).sum().item()
    assert diff > 0.1, "Disabling adapter should revert model to untouched base behavior!"
