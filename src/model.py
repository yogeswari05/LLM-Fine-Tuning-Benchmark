"""
Model and tokenizer initialization, QLoRA/LoRA configuration, and parameter accounting.
"""

import os
from typing import Dict, Any, Tuple, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training
)
from .config import BenchmarkConfig
from .utils import calculate_parameter_statistics, get_compute_dtype


def initialize_tokenizer(
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    padding_side: str = "right"
) -> PreTrainedTokenizerBase:
    """
    Loads and configures the tokenizer.
    Enforces padding token assignment and explicit padding side.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side=padding_side
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|extra_0|>"})
            
    return tokenizer


def load_base_model(
    config: BenchmarkConfig
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Loads the untouched base causal language model.
    Configures quantization (4-bit QLoRA) if requested and supported.
    """
    tokenizer = initialize_tokenizer(
        model_name_or_path=config.model.model_name_or_path,
        padding_side="right"
    )

    compute_dtype = get_compute_dtype(config.model.torch_dtype)
    quantization_config = None

    if config.model.use_qlora and torch.cuda.is_available():
        try:
            bnb_compute_dtype = getattr(torch, config.model.quantization_4bit.get("bnb_4bit_compute_dtype", "bfloat16"))
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.model.quantization_4bit.get("load_in_4bit", True),
                bnb_4bit_quant_type=config.model.quantization_4bit.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=config.model.quantization_4bit.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=bnb_compute_dtype
            )
            print("Configured BitsAndBytesConfig 4-bit NF4 Quantization for QLoRA.")
        except Exception as e:
            print(f"Failed to configure BitsAndBytesConfig ({e}). Falling back to standard precision.")
            quantization_config = None

    device_map = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        torch_dtype=compute_dtype if quantization_config is None else None,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=config.model.trust_remote_code
    )

    if config.model.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def setup_peft_model(
    base_model: AutoModelForCausalLM,
    config: BenchmarkConfig
) -> Tuple[PeftModel, Dict[str, Any]]:
    """
    Wraps the base model with LoRA / QLoRA adapters and verifies parameter counts.
    """
    if config.model.use_qlora and torch.cuda.is_available():
        # Prepare model for k-bit training if quantized
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.model.gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules
    )

    peft_model = get_peft_model(base_model, lora_config)
    param_stats = calculate_parameter_statistics(peft_model)

    print(
        f"LoRA Adapter configured:\n"
        f"  Total Parameters: {param_stats['total_params']:,}\n"
        f"  Trainable Parameters: {param_stats['trainable_params']:,}\n"
        f"  Trainable Ratio: {param_stats['trainable_percent']:.4f}%\n"
        f"  Target Modules: {config.lora.target_modules}"
    )

    # Verification assertion: LoRA parameters must be > 0 and < 5% of base parameters
    assert param_stats["trainable_params"] > 0, "No trainable parameters found in PEFT model!"
    assert param_stats["trainable_percent"] < 10.0, (
        f"Trainable percent unexpectedly high ({param_stats['trainable_percent']:.2f}%). "
        "Base parameters might not be properly frozen."
    )

    return peft_model, param_stats


def load_adapter_for_inference(
    base_model: AutoModelForCausalLM,
    adapter_path: str
) -> PeftModel:
    """
    Attaches saved trained LoRA adapter weights to the base model for inference.
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter checkpoint not found at '{adapter_path}'")
        
    print(f"Loading trained adapter from '{adapter_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.eval()
    return peft_model
