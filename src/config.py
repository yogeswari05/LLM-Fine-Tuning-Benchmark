"""
Configuration management for LLM Fine-Tuning and Evaluation Benchmark.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import List, Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    use_qlora: bool = True
    quantization_4bit: Dict[str, Any] = field(default_factory=lambda: {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16"
    })
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class DatasetConfig:
    dataset_name: str = "ccdv/arxiv-summarization"
    article_column: str = "article"
    summary_column: str = "abstract"
    train_samples: int = 240
    val_samples: int = 36
    test_samples: int = 24
    max_seq_length: int = 1024
    max_target_length: int = 256
    system_prompt: str = (
        "You are an expert scientific researcher. Summarize the following research paper "
        "article concisely and accurately. Focus on the core problem, methodology, key findings, "
        "and contributions."
    )
    user_prompt_template: str = "Please summarize the following scientific paper:\n\n{article}"


@dataclass
class TrainingConfig:
    output_dir: str = "./artifacts/checkpoints"
    logging_dir: str = "./artifacts/logs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 5
    eval_steps: int = 15
    save_steps: int = 30
    save_total_limit: int = 2
    fp16: bool = False
    bf16: bool = True
    optimizer: str = "paged_adamw_8bit"
    dataloader_num_workers: int = 0
    sanity_check_samples: int = 4
    sanity_check_steps: int = 30


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    min_new_tokens: int = 32
    do_sample: bool = False
    repetition_penalty: float = 1.05
    temperature: float = 1.0
    top_p: float = 1.0
    warmup_runs: int = 2


@dataclass
class EvaluationConfig:
    bootstrap_rounds: int = 1000
    confidence_level: float = 0.95
    short_output_threshold: int = 30
    save_predictions_dir: str = "./artifacts/predictions"
    reports_dir: str = "./artifacts/reports"


@dataclass
class BenchmarkConfig:
    project_name: str = "scientific-summarization-benchmark"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            project_name=data.get("project_name", "scientific-summarization-benchmark"),
            seed=data.get("seed", 42),
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            training=TrainingConfig(**data.get("training", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {}))
        )

    def to_json(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def adapt_to_environment(self) -> None:
        """
        Dynamically adjusts hardware-dependent parameters (CUDA, bfloat16, QLoRA)
        based on runtime hardware capabilities.
        """
        has_cuda = torch.cuda.is_available()
        
        if not has_cuda:
            # CPU runtime
            self.model.use_qlora = False
            self.model.torch_dtype = "float32"
            self.training.fp16 = False
            self.training.bf16 = False
            self.training.optimizer = "adamw_torch"
            self.model.gradient_checkpointing = False
        else:
            # GPU runtime
            supports_bf16 = torch.cuda.is_bf16_supported()
            if not supports_bf16:
                self.model.torch_dtype = "float16"
                self.training.fp16 = True
                self.training.bf16 = False
                self.model.quantization_4bit["bnb_4bit_compute_dtype"] = "float16"
            else:
                self.model.torch_dtype = "bfloat16"
                self.training.fp16 = False
                self.training.bf16 = True
                self.model.quantization_4bit["bnb_4bit_compute_dtype"] = "bfloat16"
            
            # Check bitsandbytes availability
            try:
                import bitsandbytes as bnb
            except ImportError:
                self.model.use_qlora = False
                self.training.optimizer = "adamw_torch"
