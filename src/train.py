"""
Supervised fine-tuning pipeline, small-batch overfitting sanity test, and checkpoint saving.
"""

import os
import time
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedTokenizerBase
)
from datasets import Dataset
from peft import PeftModel
from .config import BenchmarkConfig
from .loss_masking import CausalLMDataCollator
from .utils import print_gpu_memory, format_duration, save_json


class MetricsAndMemoryCallback(TrainerCallback):
    """
    Custom callback to log GPU memory, learning rate, and training step duration.
    """
    def __init__(self):
        super().__init__()
        self.step_start_time = None
        self.history = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = dict(logs)
        entry["step"] = state.global_step
        entry["epoch"] = state.epoch
        if self.step_start_time is not None:
            entry["step_duration_sec"] = time.time() - self.step_start_time
            
        mem = print_gpu_memory(tag=f"Step {state.global_step}")
        entry.update(mem)
        self.history.append(entry)


def run_overfitting_sanity_check(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    sanity_dataset: Dataset,
    config: BenchmarkConfig
) -> Dict[str, Any]:
    """
    Runs an explicit small-batch overfitting sanity check on 2-4 examples.
    Confirms that:
    1. Gradients flow into LoRA weights without breaking.
    2. Loss steadily decreases over multiple gradient steps.
    3. Label shifting and loss masking allow the model to learn.
    """
    print("\n" + "="*70)
    print("RUNNING SMALL-BATCH OVERFITTING SANITY CHECK")
    print("="*70)
    
    num_samples = min(config.training.sanity_check_samples, len(sanity_dataset))
    mini_ds = sanity_dataset.select(range(num_samples))
    collator = CausalLMDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(mini_ds, batch_size=num_samples, collate_fn=collator)
    batch = next(iter(dataloader))

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    # Use standard AdamW optimizer for the sanity check
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4
    )

    model.train()
    loss_history = []
    num_steps = config.training.sanity_check_steps

    print(f"Sanity check: optimizing {num_samples} examples over {num_steps} steps...")
    start_time = time.time()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % 5 == 0 or step == 1 or step == num_steps:
            print(f"  [Sanity Step {step:2d}/{num_steps}] Loss: {loss_val:.4f}")

    elapsed = time.time() - start_time
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    loss_reduction = initial_loss - final_loss
    relative_reduction = (loss_reduction / initial_loss) * 100 if initial_loss > 0 else 0

    print(
        f"Sanity check completed in {format_duration(elapsed)}.\n"
        f"Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f} "
        f"({relative_reduction:.1f}% reduction)"
    )

    sanity_passed = final_loss < initial_loss and (final_loss < 1.5 or relative_reduction > 40.0)
    if sanity_passed:
        print("[SANITY CHECK PASSED] Model successfully overfits the mini-batch.")
    else:
        print("[SANITY CHECK WARNING] Loss did not decrease as expected. Check learning rate or mask.")

    return {
        "passed": bool(sanity_passed),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_reduction": float(loss_reduction),
        "relative_reduction_pct": float(relative_reduction),
        "loss_history": [float(x) for x in loss_history],
        "duration_sec": elapsed
    }


def train_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: BenchmarkConfig
) -> tuple[PeftModel, Dict[str, Any]]:
    """
    Executes supervised fine-tuning using HuggingFace Trainer with custom
    collator, gradient accumulation, mixed precision, and metrics logging.
    """
    print("\n" + "="*70)
    print("STARTING FULL SUPERVISED FINE-TUNING")
    print("="*70)

    collator = CausalLMDataCollator(tokenizer=tokenizer)
    metrics_callback = MetricsAndMemoryCallback()

    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.logging_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        logging_dir=config.training.logging_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps" if config.training.eval_steps > 0 else "no",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        optim=config.training.optimizer,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to=["none"],
        logging_first_step=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[metrics_callback]
    )

    start_time = time.time()
    train_result = trainer.train()
    total_duration = time.time() - start_time

    print(f"\nTraining completed in {format_duration(total_duration)}.")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    # Save final LoRA adapter weights and tokenizer
    final_adapter_dir = os.path.join(config.training.output_dir, "final_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Saved trained LoRA adapter to '{final_adapter_dir}'")

    # Collect and save training history
    training_summary = {
        "train_loss": float(train_result.training_loss),
        "total_duration_sec": total_duration,
        "global_step": train_result.global_step,
        "metrics_history": metrics_callback.history,
        "adapter_path": final_adapter_dir
    }
    save_json(training_summary, os.path.join(config.training.output_dir, "training_summary.json"))

    return model, training_summary
