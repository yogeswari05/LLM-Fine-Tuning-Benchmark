"""
Main execution entrypoint for LLM Fine-Tuning and Evaluation Benchmark.
"""

import os
import argparse
import time
import json
import torch
import pandas as pd
from tabulate import tabulate

from src.config import BenchmarkConfig
from src.utils import set_seed, save_json, load_json, format_duration
from src.data import ArxivDatasetManager, process_dataset_split
from src.model import load_base_model, setup_peft_model, load_adapter_for_inference
from src.train import run_overfitting_sanity_check, train_model
from src.evaluate import (
    benchmark_generation,
    calculate_metrics,
    paired_bootstrap_ci,
    compare_systems
)
from src.error_analysis import (
    audit_prediction_pair,
    generate_error_analysis_report,
    generate_benchmark_plots
)


def run_pipeline(config_path: str, mode: str = "all") -> None:
    print("\n" + "="*80)
    print(" LLM FINE-TUNING & EVALUATION BENCHMARK: SCIENTIFIC SUMMARIZATION ")
    print("="*80)

    # 1. Load and Adapt Configuration
    config = BenchmarkConfig.from_json(config_path)
    config.adapt_to_environment()
    set_seed(config.seed)
    
    # Save active config artifact
    os.makedirs("./artifacts", exist_ok=True)
    config.to_json("./artifacts/active_config.json")
    print(f"Loaded config from '{config_path}' (Adapted to: {config.model.torch_dtype}, CUDA: {torch.cuda.is_available()})")

    # 2. Data Preparation & Splitting
    dataset_mgr = ArxivDatasetManager(
        dataset_name=config.dataset.dataset_name,
        article_column=config.dataset.article_column,
        summary_column=config.dataset.summary_column,
        train_samples=config.dataset.train_samples,
        val_samples=config.dataset.val_samples,
        test_samples=config.dataset.test_samples,
        seed=config.seed
    )
    train_raw, val_raw, test_raw, split_meta = dataset_mgr.load_and_split(
        splits_save_path="./artifacts/dataset_splits.json"
    )

    # 3. Model & Tokenizer Initialization
    base_model, tokenizer = load_base_model(config)

    # Tokenize and format splits
    print("\nProcessing and smart-truncating splits with chat template...")
    train_ds, train_stats = process_dataset_split(
        dataset=train_raw,
        tokenizer=tokenizer,
        max_seq_length=config.dataset.max_seq_length,
        max_target_length=config.dataset.max_target_length,
        article_column=config.dataset.article_column,
        summary_column=config.dataset.summary_column,
        system_prompt=config.dataset.system_prompt,
        user_prompt_template=config.dataset.user_prompt_template
    )
    val_ds, val_stats = process_dataset_split(
        dataset=val_raw,
        tokenizer=tokenizer,
        max_seq_length=config.dataset.max_seq_length,
        max_target_length=config.dataset.max_target_length,
        article_column=config.dataset.article_column,
        summary_column=config.dataset.summary_column,
        system_prompt=config.dataset.system_prompt,
        user_prompt_template=config.dataset.user_prompt_template
    )
    test_ds, test_stats = process_dataset_split(
        dataset=test_raw,
        tokenizer=tokenizer,
        max_seq_length=config.dataset.max_seq_length,
        max_target_length=config.dataset.max_target_length,
        article_column=config.dataset.article_column,
        summary_column=config.dataset.summary_column,
        system_prompt=config.dataset.system_prompt,
        user_prompt_template=config.dataset.user_prompt_template
    )

    print(
        f"Data Preprocessing Summary:\n"
        f"  Train: {len(train_ds)} items | Mean tokens: {train_stats['mean_total_tokens']:.1f} | "
        f"Article Truncation Rate: {train_stats['article_truncation_rate']*100:.1f}% | "
        f"Target Truncation Rate: {train_stats['target_truncation_rate']*100:.1f}%\n"
        f"  Val:   {len(val_ds)} items | Mean tokens: {val_stats['mean_total_tokens']:.1f}\n"
        f"  Test:  {len(test_ds)} items | Mean tokens: {test_stats['mean_total_tokens']:.1f}"
    )
    save_json({"train": train_stats, "val": val_stats, "test": test_stats}, "./artifacts/token_truncation_stats.json")

    if mode == "data":
        print("Data preparation complete. Exiting per mode='data'.")
        return

    # 4. LoRA Setup & Parameter Accounting
    peft_model, param_stats = setup_peft_model(base_model, config)
    save_json(param_stats, "./artifacts/parameter_statistics.json")

    # 5. Overfitting Sanity Check
    if mode in ("all", "sanity"):
        sanity_result = run_overfitting_sanity_check(
            model=peft_model,
            tokenizer=tokenizer,
            sanity_dataset=train_ds,
            config=config
        )
        save_json(sanity_result, "./artifacts/sanity_check_result.json")
        if mode == "sanity":
            print("Sanity check complete. Exiting per mode='sanity'.")
            return

    # 6. Full Training
    adapter_path = os.path.join(config.training.output_dir, "final_adapter")
    if mode in ("all", "train"):
        peft_model, train_summary = train_model(
            model=peft_model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config
        )

    # 7. Model Evaluation & Fair Comparison
    if mode in ("all", "eval"):
        print("\n" + "="*70)
        print("RIGOROUS BENCHMARK EVALUATION (BASE VS. FINE-TUNED)")
        print("="*70)

        # A. Evaluate untouched base model (adapter explicitly disabled)
        base_preds = benchmark_generation(
            model=peft_model,
            tokenizer=tokenizer,
            test_dataset=test_ds,
            config=config,
            system_name="base_model"
        )
        save_json(base_preds, os.path.join(config.evaluation.save_predictions_dir, "base_model_predictions.json"))

        # B. Evaluate fine-tuned model (adapter enabled)
        ft_preds = benchmark_generation(
            model=peft_model,
            tokenizer=tokenizer,
            test_dataset=test_ds,
            config=config,
            system_name="fine_tuned_model"
        )
        save_json(ft_preds, os.path.join(config.evaluation.save_predictions_dir, "fine_tuned_predictions.json"))

        # Calculate metrics
        base_metrics = calculate_metrics(base_preds, config.evaluation.short_output_threshold)
        ft_metrics = calculate_metrics(ft_preds, config.evaluation.short_output_threshold)

        save_json(base_metrics, "./artifacts/base_metrics.json")
        save_json(ft_metrics, "./artifacts/fine_tuned_metrics.json")

        # Paired bootstrap confidence intervals
        print(f"\nComputing paired bootstrap confidence intervals ({config.evaluation.bootstrap_rounds} rounds)...")
        boot_ci = paired_bootstrap_ci(
            base_preds=base_preds,
            ft_preds=ft_preds,
            num_rounds=config.evaluation.bootstrap_rounds,
            alpha=1.0 - config.evaluation.confidence_level,
            seed=config.seed
        )
        save_json(boot_ci, "./artifacts/bootstrap_confidence_intervals.json")

        # Comparison DataFrame
        comparison_df = compare_systems(base_metrics, ft_metrics, boot_ci)
        comparison_df.to_csv("./artifacts/model_comparison.csv", index=False)
        print("\n" + tabulate(comparison_df, headers="keys", tablefmt="github", showindex=False))

        # 8. Qualitative Error Analysis & Visualizations
        print("\nPerforming qualitative error audits and rendering plots...")
        audits = []
        for i in range(len(test_ds)):
            audit_item = audit_prediction_pair(
                example_id=test_ds[i]["example_id"],
                reference=test_ds[i].get("abstract", test_ds[i].get("target_text", "")),
                base_pred=base_preds[i],
                ft_pred=ft_preds[i]
            )
            audits.append(audit_item)

        generate_error_analysis_report(audits, os.path.join(config.evaluation.reports_dir, "qualitative_error_analysis.md"))
        generate_benchmark_plots(base_metrics, ft_metrics, base_preds, ft_preds, config.evaluation.reports_dir)

    print("\n" + "="*80)
    print(" BENCHMARK COMPLETED SUCCESSFULLY. ALL ARTIFACTS SAVED TO ./artifacts ")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning & Evaluation Benchmark")
    parser.add_argument("--config", type=str, default="configs/benchmark_config.json", help="Path to config JSON")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "data", "sanity", "train", "eval"], help="Execution mode")
    args = parser.parse_args()
    run_pipeline(args.config, args.mode)
