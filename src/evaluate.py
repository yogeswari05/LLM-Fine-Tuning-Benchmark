"""
Generation benchmarking, latency measurement, ROUGE evaluation, and paired bootstrap statistics.
"""

import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import evaluate
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
from .config import BenchmarkConfig
from .utils import save_json


def generate_single_example(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    max_new_tokens: int = 256,
    min_new_tokens: int = 32,
    do_sample: bool = False,
    repetition_penalty: float = 1.05,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> Dict[str, Any]:
    """
    Executes synchronized deterministic generation for a single prompt.
    Measures CUDA latency precisely.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    input_len = input_ids.shape[1]

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Synchronize after generation completes
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    latency_sec = t_end - t_start

    # Extract only the newly generated tokens
    new_tokens = output_ids[0, input_len:]
    output_len = len(new_tokens)

    # Decode
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Determine termination status
    eos_ids = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "encode"):
        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eos_ids.extend(im_end_ids)
    
    ended_with_eos = False
    if output_len > 0:
        last_token = new_tokens[-1].item()
        ended_with_eos = (last_token in eos_ids)

    hit_max_tokens = (output_len >= max_new_tokens)
    tokens_per_sec = (output_len / latency_sec) if latency_sec > 0 else 0.0

    return {
        "generated_text": generated_text,
        "input_tokens": input_len,
        "output_tokens": output_len,
        "latency_sec": latency_sec,
        "tokens_per_sec": tokens_per_sec,
        "ended_with_eos": ended_with_eos,
        "hit_max_tokens": hit_max_tokens
    }


def benchmark_generation(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    test_dataset: Dataset,
    config: BenchmarkConfig,
    system_name: str = "base_model"
) -> List[Dict[str, Any]]:
    """
    Runs generation benchmark across all test examples.
    Supports warmup runs and records per-example generation metrics.
    """
    print(f"\nBenchmarking generation for system: '{system_name}' ({len(test_dataset)} examples)...")
    model.eval()

    # Optional adapter disable context check
    is_base = (system_name == "base_model")
    has_adapter = hasattr(model, "disable_adapter")

    # Warmup
    if config.generation.warmup_runs > 0 and len(test_dataset) > 0:
        warmup_prompt = test_dataset[0]["prompt_text"]
        print(f"Running {config.generation.warmup_runs} warmup generation(s)...")
        for _ in range(config.generation.warmup_runs):
            if is_base and has_adapter:
                with model.disable_adapter():
                    _ = generate_single_example(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=warmup_prompt,
                        max_new_tokens=32,
                        do_sample=False
                    )
            else:
                _ = generate_single_example(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=warmup_prompt,
                    max_new_tokens=32,
                    do_sample=False
                )

    results = []
    for idx, example in enumerate(test_dataset):
        prompt_text = example["prompt_text"]
        reference_summary = example.get("abstract", example.get("target_text", ""))
        # Clean reference summary if it has chat template syntax
        if "<|im_end|>" in reference_summary:
            reference_summary = reference_summary.split("<|im_end|>")[0].strip()

        if is_base and has_adapter:
            with model.disable_adapter():
                gen_data = generate_single_example(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=config.generation.max_new_tokens,
                    min_new_tokens=config.generation.min_new_tokens,
                    do_sample=config.generation.do_sample,
                    repetition_penalty=config.generation.repetition_penalty
                )
        else:
            gen_data = generate_single_example(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                max_new_tokens=config.generation.max_new_tokens,
                min_new_tokens=config.generation.min_new_tokens,
                do_sample=config.generation.do_sample,
                repetition_penalty=config.generation.repetition_penalty
            )

        gen_data.update({
            "example_id": example.get("example_id", str(idx)),
            "system_name": system_name,
            "reference_summary": reference_summary,
            "prompt_text": prompt_text
        })
        results.append(gen_data)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(test_dataset):
            print(
                f"  [{idx+1:2d}/{len(test_dataset)}] "
                f"Latency: {gen_data['latency_sec']:.2f}s | "
                f"Out Tokens: {gen_data['output_tokens']} | "
                f"Speed: {gen_data['tokens_per_sec']:.1f} tok/s"
            )

    return results


def calculate_metrics(
    predictions: List[Dict[str, Any]],
    short_output_threshold: int = 30
) -> Dict[str, Any]:
    """
    Computes ROUGE-1/2/L/Lsum, latency percentiles, length distributions,
    throughput, and truncation rates.
    """
    preds = [p["generated_text"] for p in predictions]
    refs = [p["reference_summary"] for p in predictions]

    # Compute ROUGE
    rouge_metric = evaluate.load("rouge")
    rouge_scores = rouge_metric.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    latencies = [p["latency_sec"] for p in predictions]
    output_tokens = [p["output_tokens"] for p in predictions]
    input_tokens = [p["input_tokens"] for p in predictions]
    tokens_per_sec = [p["tokens_per_sec"] for p in predictions]

    total_examples = len(predictions)
    hit_max_count = sum(1 for p in predictions if p["hit_max_tokens"])
    short_count = sum(1 for p in predictions if p["output_tokens"] < short_output_threshold)
    empty_count = sum(1 for p in predictions if len(p["generated_text"].strip()) == 0)

    summary = {
        "rouge1": float(rouge_scores["rouge1"]),
        "rouge2": float(rouge_scores["rouge2"]),
        "rougeL": float(rouge_scores["rougeL"]),
        "rougeLsum": float(rouge_scores["rougeLsum"]),
        "latency_mean_sec": float(np.mean(latencies)),
        "latency_median_sec": float(np.median(latencies)),
        "latency_p95_sec": float(np.percentile(latencies, 95)),
        "mean_input_tokens": float(np.mean(input_tokens)),
        "mean_output_tokens": float(np.mean(output_tokens)),
        "median_output_tokens": float(np.median(output_tokens)),
        "mean_tokens_per_sec": float(np.mean(tokens_per_sec)),
        "pct_hit_max_tokens": float(100.0 * hit_max_count / total_examples),
        "pct_short_outputs": float(100.0 * short_count / total_examples),
        "pct_empty_outputs": float(100.0 * empty_count / total_examples),
        "total_evaluated": total_examples
    }

    return summary


def paired_bootstrap_ci(
    base_preds: List[Dict[str, Any]],
    ft_preds: List[Dict[str, Any]],
    num_rounds: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Computes paired bootstrap confidence intervals and empirical p-values
    for ROUGE metric differences (FineTuned - Base).
    """
    rouge_metric = evaluate.load("rouge")
    n = len(base_preds)
    rng = np.random.default_rng(seed)

    # Compute per-example scores first
    base_per_example = []
    ft_per_example = []
    for b, f in zip(base_preds, ft_preds):
        b_res = rouge_metric.compute(
            predictions=[b["generated_text"]],
            references=[b["reference_summary"]],
            use_stemmer=True
        )
        f_res = rouge_metric.compute(
            predictions=[f["generated_text"]],
            references=[f["reference_summary"]],
            use_stemmer=True
        )
        base_per_example.append(b_res)
        ft_per_example.append(f_res)

    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    ci_results = {}

    for m in metrics:
        b_scores = np.array([x[m] for x in base_per_example])
        f_scores = np.array([x[m] for x in ft_per_example])
        diffs = f_scores - b_scores
        observed_mean_diff = float(np.mean(diffs))

        bootstrap_diffs = []
        for _ in range(num_rounds):
            boot_idx = rng.integers(0, n, size=n)
            bootstrap_diffs.append(np.mean(diffs[boot_idx]))

        bootstrap_diffs = np.array(bootstrap_diffs)
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - (alpha / 2.0)) * 100.0
        ci_lower = float(np.percentile(bootstrap_diffs, lower_pct))
        ci_upper = float(np.percentile(bootstrap_diffs, upper_pct))

        # Empirical two-tailed p-value
        p_val = 2.0 * min(np.mean(bootstrap_diffs <= 0), np.mean(bootstrap_diffs >= 0))
        p_val = float(min(1.0, p_val))

        ci_results[m] = {
            "observed_mean_diff": observed_mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_val,
            "is_significant": bool(ci_lower > 0 or ci_upper < 0)
        }

    return ci_results


def compare_systems(
    base_metrics: Dict[str, Any],
    ft_metrics: Dict[str, Any],
    bootstrap_ci: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """
    Creates a comprehensive comparison table showing base vs fine-tuned metrics,
    absolute delta, percentage change, and interpretation.
    """
    rows = []
    keys_to_compare = [
        ("ROUGE-1", "rouge1", "higher_is_better"),
        ("ROUGE-2", "rouge2", "higher_is_better"),
        ("ROUGE-L", "rougeL", "higher_is_better"),
        ("ROUGE-Lsum", "rougeLsum", "higher_is_better"),
        ("Mean Latency (s)", "latency_mean_sec", "lower_is_better"),
        ("Median Latency (s)", "latency_median_sec", "lower_is_better"),
        ("P95 Latency (s)", "latency_p95_sec", "lower_is_better"),
        ("Mean Output Tokens", "mean_output_tokens", "neutral"),
        ("Throughput (tok/s)", "mean_tokens_per_sec", "higher_is_better"),
        ("Hit Max Limit (%)", "pct_hit_max_tokens", "lower_is_better"),
        ("Short Outputs (%)", "pct_short_outputs", "lower_is_better"),
        ("Empty Outputs (%)", "pct_empty_outputs", "lower_is_better")
    ]

    for label, key, direction in keys_to_compare:
        b_val = base_metrics[key]
        f_val = ft_metrics[key]
        abs_diff = f_val - b_val
        rel_change = (abs_diff / b_val * 100.0) if b_val != 0 else 0.0

        if direction == "higher_is_better":
            status = "Improved" if abs_diff > 0 else ("Degraded" if abs_diff < 0 else "Neutral")
        elif direction == "lower_is_better":
            status = "Improved" if abs_diff < 0 else ("Degraded" if abs_diff > 0 else "Neutral")
        else:
            status = "Informational"

        ci_str = "N/A"
        if bootstrap_ci and key in bootstrap_ci:
            ci_data = bootstrap_ci[key]
            ci_str = f"[{ci_data['ci_lower']:+.3f}, {ci_data['ci_upper']:+.3f}] (p={ci_data['p_value']:.3f})"

        rows.append({
            "Metric": label,
            "Base Model": round(b_val, 4),
            "Fine-Tuned": round(f_val, 4),
            "Absolute Delta": round(abs_diff, 4),
            "Relative Change (%)": f"{rel_change:+.2f}%",
            "95% Bootstrap CI": ci_str,
            "Directional Status": status
        })

    df = pd.DataFrame(rows)
    return df
