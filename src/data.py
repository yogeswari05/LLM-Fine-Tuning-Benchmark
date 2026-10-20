"""
Dataset loading, splitting, smart sequence truncation, and chat template formatting.
"""

import os
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
from .loss_masking import build_masked_inputs
from .utils import save_json


class ArxivDatasetManager:
    """
    Manages loading, non-overlapping split creation, and serialization
    of the arXiv summarization benchmark dataset.
    """
    def __init__(
        self,
        dataset_name: str = "ccdv/arxiv-summarization",
        article_column: str = "article",
        summary_column: str = "abstract",
        train_samples: int = 240,
        val_samples: int = 36,
        test_samples: int = 24,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.article_column = article_column
        self.summary_column = summary_column
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.seed = seed

    def load_and_split(
        self,
        splits_save_path: Optional[str] = "./artifacts/dataset_splits.json"
    ) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:
        """
        Loads the dataset and creates disjoint train, validation, and test splits.
        Guarantees zero data leakage or index overlap.
        """
        print(f"Loading dataset '{self.dataset_name}'...")
        try:
            # We load the dataset with streaming=False for exact indexing
            raw_dataset = load_dataset(self.dataset_name, split="train")
        except Exception as e:
            print(f"Standard loading failed ({e}). Attempting streaming fallback...")
            raw_dataset = load_dataset(self.dataset_name, split="train", streaming=True)
            # Take needed samples
            total_needed = self.train_samples + self.val_samples + self.test_samples + 50
            collected = list(raw_dataset.take(total_needed))
            raw_dataset = Dataset.from_list(collected)

        total_available = len(raw_dataset)
        total_requested = self.train_samples + self.val_samples + self.test_samples
        if total_available < total_requested:
            raise ValueError(
                f"Requested {total_requested} examples, but only {total_available} are available."
            )

        # Generate deterministic, non-overlapping random indices
        rng = np.random.default_rng(self.seed)
        all_indices = rng.permutation(total_available)

        train_indices = all_indices[:self.train_samples].tolist()
        val_indices = all_indices[self.train_samples : self.train_samples + self.val_samples].tolist()
        test_indices = all_indices[
            self.train_samples + self.val_samples : self.train_samples + self.val_samples + self.test_samples
        ].tolist()

        # Strict assertion of disjointness
        assert len(set(train_indices).intersection(set(val_indices))) == 0, "Train and Val overlap detected!"
        assert len(set(train_indices).intersection(set(test_indices))) == 0, "Train and Test overlap detected!"
        assert len(set(val_indices).intersection(set(test_indices))) == 0, "Val and Test overlap detected!"

        train_ds = raw_dataset.select(train_indices)
        val_ds = raw_dataset.select(val_indices)
        test_ds = raw_dataset.select(test_indices)

        # Build split metadata
        split_metadata = {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "num_test": len(test_indices)
        }

        if splits_save_path:
            save_json(split_metadata, splits_save_path)
            print(f"Saved split indices and verification to '{splits_save_path}'")

        return train_ds, val_ds, test_ds, split_metadata


def smart_truncate_and_format(
    article: str,
    summary: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 1024,
    max_target_length: int = 256,
    system_prompt: str = "You are an expert scientific researcher. Summarize the following research paper concisely.",
    user_prompt_template: str = "Please summarize the following scientific paper:\n\n{article}",
    eos_token: str = "<|im_end|>"
) -> Dict[str, Any]:
    """
    Performs budget-preserving sequence formatting:
    1. Preserves the target budget (up to max_target_length).
    2. Measures template and system prompt overhead.
    3. Allocates remaining token budget to the article, truncating only the article body.
    4. Applies model chat template.
    5. Masks prompt tokens with -100 for supervised causal LM loss.
    """
    # 1. Inspect raw target length
    raw_summary_ids = tokenizer.encode(summary.strip(), add_special_tokens=False)
    raw_target_len = len(raw_summary_ids)

    # Allow headroom for EOS/formatting token
    target_budget_limit = max_target_length - 4
    if raw_target_len > target_budget_limit:
        target_truncated = True
        truncated_summary_ids = raw_summary_ids[:target_budget_limit]
        final_summary = tokenizer.decode(truncated_summary_ids, skip_special_tokens=True).strip()
    else:
        target_truncated = False
        final_summary = summary.strip()

    # Format target text with explicit closing token
    target_text = f"{final_summary}{eos_token}\n"
    actual_target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    actual_target_len = len(actual_target_ids)

    # 2. Compute prompt token budget
    allowed_prompt_budget = max(64, max_seq_length - actual_target_len)

    # Compute template structural overhead using empty article
    overhead_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format(article="")}
    ]
    empty_prompt_text = tokenizer.apply_chat_template(
        overhead_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    overhead_ids = tokenizer.encode(empty_prompt_text, add_special_tokens=False)
    overhead_len = len(overhead_ids) + 4  # safety margin for subword boundary variations

    allowed_article_tokens = max(32, allowed_prompt_budget - overhead_len)

    # 3. Truncate article tokens strictly within budget
    raw_article_ids = tokenizer.encode(article.strip(), add_special_tokens=False)
    raw_article_len = len(raw_article_ids)

    if raw_article_len > allowed_article_tokens:
        article_truncated = True
        truncated_article_ids = raw_article_ids[:allowed_article_tokens]
        final_article = tokenizer.decode(truncated_article_ids, skip_special_tokens=True).strip()
    else:
        article_truncated = False
        final_article = article.strip()

    # 4. Construct final prompt with chat template
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_template.format(article=final_article)}
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 5. Build masked inputs and labels
    masked_data = build_masked_inputs(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        target_text=target_text,
        max_seq_length=max_seq_length
    )

    # 6. Aggregate metadata and truncation statistics
    masked_data.update({
        "raw_article_token_len": raw_article_len,
        "raw_target_token_len": raw_target_len,
        "final_article_token_len": len(tokenizer.encode(final_article, add_special_tokens=False)),
        "final_target_token_len": actual_target_len,
        "article_truncated": article_truncated,
        "target_truncated": target_truncated,
        "prompt_text": prompt_text,
        "target_text": target_text
    })

    return masked_data


def process_dataset_split(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 1024,
    max_target_length: int = 256,
    article_column: str = "article",
    summary_column: str = "abstract",
    system_prompt: str = "You are an expert scientific researcher. Summarize the following research paper concisely.",
    user_prompt_template: str = "Please summarize the following scientific paper:\n\n{article}"
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Transforms a HuggingFace Dataset by applying smart truncation and formatting,
    and returns token length and truncation statistics.
    """
    processed_records = []
    stats = {
        "total_examples": len(dataset),
        "article_truncated_count": 0,
        "target_truncated_count": 0,
        "article_lengths": [],
        "target_lengths": [],
        "total_lengths": []
    }

    for idx, row in enumerate(dataset):
        article = row[article_column]
        summary = row[summary_column]

        item = smart_truncate_and_format(
            article=article,
            summary=summary,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_target_length=max_target_length,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template
        )
        item["example_id"] = str(idx)
        processed_records.append(item)

        stats["article_lengths"].append(item["raw_article_token_len"])
        stats["target_lengths"].append(item["raw_target_token_len"])
        stats["total_lengths"].append(item["total_length"])
        if item["article_truncated"]:
            stats["article_truncated_count"] += 1
        if item["target_truncated"]:
            stats["target_truncated_count"] += 1

    stats["article_truncation_rate"] = stats["article_truncated_count"] / stats["total_examples"]
    stats["target_truncation_rate"] = stats["target_truncated_count"] / stats["total_examples"]
    stats["mean_article_tokens"] = float(np.mean(stats["article_lengths"]))
    stats["mean_target_tokens"] = float(np.mean(stats["target_lengths"]))
    stats["mean_total_tokens"] = float(np.mean(stats["total_lengths"]))

    processed_dataset = Dataset.from_list(processed_records)
    return processed_dataset, stats
