"""
Loss masking, label alignment, and data collation for Supervised Fine-Tuning.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import torch
import torch.nn.functional as F


def build_masked_inputs(
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    max_seq_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Constructs input_ids, attention_mask, and labels where prompt tokens
    are masked with -100 so loss is computed exclusively on target tokens (and EOS).
    """
    # Tokenize prompt and target without adding default BOS/EOS unless specified
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    
    # Combined sequence
    input_ids = prompt_ids + target_ids
    prompt_length = len(prompt_ids)
    
    # If max_seq_length is specified, truncate if needed
    if max_seq_length is not None and len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    
    total_len = len(input_ids)
    attention_mask = [1] * total_len
    
    # Construct labels: -100 for all prompt tokens, exact token IDs for target tokens
    labels = [-100] * min(prompt_length, total_len)
    if total_len > prompt_length:
        labels += input_ids[prompt_length:]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": prompt_length,
        "target_length": max(0, total_len - prompt_length),
        "total_length": total_len
    }


@dataclass
class CausalLMDataCollator:
    """
    Data collator that dynamically pads input_ids, attention_mask, and labels.
    Labels are padded with -100 (ignored in CrossEntropyLoss).
    """
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = [f["input_ids"] for f in features]
        batch_attention_mask = [f["attention_mask"] for f in features]
        batch_labels = [f["labels"] for f in features]
        
        # Determine maximum sequence length in the batch
        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of is not None and (max_len % self.pad_to_multiple_of != 0):
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
            
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbls in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }


def verify_loss_shifting(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> Dict[str, Any]:
    """
    Verifies that causal language model loss correctly implements standard label shifting:
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    and confirms that masked tokens (-100) and padding contribute zero loss.
    """
    # 1. Standard HuggingFace Causal LM loss calculation
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    vocab_size = logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
    hf_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    
    # 2. Manual step-by-step unshifted loss check
    # We find all positions where shift_labels != ignore_index
    valid_mask = (shift_labels != ignore_index)
    num_valid_tokens = valid_mask.sum().item()
    
    if num_valid_tokens == 0:
        manual_loss = torch.tensor(0.0, device=logits.device)
    else:
        # Compute log softmax on shift_logits
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather probabilities of true tokens
        valid_indices = torch.where(valid_mask)
        target_token_ids = shift_labels[valid_indices]
        chosen_log_probs = log_probs[valid_indices[0], valid_indices[1], target_token_ids]
        manual_loss = -chosen_log_probs.mean()
        
    abs_diff = torch.abs(hf_loss - manual_loss).item()
    
    return {
        "hf_loss": hf_loss.item(),
        "manual_loss": manual_loss.item(),
        "abs_diff": abs_diff,
        "is_numerically_identical": abs_diff < 1e-5,
        "num_supervised_tokens": num_valid_tokens
    }
