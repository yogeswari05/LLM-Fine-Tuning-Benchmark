# Guide: LLM Fine-Tuning, Parameter-Efficient Adaptation & Evaluation

This document provides explanations. It focuses on the engineering decisions, mathematical foundations, and system trade-offs implemented in this project.

---

## 1. Parameter-Efficient Fine-Tuning: LoRA and QLoRA

### Q1: What is the mathematical formulation of LoRA, and why are its matrices initialized asymmetrically?
**Answer:**  
In standard fine-tuning, a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ is updated by adding a full-rank delta matrix $\Delta W \in \mathbb{R}^{d \times k}$. LoRA hypothesizes that weight updates have a low "intrinsic dimension" and decomposes $\Delta W$ into the product of two low-rank matrices:
$$\Delta W = \frac{\alpha}{r} B A$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d, k)$, and $\alpha$ is a constant scaling hyperparameter. The forward pass becomes:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$

**Initialization Rationale:**
- Matrix $A$ is initialized using a Gaussian distribution $\mathcal{N}(0, \sigma^2)$ (or Kaiming uniform).
- Matrix $B$ is initialized to exact zeros ($B = 0$).
- **Why?** Because $B = 0$, the product $B A = 0$ at step 0. This ensures that $\Delta W = 0$ at the start of training, meaning the initial model output is mathematically identical to the pre-trained base model $W_0$. If both $A$ and $B$ were initialized randomly, the initial model would undergo random perturbation, destroying pre-trained knowledge before the first gradient step.

### Q2: What is the role of the LoRA scaling factor $\alpha$?
**Answer:**  
$\alpha$ acts as a constant learning rate multiplier. When updating rank $r$, the update is scaled by $\frac{\alpha}{r}$. Setting $\alpha = 2r$ (e.g., $r=16, \alpha=32$) keeps the magnitude of the adapter activations proportional when experimenting with different values of $r$, reducing the need to retune the optimizer's base learning rate whenever $r$ changes.

### Q3: How does QLoRA work, and what are its three key technical contributions?
**Answer:**  
QLoRA (Dettmers et al., 2023) enables fine-tuning 4-bit quantized base models without sacrificing performance. Its three core innovations are:
1. **NF4 (NormalFloat4) Quantization**: Standard INT4 quantization uniformly spaces quantization bins. However, pre-trained neural network weights follow a zero-centered normal distribution. NF4 constructs information-theoretically optimal quantile bins such that each bin has an equal number of expected weight parameters, minimizing quantization error.
2. **Double Quantization (DQ)**: Quantization requires storing FP32 quantization constants (scale factors) for blocks of weights (e.g., block size 64). Double Quantization quantizes these FP32 constants themselves into 8-bit FP with block size 256, saving approximately $0.37$ bits per parameter (roughly 3 GB for a 65B model).
3. **Paged Optimizers**: Leverages CUDA Unified Memory to automatically page optimizer states (e.g., AdamW 8-bit states) between GPU VRAM and CPU system RAM when memory spikes occur, preventing Out-Of-Memory (OOM) crashes during gradient accumulation spikes.

---

## 2. Supervised Fine-Tuning & Loss Masking Mechanics

### Q4: Why must we mask prompt tokens with `-100` during causal language model training?
**Answer:**  
In causal language modeling, every token is autoregressively conditioned on prior tokens. If prompt tokens are not masked:
1. **Capacity Waste**: The loss function computes cross-entropy over the tokens in the system prompt, instructions, and paper body. The model spends gradient budget learning to predict the phrasing of the user prompt rather than learning to summarize.
2. **Distribution Shift**: The model may learn to generate prompt headers (e.g., `<|im_start|>user`) rather than learning when to stop and produce summary content.
3. **Loss Distortions**: Long articles with thousands of tokens dominate the loss calculation, meaning a $90\%$ reduction in loss could merely mean the model learned the grammar of the input article rather than producing better summaries.

Setting `labels[i] = -100` instructs PyTorch's `CrossEntropyLoss(ignore_index=-100)` to completely exclude prompt positions from loss summation and gradient computation.

### Q5: Explain the exact index shift between `logits` and `labels` in Causal LM training.
**Answer:**  
In causal language models, token at index $t$ predicts token at index $t+1$.  
Given an input sequence $x = [x_0, x_1, \dots, x_{T-1}]$:
- Logits have shape `[batch_size, seq_len, vocab_size]`. Logit $i$ corresponds to the distribution predicted after observing input $x_i$.
- To evaluate whether logit $i$ correctly predicted the subsequent token $x_{i+1}$, the loss compares:
  $$\text{shift\_logits} = \text{logits}[:, :-1, :] \quad (\text{tokens } 0 \dots T-2)$$
  $$\text{shift\_labels} = \text{labels}[:, 1:] \quad (\text{tokens } 1 \dots T-1)$$
If prompt tokens span indices $0 \dots P-1$ and summary tokens span $P \dots T-1$:
- Setting `labels[0:P] = -100` ensures that `shift_labels[0:P-1] = -100`.
- The first non-masked target is `shift_labels[P-1]`, which is `labels[P]` (the first token of the summary).
- Its prediction is conditioned on `logits[P-1]`, which ingested `input_ids[0:P-1]` (the entire prompt).
- The final summary token $T-2$ predicts $T-1$ (`<|im_end|>`), teaching the model when to cease generation.

---

## 3. Data Engineering & Smart Truncation

### Q6: Why is naive truncation (`tokenizer(text, max_length=1024, truncation=True)`) a critical bug in summarization tasks?
**Answer:**  
In scientific summarization, the input article is typically 5,000–10,000 words, while the summary is 150–250 words.  
If you concatenate `[Prompt + Article + Summary]` and apply naive right-truncation:
- All tokens after index 1024 are discarded.
- Because the article alone exceeds 1024 tokens, the target summary is 100% truncated! The model trains on zero target tokens, resulting in NaN loss or total failure to learn.

If you concatenate and apply left-truncation:
- The system prompt, instruction, and beginning of the article are discarded, corrupting the chat template delimiters.

**The Solution (Smart Budget Truncation):**
1. Allocate an explicit budget for the target summary ($T_{\text{budget}} = 256$).
2. Calculate the structural token overhead of the prompt template $W_{\text{overhead}}$.
3. Compute the maximum allowable article tokens: $A_{\text{budget}} = L_{\text{max}} - T_{\text{actual}} - W_{\text{overhead}}$.
4. Truncate only the article body to $A_{\text{budget}}$ while leaving the instructions and target summary 100% intact.

---

## 4. Evaluation, Benchmarking & Metrics

### Q7: Why is generation latency benchmarking invalid without `torch.cuda.synchronize()`?
**Answer:**  
GPU operations in PyTorch are executed asynchronously. When `model.generate(...)` is called, Python immediately returns control to the CPU after queuing the CUDA kernels into the CUDA stream. If you measure elapsed time using standard Python timing (`time.perf_counter()`):
```python
t0 = time.perf_counter()
model.generate(...)
t1 = time.perf_counter()  # Measures CPU launch overhead, NOT GPU kernel execution!
```
Without `torch.cuda.synchronize()`, you are measuring the fraction of a millisecond it took CPU to dispatch the command, not the actual multi-second token generation time. Calling `torch.cuda.synchronize()` forces the CPU thread to block until all queued GPU tasks complete.

### Q8: Why can comparing raw tokens/second throughput be misleading between base and fine-tuned models?
**Answer:**  
In autoregressive decoding:
1. **Prefill Phase (Time-To-First-Token, TTFT)**: Processes all prompt tokens in parallel (compute-bound matrix multiplication).
2. **Decode Phase**: Generates tokens sequentially one-by-one with KV-cache (memory bandwidth-bound).

If Model A produces a short summary (50 tokens) and Model B produces a detailed summary (200 tokens):
- Model A has a higher proportion of its total latency spent in the fast prefill phase, artificially skewing tokens/second.
- Conversely, longer generations suffer higher KV-cache memory transfer overhead per token.
- Furthermore, a model that repeats degenerate loops generates tokens rapidly but provides zero semantic value. Throughput must always be evaluated alongside output length distributions and quality metrics.

### Q9: What are the differences between ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum?
**Answer:**  
- **ROUGE-1**: Overlap of unigrams (single words) between generated and reference text. Measures basic vocabulary recall.
- **ROUGE-2**: Overlap of bigrams (two-word sequences). Measures local phrasing fluency and domain terminology.
- **ROUGE-L**: Longest Common Subsequence (LCS). Identifies the longest sequence of words appearing in both texts in the same order, regardless of intervening words. Captures sentence-level structural similarity.
- **ROUGE-Lsum**: Computes LCS by splitting text into sentences (using newline delimiters) and taking the union of LCS across sentences. This is specifically tailored for multi-sentence summarization tasks (like scientific abstracts).

### Q10: Why are Paired Bootstrap Confidence Intervals preferred over Student's t-tests for LLM evaluation?
**Answer:**  
1. **Non-Normality**: ROUGE scores across test sets are bounded in $[0, 1]$ and heavily skewed (often bimodal or with long tails), violating the normality assumption required by Student's t-test.
2. **Paired Dependency**: Both base and fine-tuned models are evaluated on the exact same test prompts. Paired bootstrap resampling preserves the correlation structure of sample pairs $(y_i^{\text{base}}, y_i^{\text{ft}})$, computing the empirical distribution of the difference $\Delta_i$ without parametric assumptions.
3. **Robust Confidence Limits**: By resampling with replacement $B=1,000$ times, the percentile method provides robust $95\%$ confidence bounds $[\Delta_{0.025}, \Delta_{0.975}]$ and direct two-tailed empirical p-values.

---

## 5. System Design & Production Deployment

### Q11: How do you deploy fine-tuned LoRA models in high-throughput production?
**Answer:**  
1. **Weight Merging**: If serving a single specialized model, merge the LoRA weights permanently into the base model:
   $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$$
   via `model.merge_and_unload()`. This eliminates all adapter lookup latency during inference.
2. **Multi-Tenant Dynamic LoRA Serving (e.g., vLLM / S-LoRA)**: If serving hundreds of customized adapters across different customers or tasks:
   - Keep a single shared base model $W_0$ in GPU VRAM.
   - Dynamically load the tiny LoRA matrices ($A_k, B_k$, ~30 MB each) into a GPU adapter cache.
   - Batched inference processes multiple requests simultaneously, routing queries to their respective adapter weights using segmented matrix multiplication kernels without duplicating base model weights.
