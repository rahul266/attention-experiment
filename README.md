# Attention Experiment — Word Level Language Model

A from-scratch implementation of a self-attention based language model trained on the TinyStories dataset. Built as a learning experiment to understand attention and organically understand transformer attention mechanism.
The goal was to get to a point where the limitations of simple attention become obvious — and from there, understand *why* transformers use separate Q, K, V projections.

## Dataset

- **TinyStories** (`TinyStories-train.txt`)
- 50,000 stories loaded and tokenized at the word level
- Tokenization: words, punctuation, and special tokens (`<|startoftext|>`, `<|endoftext|>`)

## Model Architecture

```
Input tokens
    → Embedding  (vocab_size → 32)
    → Self-Attention  (E @ E.T @ Wa → causal mask → softmax → @ E)
    → Linear (32 → 256) + LeakyReLU
    → Linear (256 → 256) + LeakyReLU
    → Linear (256 → vocab_size)
    → Logits
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding size | 32 |
| Hidden size | 256 |
| Context length | 256 tokens |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 0.01 |
| Loss | CrossEntropyLoss (ignore_index=-1) |

### Attention mechanism

The attention used here is a simplified self-attention — no separate Q, K, V projections:

```python
attn = E @ E.transpose(-1, -2)   # how each word relates to every other word
attn = attn @ self.Wa             # learnable attention weight matrix
attn = causal_mask(attn)          # mask future tokens (lower triangle only)
attn = softmax(attn)              # probabilities
x    = attn @ E                   # weighted sum of embeddings
```

`Wa` is a learnable `(context_len, context_len)` parameter. Padding tokens are masked out before softmax.

## Results

- Best training loss: **~2.1**
- The model picks up basic structure — punctuation, subject-object patterns, names — but has no long-range consistency.

## Requirements

```
torch
matplotlib
```

Runs on CPU or Apple MPS (`mps` is used automatically if available).

## Files

| File | Description |
|---|---|
| `attention-experiment.ipynb` | Main notebook — data loading, model, training loop, generation |
| `TinyStories-train.txt` | Training data (not included) |
| `model_weights.pth` | Saved model weights (after training) |

## What's next

The model works but the attention design (using the same embeddings for query and key) is the bottleneck. The next step is understanding why transformers project embeddings into separate Q, K, V spaces — and what that actually buys you.