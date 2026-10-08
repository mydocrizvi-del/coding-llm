# Architecture

## Model

The first implementation is a decoder-only causal Transformer initialized from scratch. It uses:

- RMSNorm
- RoPE positional embeddings
- causal scaled dot-product attention
- gated SiLU MLP
- tied input/output embeddings

The model is deliberately small enough to experiment with locally.

## Training

The initial pipeline is next-token prediction over tokenized text. The next milestones add:

1. curated code pretraining data
2. instruction/chat tuning
3. execution-verified coding tasks
4. preference/reward optimization
5. repository-level agent training

## Data principles

Prefer data for which licensing/provenance is known. Keep source metadata, language labels, repository identifiers, and license information alongside training shards.

## Scaling

We can scale depth, width, context and eventually move to MoE/distributed training without changing the public training interface.
