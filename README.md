# Coding LLM

An independent coding-language-model research project.

## Goal

Build a coding-native language model from our own training pipeline rather than wrapping a hosted LLM API.

## Roadmap

- [x] Repository foundation
- [x] Configurable decoder-only Transformer
- [x] Code-aware tokenizer training pipeline
- [x] Streaming text dataset pipeline
- [x] Pretraining loop with checkpointing
- [x] Local inference CLI
- [x] Evaluation harness
- [ ] Instruction tuning
- [ ] Repository-level coding tasks
- [ ] Execution-based reinforcement learning
- [ ] Larger-scale training

## First target

The first milestone is a small, fully trainable model that can be initialized from scratch and trained locally.

See `docs/ARCHITECTURE.md` for the design.
