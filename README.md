# GPT-Based Character-Level Language Model

This project implements a **character-level GPT (Generative Pre-trained Transformer) model** from scratch using **PyTorch**. The model is trained on a custom text corpus and can generate coherent sequences of text. It follows a **transformer-based architecture**, incorporating **multi-head self-attention, feedforward networks, and layer normalization**.

## Features
- **Token & Positional Embeddings:** Maps characters to numerical representations while encoding sequence order.
- **Multi-Head Self-Attention:** Implements both **custom** and **optimized causal self-attention** for efficient computation.
- **Transformer Blocks:** Stacks multiple self-attention layers followed by feedforward layers.
- **Training Pipeline:** Implements **AdamW optimizer, loss estimation, and batch sampling**.
- **Text Generation:** Generates text autoregressively based on learned patterns.

## What I Learned
- **Transformer Architecture:** Deep understanding of **self-attention, masked attention, feedforward layers, Layer Normalization, Residual Connections**.
- **PyTorch Implementation:** Built a GPT-style model from scratch, including **custom attention mechanisms**.
- **Efficient Training Techniques:** Implemented **AdamW optimization** with **learning rate scheduling**.
- **Text Generation:** Generated text sequences using a **character-level transformer**.
