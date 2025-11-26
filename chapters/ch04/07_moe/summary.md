# Mixture of Experts (MoE) Summary

## Overview

This directory implements Mixture of Experts (MoE), a sparse model architecture that dramatically increases model capacity while maintaining computational efficiency during inference. MoE replaces the traditional dense feed-forward networks in transformer blocks with multiple specialized "expert" networks, where only a small subset of experts are activated for each token. This approach enables models to have billions of parameters while using only a fraction during inference.

The implementation demonstrates how MoE architectures achieve the best of both worlds: massive parameter counts for increased model capacity during training, combined with sparse activation patterns that keep inference costs manageable. For example, DeepSeek-V3 contains 671 billion total parameters but activates only 37 billion per token (roughly 5.5% of the total), maintaining inference efficiency while providing enormous modeling capacity.

The directory contains implementations showing the transition from dense FFN layers to MoE modules (`gpt_moe.py`), along with routing mechanisms that intelligently select which experts should process each token. The code demonstrates both standard MoE with router-selected experts and advanced variants that include shared experts - specialists that are always active and handle common patterns across all tokens.

The implementation includes detailed memory estimation tools (`memory_estimator_moe.py`) that quantify the memory savings per token compared to equivalent dense models. The routing logic is carefully implemented to handle expert selection, load balancing, and the computational mechanics of sparse expert activation.

A particularly important aspect covered is the shared expert mechanism introduced in DeepSeek MoE, where one expert is always active to handle common patterns, allowing the routed experts to specialize in more specific tasks without redundantly learning universal features.

## Lesson Learned

Model capacity and computational efficiency are not mutually exclusive when proper sparsity patterns are applied - strategic expert routing allows models to access vast parameter spaces while maintaining practical inference costs. The key insight is that most tokens require only specialized knowledge from a small subset of the model's total capacity, making sparse activation both feasible and highly effective for scaling language models beyond traditional dense architecture limits.