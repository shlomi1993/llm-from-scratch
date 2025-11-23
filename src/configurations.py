from dataclasses import dataclass


# NOTE: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
# NOTE: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=dOad5HoAAAAJ&citation_for_view=dOad5HoAAAAJ:YsMSGLbcyi4C

@dataclass
class GptConfig:
    emb_dim: int                # Embedding dimension
    n_layers: int               # Number of layers
    n_heads: int                # Number of attention heads
    vocab_size: int = 50257     # Vocabulary size
    context_length: int = 1024  # Context length
    drop_rate: float = 0.1      # Dropout rate
    qkv_bias: bool = False      # Query-Key-Value bias


# GPT2-Small
GPT_CONFIG_124M = GptConfig(
    emb_dim=768,
    n_layers=12,
    n_heads=12
)


# GPT2-Medium
GPT_CONFIG_355M = GptConfig(
    emb_dim=1024,
    n_layers=24,
    n_heads=16
)


# GPT2-Large
GPT_CONFIG_774M = GptConfig(
    emb_dim=1280,
    n_layers=36,
    n_heads=20
)


# GPT2-XL
GPT_CONFIG_1558M = GptConfig(
    emb_dim=1600,
    n_layers=48,
    n_heads=25
)
