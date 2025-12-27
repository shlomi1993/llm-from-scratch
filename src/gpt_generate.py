import numpy as np
import torch

from src.config import GptConfig
from src.gpt import GptModel
from src.transformer import TransformerBlock


def _assign(left: torch.nn.Parameter, right: np.ndarray) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(config: GptConfig, params: dict[str, np.ndarray]) -> GptModel:
    gpt = GptModel(config)
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, params["wte"])

    # Load transformer block weights
    for b in range(len(params["blocks"])):

        # Attention weights
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        block: TransformerBlock = gpt.trf_blocks[b]
        block.att.W_query.weight = _assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = _assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = _assign(block.att.W_value.weight, v_w.T)

        # Attention biases
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        block.att.W_query.bias = _assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = _assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = _assign(block.att.W_value.bias, v_b)

        # Output projection
        block.att.out_proj.weight = _assign(block.att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        block.att.out_proj.bias = _assign(block.att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        # Feed-forward weights and biases
        block.ff.layers[0].weight = _assign(block.ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        block.ff.layers[0].bias = _assign(block.ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        block.ff.layers[2].weight = _assign(block.ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        block.ff.layers[2].bias = _assign(block.ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

        # Layer norm parameters
        block.norm1.scale = _assign(block.norm1.scale, params["blocks"][b]["ln_1"]["g"])
        block.norm1.shift = _assign(block.norm1.shift, params["blocks"][b]["ln_1"]["b"])
        block.norm2.scale = _assign(block.norm2.scale, params["blocks"][b]["ln_2"]["g"])
        block.norm2.shift = _assign(block.norm2.shift, params["blocks"][b]["ln_2"]["b"])

    # Final layer norm and output head
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])
