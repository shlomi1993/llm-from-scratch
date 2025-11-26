from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass

from .common import bytes_convert
from src.configurations import GptConfig


@dataclass
class AbstractResults(ABC):

    @abstractmethod
    def print(self) -> None:
        """
        Print estimation results.
        """
        pass


@dataclass
class MhaGqaResult(AbstractResults):
    """
    Results from MHA vs GQA estimation.
    """
    bytes_per_elem: int
    head_dim: int
    n_kv_heads_gqa: int
    total_mha: int
    total_gqa: int
    ratio: float
    savings: float

    def print(self, config: GptConfig, args: Namespace) -> None:
        print("==== Config ====")
        for k, v in vars(config).items():
            if v is not None:
                print(f"{k:23}: {v}")
        print(f"{'batch_size':23}: {args.batch_size}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'head_dim':23}: {self.head_dim}")
        print(f"{'GQA n_kv_heads':23}: {self.n_kv_heads_gqa}")
        print()
        print("==== KV-cache totals across all layers ====")
        print(f"MHA total KV cache  : {bytes_convert(self.total_mha)}")
        print(f"GQA total KV cache  : {bytes_convert(self.total_gqa)}")
        print(f"Ratio (MHA / GQA)   : {self.ratio:,.2f}x")
        print(f"Savings (GQA vs MHA): {self.savings * 100:,.2f}%")


@dataclass
class MlaResult(AbstractResults):
    """
    Results from MHA vs GQA vs MLA estimation.
    """
    bytes_per_elem: int
    head_dim: int
    n_kv_heads_gqa: int
    total_mha: int
    total_gqa: int
    ratio: float
    savings: float
    latent_dim: int
    total_mla: int
    ratio_mha_mla: float
    savings_mla: float

    def print(self, config: GptConfig, args: Namespace) -> None:
        """Print MHA vs GQA vs MLA estimation results."""
        print("==== Config ====")
        for k, v in vars(config).items():
            if v is not None:
                print(f"{k:23}: {v}")
        print(f"{'batch_size':23}: {args.batch_size}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'head_dim':23}: {self.head_dim}")
        print(f"{'GQA n_kv_heads':23}: {self.n_kv_heads_gqa}")
        print()
        print("==== KV-cache totals across all layers ====")
        print(f"MHA total KV cache  : {bytes_convert(self.total_mha)}")
        print(f"GQA total KV cache  : {bytes_convert(self.total_gqa)}")
        print(f"MLA total KV cache  : {bytes_convert(self.total_mla)}")
        print(f"Ratio (MHA / GQA)   : {self.ratio:,.2f}x")
        print(f"Savings (GQA vs MHA): {self.savings * 100:,.2f}%")
        print(f"Ratio (MHA / MLA)   : {self.ratio_mha_mla:,.2f}x")
        print(f"Savings (MLA vs MHA): {self.savings_mla * 100:,.2f}%")


@dataclass
class SwaResult(AbstractResults):
    """
    Results from Sliding Window Attention estimation.
    """
    bytes_per_elem: int
    head_dim: int
    n_kv_heads_gqa: int
    eff_W: int
    n_swa_layers: int
    n_full_layers: int
    total_mha_all_full: int
    total_gqa_all_full: int
    total_mixed_mha: int
    total_mixed_gqa: int

    def print(self, config: GptConfig, args: Namespace) -> None:
        """Print SWA estimation results."""
        print("==== Config ====")
        for k, v in vars(config).items():
            if v is not None:
                print(f"{k:23}: {v}")
        print(f"{'sliding_window_size':23}: {args.sliding_window_size}")
        print(f"{'batch_size':23}: {args.batch_size}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'head_dim':23}: {self.head_dim}")
        print(f"{'GQA n_kv_heads':23}: {self.n_kv_heads_gqa}")
        print(f"{'Effective SWA window W':23}: {self.eff_W}")
        print(f"{'Layer ratio (SWA:Full)':23}: {args.swa_ratio} -> {self.n_swa_layers} SWA, {self.n_full_layers} Full")
        print()
        print("==== KV-cache totals across all layers ====")
        print(f"MHA KV total           : {bytes_convert(self.total_mha_all_full)}")
        print(f"GQA KV total           : {bytes_convert(self.total_gqa_all_full)}")
        print(f"MHA + SWA (ratio {args.swa_ratio})  : {bytes_convert(self.total_mixed_mha)}")
        print(f"GQA + SWA (ratio {args.swa_ratio})  : {bytes_convert(self.total_mixed_gqa)}")


@dataclass
class MoeResult(AbstractResults):
    """
    Results from MoE FFN estimation.
    """
    dense_params: int
    router: int
    moe_hidden_dim: int
    per_expert_params: int
    moe_total: int
    moe_active_params_per_token: int
    bytes_per_elem: int

    def print(self, args: Namespace) -> None:
        """Print MoE estimation results."""
        print("==== Config ====")
        print(f"{'emb_dim':23}: {args.emb_dim}")
        print(f"{'hidden_dim':23}: {args.hidden_dim}")
        print(f"{'ffn_type':23}: {args.ffn_type}")
        print(f"{'num_experts':23}: {args.num_experts}")
        print(f"{'top_k':23}: {args.top_k}")
        print(f"{'dtype':23}: {args.dtype} ({self.bytes_per_elem} Bytes/elem)")
        print(f"{'match_dense':23}: {args.match_dense}")
        print()
        print("==== Model weights (parameters) ====")
        print(f"{'Dense FFN params':23}: {self.dense_params:,} ({bytes_convert(self.dense_params * self.bytes_per_elem)})")
        print(f"{'Per-expert params':23}: {self.per_expert_params:,} ({bytes_convert(self.per_expert_params * self.bytes_per_elem)})")
        print(f"{'Router params':23}: {self.router:,} ({bytes_convert(self.router * self.bytes_per_elem)})")
        print(f"{'MoE TOTAL params':23}: {self.moe_total:,} ({bytes_convert(self.moe_total * self.bytes_per_elem)})")
        print(f"{'MoE ACTIVE/Token':23}: {self.moe_active_params_per_token:,} ({bytes_convert(self.moe_active_params_per_token * self.bytes_per_elem)})")
        print(f"{'moe_hidden_dim':23}: {self.moe_hidden_dim}")
