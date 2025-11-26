from enum import Enum


class Mode(Enum):
    MHA = "mha"
    GQA = "gqa"
    MLA = "mla"
    SWA = "swa"
    MOE = "moe"


MODE_FLAG_REQUIREMENTS = {
    Mode.MHA: ["--emb-dim", "--n-heads", "--n-layers"],
    Mode.GQA: ["--emb-dim", "--n-heads", "--n-layers", "--n-kv-groups"],
    Mode.MLA: ["--emb-dim", "--n-heads", "--n-layers", "--n-kv-groups", "--latent-dim"],
    Mode.SWA: ["--emb-dim", "--n-heads", "--n-layers", "--n-kv-groups", "--sliding-window-size"],
    Mode.MOE: ["--emb-dim", "--hidden-dim", "--ffn-type", "--num-experts", "--top-k"],
}
