from src.model.gpt import GptModel
from src.utils.tokenization import tokenizer


def test_tokenizer_sanity(pretrained_model: GptModel) -> None:
    token_ids = pretrained_model.generate(
        idx=tokenizer.text_to_token_ids("Every effort moves you"),
        max_new_tokens=10,
        context_size=pretrained_model.pos_emb.num_embeddings
    )
    generated_text = tokenizer.token_ids_to_text(token_ids)
    expected_text = "Every effort moves you rentingetic wasnم refres RexMeCHicular stren"
    assert generated_text ==  expected_text, f"{generated_text} != {expected_text}"
