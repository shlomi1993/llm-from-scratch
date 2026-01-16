import tiktoken
import torch


# Special tokens
EOT = "<|endoftext|>"
PAD_IDX = 50256  # <|endoftext|>
IGNORE_IDX = -100  # Used to ignore padding tokens in cross-entropy loss computation


class Gpt2Tokenizer:
    """
    A wrapper class for the tiktoken tokenizer, providing convenience methods for encoding and decoding.
    """

    def __init__(self) -> None:
        """
        Initialize the GPT-2 tokenizer using tiktoken.
        """
        self._tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, **kwargs) -> list[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text (str): The input text to encode.
            **kwargs: Additional keyword arguments to pass to the tokenizer's encode method.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self._tokenizer.encode(text, **kwargs)

    def decode(self, tokens: list[int], **kwargs) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            tokens (list[int]): A list of token IDs to decode.
            **kwargs: Additional keyword arguments to pass to the tokenizer's decode method.

        Returns:
            str: The decoded text string.
        """
        return self._tokenizer.decode(tokens, **kwargs)

    def text_to_token_ids(self, text: str) -> torch.Tensor:
        """
        Convert a text string into a tensor of token IDs using the GPT-2 tokenizer.

        Args:
            text (str): The input text to encode.

        Returns:
            torch.Tensor: A tensor of token IDs representing the encoded text.
        """
        encoded = self._tokenizer.encode(text)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor

    def token_ids_to_text(self, token_ids: torch.Tensor) -> str:
        """
        Convert a tensor of token IDs back into a text string using the GPT-2 tokenizer.

        Args:
            token_ids (torch.Tensor): A tensor of token IDs to decode.

        Returns:
            str: The decoded text string.
        """
        flat = token_ids.squeeze(0)  # remove batch dimension
        return self._tokenizer.decode(flat.tolist())


# Initialize a global GPT-2 tokenizer instance
g_tokenizer = Gpt2Tokenizer()
