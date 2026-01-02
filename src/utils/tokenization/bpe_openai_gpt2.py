# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License
#

# Software Copyright (c) 2019 OpenAI

# We don't claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# The above copyright notice and this permission notice need not be included
# with content created by the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

# MODIFICATIONS MADE (2025):
# - Added comprehensive type hints for all functions and methods
# - Enhanced docstrings with detailed parameter descriptions and examples
# - Improved variable names for better code readability (e.g., get_pairs function)
# - Added detailed comments throughout the BPE algorithm implementation
# - Enhanced error handling and progress reporting in download_vocab function
# - Maintained full backward compatibility with original OpenAI implementation

import json
import os
import regex as re
import requests

from functools import lru_cache
from logging import getLogger
from tqdm import tqdm
from typing import Dict, List, Set, Tuple


_logger = getLogger(__name__)


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    Creates a mapping from UTF-8 byte values to unicode characters for BPE encoding.

    The reversible BPE codes work on unicode strings. To avoid requiring a large number of unicode characters in the
    vocabulary (which would be needed for decent coverage of a 10B+ token dataset), this function creates lookup tables
    between UTF-8 bytes and unicode strings. It avoids mapping to whitespace/control characters that could cause issues
    during BPE processing.

    Returns:
        Dict[int, str]: A dictionary mapping byte values (0-255) to corresponding unicode character representations.
    """
    # Start with printable ASCII characters (! to ~)
    byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values[:]

    # Add remaining byte values (0-255) that aren't already covered
    next_unicode_value = 0
    for byte_val in range(2 ** 8):
        if byte_val not in byte_values:
            byte_values.append(byte_val)
            unicode_values.append(2 ** 8 + next_unicode_value)
            next_unicode_value += 1

    # Convert to actual unicode characters
    unicode_chars = [chr(val) for val in unicode_values]
    return dict(zip(byte_values, unicode_chars))


def get_pairs(symbol_sequence: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """
    Extracts all consecutive symbol pairs from a sequence of symbols.

    This function is used in the BPE algorithm to identify which symbol pairs are present in a word representation. Each
    symbol can be a variable-length string, and we create pairs from consecutive symbols.

    Args:
        symbol_sequence (Tuple[str, ...]): A tuple of symbols representing a word, where each symbol is a string.

    Returns:
        Set[Tuple[str, str]]: A set of tuples, where each tuple contains two consecutive symbols from the input sequence.

    Example:
        >>> get_pairs(("h", "e", "l", "l", "o"))
        {("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")}
    """
    symbol_pairs = set()
    previous_symbol = symbol_sequence[0]

    for current_symbol in symbol_sequence[1:]:
        symbol_pairs.add((previous_symbol, current_symbol))
        previous_symbol = current_symbol

    return symbol_pairs


class Encoder:
    """
    Byte Pair Encoding (BPE) encoder/decoder for GPT-2 style tokenization.

    This class implements the BPE algorithm used by GPT-2 for converting text to tokens and vice versa. It handles the
    encoding of text to byte-level representations, applies BPE merges, and converts to/from token IDs.

    Attributes:
        encoder (Dict[str, int]): Mapping from BPE tokens to token IDs.
        decoder (Dict[int, str]): Mapping from token IDs to BPE tokens.
        errors (str): How to handle decoding errors ('replace', 'ignore', etc.).
        byte_encoder (Dict[int, str]): Mapping from bytes to unicode chars.
        byte_decoder (Dict[str, int]): Mapping from unicode chars to bytes.
        bpe_ranks (Dict[Tuple[str, str], int]): Ranking of BPE merge operations.
        cache (Dict[str, str]): Cache for BPE results to avoid recomputation.
        pat (re.Pattern): Regex pattern for tokenization.
    """

    def __init__(self, encoder: Dict[str, int], bpe_merges: List[Tuple[str, str]], errors: str = "replace") -> None:
        """
        Initialize the BPE encoder with vocabulary and merge rules.

        Args:
            encoder (Dict[str, int]): Dictionary mapping BPE tokens to IDs.
            bpe_merges (List[Tuple[str, str]]): List of BPE merge operations in order of priority.
            errors (str): How to handle decoding errors. Defaults to "replace".
        """
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Regex pattern for tokenization - handles contractions, letters, numbers, etc.
        # Should have added re.IGNORECASE for capitalized contractions but keeping original
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token: str) -> str:
        """
        Apply Byte Pair Encoding algorithm to a token.

        This method iteratively finds the most frequent pair of symbols in the token and merges them according to the
        learned BPE merge rules. The process continues until no more merges can be applied.

        Args:
            token (str): The input token to encode.

        Returns:
            str: The BPE-encoded token with spaces separating sub-tokens.
        """
        # Check cache first to avoid recomputation
        if token in self.cache:
            return self.cache[token]

        # Convert token to tuple of characters for processing
        symbol_sequence = tuple(token)
        symbol_pairs = get_pairs(symbol_sequence)

        # If no pairs exist (single character), return as-is
        if not symbol_pairs:
            return token

        while True:
            # Find the pair with the highest priority (lowest rank number)
            # If no pair exists in our merge rules, it gets infinite rank
            highest_priority_pair = min(symbol_pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

            # If the highest priority pair isn't in our merge rules, we're done
            if highest_priority_pair not in self.bpe_ranks:
                break

            # Apply the merge: combine the pair into a single symbol
            first_symbol, second_symbol = highest_priority_pair
            new_sequence = []
            i = 0

            # Scan through the sequence and merge occurrences of the pair
            while i < len(symbol_sequence):
                try:
                    # Find the next occurrence of the first symbol
                    next_first_index = symbol_sequence.index(first_symbol, i)
                    # Add all symbols up to this point
                    new_sequence.extend(symbol_sequence[i:next_first_index])
                    i = next_first_index
                except ValueError:
                    # No more occurrences of first symbol, add remaining symbols
                    new_sequence.extend(symbol_sequence[i:])
                    break

                # Check if we can merge at current position
                if symbol_sequence[i] == first_symbol and i < len(symbol_sequence) - 1 and symbol_sequence[i + 1] == second_symbol:
                    # Merge the pair into a single symbol
                    new_sequence.append(first_symbol + second_symbol)
                    i += 2
                else:
                    # Can't merge, add the symbol and continue
                    new_sequence.append(symbol_sequence[i])
                    i += 1

            # Update the sequence with merged symbols
            symbol_sequence = tuple(new_sequence)

            # If we're down to a single symbol, we're done. Otherwise, get new pairs for next iteration.
            if len(symbol_sequence) == 1:
                break
            else:
                symbol_pairs = get_pairs(symbol_sequence)

        # Convert back to space-separated string and cache the result
        result = " ".join(symbol_sequence)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs using BPE.

        This method first tokenizes the input text using regex patterns, then converts each token to byte-level
        representation, applies BPE encoding, and finally maps to token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of token IDs representing the encoded text.
        """
        bpe_token_ids = []

        # Split text into tokens using regex pattern
        for raw_token in re.findall(self.pat, text):
            # Convert token to byte-level unicode representation
            byte_encoded_token = "".join(self.byte_encoder[byte_val] for byte_val in raw_token.encode("utf-8"))

            # Apply BPE encoding and convert each sub-token to ID
            bpe_encoded = self.bpe(byte_encoded_token)
            bpe_token_ids.extend(self.encoder[sub_token] for sub_token in bpe_encoded.split(" "))

        return bpe_token_ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        This method converts token IDs back to BPE tokens, joins them, converts from byte-level unicode representation
        back to bytes, and finally decodes to UTF-8 text.

        Args:
            tokens (List[int]): A list of token IDs to decode.

        Returns:
            str: The decoded text string.
        """
        # Convert token IDs back to BPE tokens and join
        bpe_text = "".join([self.decoder[token_id] for token_id in tokens])

        # Convert from byte-level unicode back to bytes, then decode to UTF-8
        byte_array = bytearray([self.byte_decoder[unicode_char] for unicode_char in bpe_text])
        decoded_text = byte_array.decode("utf-8", errors=self.errors)

        return decoded_text


def get_encoder(model_name: str, models_dir: str) -> Encoder:
    """
    Load a pre-trained BPE encoder from disk.

    Loads the encoder vocabulary and BPE merge rules from the specified model directory and creates an Encoder instance.

    Args:
        model_name (str): Name of the model subdirectory.
        models_dir (str): Path to the directory containing model files.

    Returns:
        Encoder: A configured BPE encoder instance.

    Raises:
        FileNotFoundError: If the required files don't exist.
        json.JSONDecodeError: If encoder.json is malformed.
    """
    # Load the vocabulary mapping
    encoder_path = os.path.join(models_dir, model_name, "encoder.json")
    with open(encoder_path, "r") as f:
        encoder_vocab = json.load(f)

    # Load the BPE merge rules
    bpe_path = os.path.join(models_dir, model_name, "vocab.bpe")
    with open(bpe_path, "r", encoding="utf-8") as f:
        bpe_data = f.read()

    # Parse merge rules (skip header line and empty last line)
    merge_lines = bpe_data.split("\n")[1:-1]
    bpe_merges = [tuple(merge_str.split()) for merge_str in merge_lines]

    return Encoder(encoder=encoder_vocab, bpe_merges=bpe_merges)


def download_vocab() -> None:
    """
    Download GPT-2 vocabulary files from OpenAI's public storage.

    Downloads the encoder.json and vocab.bpe files required for GPT-2 tokenization and saves them to a local directory
    named 'tokenizers'. Creates the directory if it doesn't exist.

    The files are downloaded with progress bars showing download status.
    """
    # Create directory for model files
    model_dir = "tokenizers"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Normalize path for Windows compatibility
    if os.name == 'nt':
        model_dir = model_dir.replace("\\", "/")

    # Download required files
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models/117M/"
    files_to_download = ["encoder.json", "vocab.bpe"]

    for filename in files_to_download:
        _logger.info(f"Downloading {filename}...")
        response = requests.get(base_url + filename, stream=True)
        response.raise_for_status()  # Raise an exception for bad responses

        file_path = os.path.join(model_dir, filename)
        with open(file_path, "wb") as f:
            file_size = int(response.headers.get("content-length", 0))
            chunk_size = 1000  # 1KB chunks (good for Ethernet packet size ~1500 bytes)

            with tqdm(ncols=100, desc=f"Fetching {filename}",
                     total=file_size, unit_scale=True) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))
    _logger.info("Download complete.")
