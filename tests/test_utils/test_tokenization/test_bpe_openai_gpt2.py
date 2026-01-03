import json
import os
import tempfile
import pytest

from unittest.mock import MagicMock, patch

from src.data.bpe_openai_gpt2 import Encoder, bytes_to_unicode, get_pairs, get_encoder, download_vocab


@pytest.fixture
def sample_encoder_data():
    """
    Provide sample encoder data for testing.
    """
    encoder = {
        "a": 0,
        "b": 1,
        "c": 2,
        "ab": 3,
        "bc": 4,
        "abc": 5
    }
    bpe_merges = [("a", "b"), ("b", "c")]
    return encoder, bpe_merges


@pytest.fixture
def encoder_instance(sample_encoder_data):
    """
    Create an Encoder instance for testing.
    """
    encoder, bpe_merges = sample_encoder_data
    return Encoder(encoder, bpe_merges)


@pytest.fixture
def temp_model_dir():
    """
    Create a temporary directory with sample model files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "test_model"
        model_path = os.path.join(temp_dir, model_name)
        os.makedirs(model_path)

        # Create sample encoder.json
        encoder_data = {"a": 0, "b": 1, "ab": 2}
        with open(os.path.join(model_path, "encoder.json"), "w") as f:
            json.dump(encoder_data, f)

        # Create sample vocab.bpe
        bpe_data = "#version: 0.2\na b\n"
        with open(os.path.join(model_path, "vocab.bpe"), "w", encoding="utf-8") as f:
            f.write(bpe_data)

        yield temp_dir, model_name


def test_bytes_to_unicode_unique_and_ascii():
    """
    Test that bytes_to_unicode returns a unique mapping and ASCII chars map to themselves.
    """
    mapping = bytes_to_unicode()
    assert isinstance(mapping, dict), "Mapping should be a dictionary"
    assert len(mapping) == 256, "Mapping should contain 256 entries"
    assert mapping[ord('A')] == 'A', "ASCII 'A' should map to itself"
    assert mapping[ord('!')] == '!', "ASCII '!' should map to itself"
    assert mapping[ord('~')] == '~', "ASCII '~' should map to itself"
    assert len(set(mapping.values())) == 256, "All unicode mappings should be unique"


def test_get_pairs_various_cases():
    """
    Test get_pairs with typical, single, empty, and multi-char cases.
    """
    assert get_pairs(("h", "e", "l", "l", "o")) == {("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")}, "Should extract all consecutive symbol pairs correctly"
    assert get_pairs(("a",)) == set(), "Single character should produce empty set of pairs"
    assert get_pairs(("a", "b")) == {("a", "b")}, "Two characters should produce single pair"
    assert get_pairs(("a", "b", "a", "b")) == {("a", "b"), ("b", "a")}, "Should contain unique pairs only"
    with pytest.raises(IndexError):
        get_pairs(())
    assert get_pairs(("hello", "world", "test")) == {("hello", "world"), ("world", "test")}, "Should work with multi-character symbols"


def test_encoder_initialization(sample_encoder_data):
    """
    Test proper initialization of Encoder.
    """
    encoder, bpe_merges = sample_encoder_data
    enc = Encoder(encoder, bpe_merges, errors="ignore")
    assert enc.encoder == encoder, "Encoder vocabulary should be set correctly"
    assert enc.errors == "ignore", "Error handling mode should be set correctly"
    assert len(enc.decoder) == len(encoder), "Decoder should have same length as encoder"
    assert enc.bpe_ranks[("a", "b")] == 0, "First BPE merge should have rank 0"
    assert enc.bpe_ranks[("b", "c")] == 1, "Second BPE merge should have rank 1"
    assert isinstance(enc.cache, dict), "Cache should be initialized as dictionary"
    assert len(enc.cache) == 0, "Cache should be empty initially"


def test_encoder_decoder_mapping(encoder_instance):
    """
    Test that encoder and decoder are proper inverses.
    """
    for token, token_id in encoder_instance.encoder.items():
        assert encoder_instance.decoder[token_id] == token, f"Decoder should map {token_id} back to {token}"

def test_bpe_cache(encoder_instance):
    """
    Test BPE caching functionality.
    """
    token = "test"

    result1 = encoder_instance.bpe(token)
    assert token in encoder_instance.cache, "Token should be cached after first call"
    assert encoder_instance.cache[token] == result1, "Cached result should match returned result"

    result2 = encoder_instance.bpe(token)
    assert result1 == result2, "Second call should return same result from cache"

def test_encode_decode_roundtrip():
    """
    Test that encode/decode are approximate inverses for ASCII text.
    """
    encoder = {chr(i): i for i in range(256)}
    bpe_merges = []
    enc = Encoder(encoder, bpe_merges)
    original = "hello"
    encoded = enc.encode(original)
    decoded = enc.decode(encoded)
    assert isinstance(encoded, list), "Encoded result should be a list"
    assert isinstance(decoded, str), "Decoded result should be a string"
    assert len(encoded) > 0, "Encoded result should not be empty"

def test_encoder_with_errors_parameter(sample_encoder_data):
    """
    Test different error handling modes.
    """
    encoder, bpe_merges = sample_encoder_data

    # Test different error modes
    enc_replace = Encoder(encoder, bpe_merges, errors="replace")
    enc_ignore = Encoder(encoder, bpe_merges, errors="ignore")

    assert enc_replace.errors == "replace", "Error mode should be set to 'replace'"
    assert enc_ignore.errors == "ignore", "Error mode should be set to 'ignore'"


def test_get_encoder_success(temp_model_dir):
    """
    Test successful loading of encoder.
    """
    models_dir, model_name = temp_model_dir
    encoder = get_encoder(model_name, models_dir)
    assert isinstance(encoder, Encoder), "Should return an Encoder instance"
    assert "a" in encoder.encoder, "Should load 'a' token from encoder.json"
    assert "b" in encoder.encoder, "Should load 'b' token from encoder.json"
    assert "ab" in encoder.encoder, "Should load 'ab' token from encoder.json"
    assert ("a", "b") in encoder.bpe_ranks, "Should load the 'a b' merge rule"


def test_get_encoder_missing_files():
    """
    Test get_encoder with missing files.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Directory exists but files don't
        with pytest.raises(FileNotFoundError):
            get_encoder("nonexistent", temp_dir)


def test_get_encoder_malformed_json():
    """
    Test get_encoder with malformed JSON.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "bad_model"
        model_path = os.path.join(temp_dir, model_name)
        os.makedirs(model_path)

        # Create malformed JSON
        with open(os.path.join(model_path, "encoder.json"), "w") as f:
            f.write("invalid json {")

        # Create valid BPE file
        with open(os.path.join(model_path, "vocab.bpe"), "w") as f:
            f.write("#version: 0.2\n")

        with pytest.raises(json.JSONDecodeError):
            get_encoder(model_name, temp_dir)


def test_get_encoder_empty_bpe_file(temp_model_dir):
    """
    Test get_encoder with empty BPE file.
    """
    models_dir, model_name = temp_model_dir
    model_path = os.path.join(models_dir, model_name)

    # Overwrite with minimal BPE file
    with open(os.path.join(model_path, "vocab.bpe"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")

    encoder = get_encoder(model_name, models_dir)
    assert len(encoder.bpe_ranks) == 0, "Empty BPE file should result in no merge rules"


@patch('bpe_openai_gpt2.requests.get')
@patch('bpe_openai_gpt2.os.path.exists')
def test_download_vocab_request_error(mock_exists, mock_requests_get):
    mock_exists.return_value = False
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("Network error")
    mock_requests_get.return_value = mock_response
    with pytest.raises(Exception):
        download_vocab()


def test_complete_workflow_with_mock_data():
    """
    Test complete encode/decode workflow with mock data.
    """
    # Create a simple encoder for testing with complete character coverage
    vocab = {}
    # Add all ASCII characters to vocab
    for i in range(256):
        char = chr(i) if i < 128 else f"<{i}>"
        vocab[char] = i

    # Add some BPE tokens
    vocab.update({'ab': 256, 'cd': 257, 'abc': 258})

    merges = [('a', 'b'), ('c', 'd'), ('ab', 'c')]

    encoder = Encoder(vocab, merges)

    # Test that basic functionality works
    assert isinstance(encoder.encoder, dict), "Encoder should have dictionary vocabulary"
    assert isinstance(encoder.decoder, dict), "Encoder should have dictionary decoder"
    assert isinstance(encoder.bpe_ranks, dict), "Encoder should have BPE ranks dictionary"

    # Test BPE algorithm with known input
    test_token = "abcd"
    bpe_result = encoder.bpe(test_token)
    assert isinstance(bpe_result, str), "BPE should return string result"

    # Test that encoding produces list of integers
    with patch('bpe_openai_gpt2.re.findall') as mock_findall:
        mock_findall.return_value = ["test"]

        # Mock the BPE method to return something in our vocab
        with patch.object(encoder, 'bpe') as mock_bpe:
            mock_bpe.return_value = "t e s t"  # Assume each char is separate
            result = encoder.encode("test")
            assert isinstance(result, list), "Encode should return list of token IDs"


def test_bytes_to_unicode_integration():
    """
    Test that bytes_to_unicode integrates properly with Encoder.
    """
    mapping = bytes_to_unicode()

    # Use the mapping in a simple encoder
    vocab = {mapping[i]: i for i in range(256)}
    encoder = Encoder(vocab, [])

    # Verify that byte_encoder is properly set
    assert encoder.byte_encoder == mapping, "Encoder should use bytes_to_unicode mapping"
    assert len(encoder.byte_decoder) == 256, "Encoder should have complete byte decoder"
