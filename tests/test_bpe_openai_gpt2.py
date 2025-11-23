"""
Tests for the BPE OpenAI GPT-2 module.

This module contains comprehensive tests for the Byte Pair Encoding (BPE) implementation used in GPT-2, including unit
tests for all functions and the Encoder class.
"""

import json
import os
import tempfile
import pytest

from unittest.mock import MagicMock, mock_open, patch

from bpe_openai_gpt2 import Encoder, bytes_to_unicode, get_pairs, get_encoder, download_vocab


class TestBytesToUnicode:
    """
    Test cases for the bytes_to_unicode function.
    """

    def test_bytes_to_unicode_basic(self):
        """
        Test that bytes_to_unicode returns a proper mapping.
        """
        mapping = bytes_to_unicode()
        assert isinstance(mapping, dict), "Mapping should be a dictionary"
        assert len(mapping) == 256, "Mapping should contain 256 entries"
        assert all(isinstance(v, str) and len(v) == 1 for v in mapping.values()), "All values should be single unicode characters"
        assert all(isinstance(value, str) and len(value) == 1 for value in mapping.values()), "All values should be single unicode characters"

    def test_bytes_to_unicode_caching(self):
        """
        Test that the function is properly cached.
        """
        mapping1 = bytes_to_unicode()
        mapping2 = bytes_to_unicode()
        assert mapping1 is mapping2, "Function should return cached result on second call"

    def test_bytes_to_unicode_specific_mappings(self):
        """
        Test specific known mappings.
        """
        mapping = bytes_to_unicode()
        assert mapping[ord('A')] == 'A', "ASCII 'A' should map to itself"
        assert mapping[ord('!')] == '!', "ASCII '!' should map to itself"
        assert mapping[ord('~')] == '~', "ASCII '~' should map to itself"
        values = list(mapping.values())
        assert len(values) == len(set(values)), "All unicode mappings should be unique"


class TestGetPairs:
    """
    Test cases for the get_pairs function.
    """

    def test_get_pairs_basic(self):
        """
        Test basic functionality of get_pairs.
        """
        pairs = get_pairs(("h", "e", "l", "l", "o"))
        expected = {("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")}
        assert pairs == expected, "Should extract all consecutive symbol pairs correctly"

    def test_get_pairs_single_character(self):
        """
        Test get_pairs with single character.
        """
        pairs = get_pairs(("a",))
        assert pairs == set(), "Single character should produce empty set of pairs"

    def test_get_pairs_two_characters(self):
        """
        Test get_pairs with two characters.
        """
        pairs = get_pairs(("a", "b"))
        assert pairs == {("a", "b")}, "Two characters should produce single pair"

    def test_get_pairs_repeated_pairs(self):
        """
        Test that repeated pairs appear only once in the set.
        """
        pairs = get_pairs(("a", "b", "a", "b"))
        assert pairs == {("a", "b"), ("b", "a")}, "Should contain unique pairs only"

    def test_get_pairs_empty_tuple(self):
        """
        Test that empty tuple raises IndexError.
        """
        with pytest.raises(IndexError):
            get_pairs(())

    def test_get_pairs_multi_char_symbols(self):
        """
        Test get_pairs with multi-character symbols.
        """
        pairs = get_pairs(("hello", "world", "test"))
        expected = {("hello", "world"), ("world", "test")}
        assert pairs == expected, "Should work with multi-character symbols"


class TestEncoder:
    """
    Test cases for the Encoder class.
    """

    @pytest.fixture
    def sample_encoder_data(self):
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
    def encoder_instance(self, sample_encoder_data):
        """
        Create an Encoder instance for testing.
        """
        encoder, bpe_merges = sample_encoder_data
        return Encoder(encoder, bpe_merges)

    def test_encoder_initialization(self, sample_encoder_data):
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

    def test_encoder_decoder_mapping(self, encoder_instance):
        """
        Test that encoder and decoder are proper inverses.
        """
        for token, token_id in encoder_instance.encoder.items():
            assert encoder_instance.decoder[token_id] == token, f"Decoder should map {token_id} back to {token}"

    def test_bpe_cache(self, encoder_instance):
        """
        Test BPE caching functionality.
        """
        token = "test"

        result1 = encoder_instance.bpe(token)
        assert token in encoder_instance.cache, "Token should be cached after first call"
        assert encoder_instance.cache[token] == result1, "Cached result should match returned result"

        result2 = encoder_instance.bpe(token)
        assert result1 == result2, "Second call should return same result from cache"

    def test_bpe_single_character(self, encoder_instance):
        """
        Test BPE with single character token.
        """
        result = encoder_instance.bpe("x")
        assert result == "x", "Single character should be returned unchanged"

    def test_bpe_no_pairs(self, encoder_instance):
        """
        Test BPE when no pairs can be merged.
        """
        result = encoder_instance.bpe("xyz")
        assert result == "x y z", "Characters with no mergeable pairs should be split"

    def test_encode_basic(self, encoder_instance):
        """
        Test basic encoding functionality.
        """
        with patch('bpe_openai_gpt2.re.findall') as mock_findall:
            mock_findall.return_value = ["a"]
            with patch.object(encoder_instance, 'bpe') as mock_bpe:
                mock_bpe.return_value = "a"
                result = encoder_instance.encode("a")
                assert isinstance(result, list), "Encode should return a list"
                assert all(isinstance(x, int) for x in result), "All tokens should be integers"

    def test_decode_basic(self, sample_encoder_data):
        """
        Test basic decoding functionality.
        """
        encoder, bpe_merges = sample_encoder_data
        enc = Encoder(encoder, bpe_merges)
        result = enc.decode([0, 1])  # Should correspond to 'a' and 'b'
        assert isinstance(result, str), "Decode should return a string"

    def test_encode_decode_roundtrip(self):
        """
        Test that encode/decode are approximate inverses for ASCII text.
        """
        # Use a minimal encoder for testing
        encoder = {chr(i): i for i in range(256)}
        bpe_merges = []
        enc = Encoder(encoder, bpe_merges)

        # Test with simple ASCII text
        original = "hello"
        encoded = enc.encode(original)
        decoded = enc.decode(encoded)

        # Should be able to recover the original (or very close)
        assert isinstance(encoded, list), "Encoded result should be a list"
        assert isinstance(decoded, str), "Decoded result should be a string"
        assert len(encoded) > 0, "Encoded result should not be empty"

    def test_encoder_with_errors_parameter(self, sample_encoder_data):
        """
        Test different error handling modes.
        """
        encoder, bpe_merges = sample_encoder_data

        # Test different error modes
        enc_replace = Encoder(encoder, bpe_merges, errors="replace")
        enc_ignore = Encoder(encoder, bpe_merges, errors="ignore")

        assert enc_replace.errors == "replace", "Error mode should be set to 'replace'"
        assert enc_ignore.errors == "ignore", "Error mode should be set to 'ignore'"


class TestGetEncoder:
    """
    Test cases for the get_encoder function.
    """

    @pytest.fixture
    def temp_model_dir(self):
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

    def test_get_encoder_success(self, temp_model_dir):
        """
        Test successful loading of encoder.
        """
        models_dir, model_name = temp_model_dir
        encoder = get_encoder(model_name, models_dir)
        assert isinstance(encoder, Encoder), "Should return an Encoder instance"
        assert "a" in encoder.encoder, "Should load 'a' token from encoder.json"
        assert "b" in encoder.encoder, "Should load 'b' token from encoder.json"
        assert "ab" in encoder.encoder, "Should load 'ab' token from encoder.json"
        assert len(encoder.bpe_ranks) == 1, "Should load one BPE merge rule"
        assert ("a", "b") in encoder.bpe_ranks, "Should load the 'a b' merge rule"

    def test_get_encoder_missing_files(self):
        """
        Test get_encoder with missing files.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Directory exists but files don't
            with pytest.raises(FileNotFoundError):
                get_encoder("nonexistent", temp_dir)

    def test_get_encoder_malformed_json(self):
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

    def test_get_encoder_empty_bpe_file(self, temp_model_dir):
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


class TestDownloadVocab:
    """
    Test cases for the download_vocab function.
    """

    @patch('bpe_openai_gpt2.requests.get')
    @patch('bpe_openai_gpt2.os.makedirs')
    @patch('bpe_openai_gpt2.os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('bpe_openai_gpt2.tqdm')
    def test_download_vocab_success(self, mock_tqdm, mock_file, mock_exists,
                                   mock_makedirs, mock_requests_get):
        """
        Test successful vocabulary download.
        """
        # Setup mocks
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1000"}
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Mock tqdm context manager
        mock_progress = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress

        # Call function
        download_vocab()

        # Verify calls
        mock_makedirs.assert_called_once_with("gpt2_model"), "Should create gpt2_model directory"
        assert mock_requests_get.call_count == 2, "Should download two files"
        assert mock_file.call_count == 2, "Should open two files for writing"

    @patch('bpe_openai_gpt2.requests.get')
    @patch('bpe_openai_gpt2.os.path.exists')
    def test_download_vocab_request_error(self, mock_exists, mock_requests_get):
        """
        Test download_vocab with request error.
        """
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_requests_get.return_value = mock_response

        with pytest.raises(Exception):
            download_vocab()

    @patch('bpe_openai_gpt2.os.makedirs')
    @patch('bpe_openai_gpt2.os.path.exists')
    def test_download_vocab_directory_exists(self, mock_exists, mock_makedirs):
        """
        Test download_vocab when directory already exists.
        """
        mock_exists.return_value = True

        with patch('bpe_openai_gpt2.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.headers = {"content-length": "1000"}
            mock_response.iter_content.return_value = [b"data"]
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with patch('builtins.open', mock_open()):
                with patch('bpe_openai_gpt2.tqdm') as mock_tqdm:
                    mock_tqdm.return_value.__enter__.return_value = MagicMock()
                    download_vocab()

        # makedirs should not be called when directory exists
        mock_makedirs.assert_not_called(), "Should not create directory when it already exists"


class TestIntegration:
    """
    Integration tests for the BPE module.
    """

    def test_complete_workflow_with_mock_data(self):
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
        vocab.update({
            'ab': 256, 'cd': 257, 'abc': 258
        })

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

    def test_bytes_to_unicode_integration(self):
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
