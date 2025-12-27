import os
import pytest
import shutil
import tempfile

from src.gpt_download import download_and_load_gpt2, FILES_TO_DOWNLOAD


@pytest.mark.skip(reason="Downloads files from the internet takes time, run manually when needed")
def test_download_and_load_gpt2_124M_creates_files():
    temp_dir = tempfile.mkdtemp()
    try:
        settings, params = download_and_load_gpt2("124M", temp_dir)
        model_dir = os.path.join(temp_dir, "124M")

        # Check that required files are downloaded
        for fname in FILES_TO_DOWNLOAD:
            fpath = os.path.join(model_dir, fname)
            assert os.path.exists(fpath), f"File missing: {fpath}"

        # Check settings keys
        expected_settings_keys = {"n_vocab", "n_ctx", "n_embd", "n_head", "n_layer"}
        assert expected_settings_keys.issubset(set(settings.keys())), f"Settings keys missing: {set(settings.keys())}"

        # Check params keys
        expected_params_keys = {"blocks", "wte", "wpe"}
        assert expected_params_keys.issubset(set(params.keys())), f"Params keys missing: {set(params.keys())}"

    finally:
        shutil.rmtree(temp_dir)


def test_download_and_load_gpt2_invalid_size_raises():
    temp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError):
            download_and_load_gpt2("invalid_size", temp_dir)
    finally:
        shutil.rmtree(temp_dir)
