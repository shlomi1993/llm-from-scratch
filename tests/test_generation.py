from tests.common import run_subprocess


def test_generate_cli_vs_chapter_script():
    cmd = [
        "gpt2", "generate",
        "--model-path", "./models/124M/model.pth",
        "--prompt", "Every effort moves you",
        "--max-new-tokens", "25",
        "--temperature", "1.0",
        "--top-k", "50",
        "--device", "cpu",
        "--seed", "123"
    ]

    cli_output: str = run_subprocess(cmd)
    model_output = cli_output.splitlines()[-1]

    # Expected output (validated from chapter script)
    expected_text = ("Every effort moves you toward finding an ideal life. You don't have to accept your problems by "
                     "trying to remedy them, because that would be foolish")

    assert model_output == expected_text, f"Output mismatch!\n  Expected: {expected_text}\n  Got:      {model_output}"
