from tests.common import run_subprocess, print_title


def test_generate_cli_vs_chapter_script():

    # Expected output (validated from chapter script)
    expected_text = ("Every effort moves you toward finding an ideal life. You don't have to accept your problems by "
                     "trying to remedy them, because that would be foolish")

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

    print_title("Running CLI generate command")
    cli_output: str = run_subprocess(cmd)
    model_output = cli_output.splitlines()[-1]

    assert model_output, f"Could not extract output text from CLI. Output was:\n{cli_output}"
    assert model_output == expected_text, f"Output mismatch!\n  Expected: {expected_text}\n  Got:      {model_output}"
