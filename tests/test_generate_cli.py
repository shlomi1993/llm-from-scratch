import subprocess

from tests.common import print_title


def test_generate_cli_vs_chapter_script():

    # Expected output (validated from chapter script)
    expected_text = "Every effort moves you toward finding an ideal life. You don't have to accept your problems by trying to remedy them, because that would be foolish"

    # Compose CLI command
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

    # Run the CLI command and capture output
    print_title("Running CLI generate command")
    result_cli_capture = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)

    # Extract the generated text from CLI output
    cli_output = result_cli_capture.stdout
    cli_text = None
    lines = cli_output.split('\n')
    for i, line in enumerate(lines):
        if "Output text:" in line:
            # The text is on the next line
            if i + 1 < len(lines):
                cli_text = lines[i + 1].strip()
                break

    assert cli_text, f"Could not extract output text from CLI. Output was:\n{cli_output}"
    assert cli_text == expected_text, f"Output mismatch!\n  Expected: {expected_text}\n  Got:      {cli_text}"
