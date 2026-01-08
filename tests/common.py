from subprocess import Popen, PIPE, STDOUT
from sys import stdout


def run_subprocess(cmd: list[str] | str, cwd: str = None) -> list[str]:
    output_lines = []
    shell = isinstance(cmd, str)
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, cwd=cwd, bufsize=1, shell=shell)
    for line in process.stdout:
        print(line, end='')
        stdout.flush()  # Force immediate display
        output_lines.append(line)
    process.wait()
    assert process.returncode == 0, f"Subprocess failed: {' '.join(cmd) if not shell else cmd}"
    return output_lines
