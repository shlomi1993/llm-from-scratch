import json
import ollama
import psutil
import re

from tqdm import tqdm
from typing import Callable

from src.utils.logger import g_logger


def format_input(entry: dict[str, str]) -> str:
    """
    NOTE: This format is not ideal because it omits the trailing `\n\n### Response:\n`, but it is kept to remain
    consistent with the source repository notebook `ch07/01_main-chapter-code/ch07.ipynb`, so that training with the
    same seed produces identical losses.

    A more correct format would be:
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n\n"
    )

    The prompt can then be constructed using PROMPT_TEMPLATE.format(...).
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def coding_format_input(entry: dict) -> str:
    return f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"


class OllamaEvaluator:

    PROMPT_TEMPLATE = (
        "Given the input `{input}` and correct output `{output}`, score the model response `{response}` on a scale "
        "from 0 to 100, where 100 is the best score. Respond with the integer number only."
    )

    def __init__(self, tester: str = "llama3", seed: int = 123, formatter: Callable = format_input) -> None:
        self.tester = tester
        self.seed = seed
        self.formatter = formatter
        g_logger.info(f"Initialized {self.__class__.__name__} with tester model '{self.tester}' and seed {self.seed}")

    @staticmethod
    def is_server_running() -> bool:
        return any("ollama" in proc.info["name"] for proc in psutil.process_iter(["name"]))

    def query_tester(self, prompt: str, temperature: float = 0.0, num_ctx: int = 2048) -> str:
        response = ollama.chat(
            model=self.tester,
            messages=[{"role": "user", "content": prompt}],
            options={"seed": self.seed, "temperature": temperature, "num_ctx": num_ctx}
        )
        return response["message"]["content"]

    def _parse_score(self, response: str) -> int:
        match = re.search(r"\b(\d{1,3})\b", response)
        score = match.group(1) if match else response
        try:
            score = int(score)
        except ValueError:
            g_logger.warning(f"Could not parse score from response: {response}")
            score = 0
        return score

    def _get_scores(self, test_responses: list, answer_key: str) -> list[int]:
        scores = []
        for entry in tqdm(test_responses, desc="Scoring", leave=False):
            entry: dict
            if not entry.get(answer_key):
                scores.append(0)
            else:
                prompt = self.PROMPT_TEMPLATE.format(input=self.formatter(entry), output=entry['output'], response=entry[answer_key])
                response = self.query_tester(prompt)
                score = self._parse_score(response)
                scores.append(score)
        return scores

    def evaluate(self, test_responses_json_path: str, answer_key = "model_response") -> tuple[float, list[int]]:
        g_logger.info("Starting Ollama evaluation...")

        if not self.is_server_running():
            raise RuntimeError("Ollama server is not running. Please start it by running `ollama serve` and try again.")
        g_logger.info("Ollama server is running")

        with open(test_responses_json_path, "r") as file:
            test_responses = json.load(file)
        g_logger.info(f"Loaded {len(test_responses)} test responses from {test_responses_json_path}")

        scores = self._get_scores(test_responses, answer_key)
        avg_score = sum(scores) / len(scores) if scores else 0
        g_logger.info(f"Average score: {avg_score:.2f}% (n={len(scores)})")

        return avg_score, scores
