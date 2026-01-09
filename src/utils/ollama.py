import json
import ollama
import psutil
import re

from logging import getLogger as get_logger
from tqdm import tqdm


_logger = get_logger(__name__)


INSTRUCTION_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{instruction}"
)


def format_input(entry: dict[str, str]) -> str:
    instruction_text = INSTRUCTION_TEMPLATE.format(instruction=entry['instruction'])
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class OllamaEvaluator:

    PROMPT_TEMPLATE = (
        "Given the input `{input}` and correct output `{output}`, score the model response `{response}` on a scale "
        "from 0 to 100, where 100 is the best score. Respond with the integer number only."
    )

    def __init__(self, tester: str = "llama3", seed: int = 123) -> None:
        self.tester = tester
        self.seed = seed
        _logger.info(f"Initialized {self.__class__.__name__} with tester model '{self.tester}' and seed {self.seed}")

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
            _logger.warning(f"Could not parse score from response: {response}")
            score = 0
        return score

    def _get_scores(self, test_responses: list, answer_key: str) -> list[int]:
        scores = []
        for entry in tqdm(test_responses, desc="Scoring entries"):
            entry: dict
            if not entry.get(answer_key):
                scores.append(0)
            else:
                prompt = self.PROMPT_TEMPLATE.format(input=format_input(entry), output=entry['output'], response=entry[answer_key])
                response = self.query_tester(prompt)
                score = self._parse_score(response)
                scores.append(score)
        return scores

    def evaluate(self, test_responses_json_path: str, answer_key = "model_response") -> tuple[float, list[int]]:
        _logger.info("Starting Ollama evaluation...")

        if not self.is_server_running():
            raise RuntimeError("Ollama server is not running. Please start it by running `ollama serve` and try again.")
        _logger.info("Ollama server is running")

        with open(test_responses_json_path, "r") as file:
            test_responses = json.load(file)
        _logger.info(f"Loaded {len(test_responses)} test responses from {test_responses_json_path}")

        scores = self._get_scores(test_responses, answer_key)
        avg_score = sum(scores) / len(scores) if scores else 0
        _logger.info(f"Average score: {avg_score:.2f}% (n={len(scores)})")

        return avg_score, scores
