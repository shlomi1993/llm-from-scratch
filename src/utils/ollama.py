import json
import ollama
import psutil
import re

from tqdm import tqdm
from typing import Callable

from src.utils.logger import g_logger


def format_input(entry: dict[str, str]) -> str:
    """
    Format a dataset entry dict into a string suitable as a prompt for a language model.

    NOTE: This format is not ideal because it omits the trailing `\n\n### Response:\n`, but it is kept to remain
    consistent with the source repository notebook `ch07/01_main-chapter-code/ch07.ipynb`, so that training with the
    same seed produces identical losses.

    A more correct format would be:
    PROMPT_TEMPLATE = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n\n"
    )

    The prompt can then be constructed using PROMPT_TEMPLATE.format(...).

    Args:
        entry (dict): A dictionary containing at least the keys 'instruction' and 'input'.

    Returns:
        str: A formatted string suitable as a prompt for a language model.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def coding_format_input(entry: dict[str, str]) -> str:
    """
    Format a dataset entry dict into a string suitable as a prompt for coding tasks.

    Args:
        entry (dict): A dictionary containing at least the key 'instruction'.

    Returns:
        str: A formatted string suitable as a prompt for coding tasks.
    """
    return f"### Instruction:\n{entry['instruction']}\n\n### Response:\n"


class OllamaEvaluator:
    """
    Evaluates model responses using a language model through the Ollama Python API.

    Note that the notebook of chapter 7 uses a standard server based API call to Ollama, while this class uses
    the Ollama Python API for simplicity.
    """

    PROMPT_TEMPLATE = (
        "Given the input `{input}` and correct output `{output}`, score the model response `{response}` on a scale "
        "from 0 to 100, where 100 is the best score. Respond with the integer number only."
    )

    def __init__(self, tester: str = "llama3", seed: int = 123, formatter: Callable = format_input) -> None:
        """
        Initialize the OllamaEvaluator.

        Args:
            tester (str): The name of the Ollama model to use for evaluation. Default is "llama3".
            seed (int): Seed for the Ollama model to ensure reproducibility. Default is 123.
            formatter (Callable): A function that formats a dataset entry into a prompt string. Default is `format_input`.
        """
        self.tester = tester
        self.seed = seed
        self.formatter = formatter
        g_logger.info(f"Initialized {self.__class__.__name__} with tester model '{self.tester}' and seed {self.seed}")

    @staticmethod
    def is_server_running() -> bool:
        """
        Check if the Ollama server is currently running.

        Note that this check is needed even when using the Ollama Python API, as the API requires the server to be running.

        Returns:
            bool: True if the server is running, False otherwise.
        """
        return any("ollama" in proc.info["name"] for proc in psutil.process_iter(["name"]))

    def query_tester(self, prompt: str, temperature: float = 0.0, num_ctx: int = 2048) -> str:
        """
        Query the Ollama tester model with a given prompt.

        Args:
            prompt (str): The prompt to send to the model.
            temperature (float, optional): The temperature setting for the model. Default is 0.0 (deterministic).
            num_ctx (int, optional): The maximum context length for the model. Default is 2048.

        Returns:
            str: The model's response to the prompt.
        """
        response = ollama.chat(
            model=self.tester,
            messages=[{"role": "user", "content": prompt}],
            options={"seed": self.seed, "temperature": temperature, "num_ctx": num_ctx}
        )
        return response["message"]["content"]

    def _parse_score(self, response: str) -> int:
        """
        Parse the integer score from the model's response using regex.

        Args:
            response (str): The model's response to the prompt.

        Returns:
            int: The parsed integer score.
        """
        match = re.search(r"\b(\d{1,3})\b", response)
        score = match.group(1) if match else response
        try:
            score = int(score)
        except ValueError:
            g_logger.warning(f"Could not parse score from response: {response}")
            score = 0
        return score

    def _get_scores(self, test_responses: list, answer_key: str) -> list[int]:
        """
        Get scores for a list of test responses.

        Args:
            test_responses (list): A list of dictionaries containing test responses.
            answer_key (str): The key in the dictionary that contains the model's response.

        Returns:
            list[int]: A list of scores for each test response. If the answer_key is missing, a score of 0 is assigned.
        """
        scores = []
        for entry in tqdm(test_responses, desc="Scoring", leave=True):
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
        """
        Evaluate model responses stored in a JSON file.

        Args:
            test_responses_json_path (str): Path to the JSON file containing test responses.
            answer_key (str, optional): The key in the JSON file that contains the model's response. Default is "model_response".

        Returns:
            tuple[float, list[int]]: A tuple containing the average score and a list of individual scores.
        """
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
