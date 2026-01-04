import argparse
import json
import psutil
import requests

from logging import getLogger as get_logger
from tqdm import tqdm

from src.data.formatting import format_input


_logger = get_logger(__name__)


def query_model(prompt: str, model="llama3", url="http://localhost:11434/api/chat") -> str:

    # Prepare the request payload
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {  # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }

    # Send the POST request
    with requests.post(url, json=data, stream=True, timeout=30) as r:
        r.raise_for_status()
        response_data = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            response_json = json.loads(line)
            if "message" in response_json:
                response_data += response_json["message"]["content"]

    return response_data


def is_ollama_running() -> bool:
    return any("ollama" in proc.info["name"] for proc in psutil.process_iter(["name"]))


def generate_model_scores(json_data: list, json_key: str, model: str = "llama3") -> list:
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":
            scores.append(0)
        else:
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                _logger.warning(f"Could not convert score: {score}")
                continue
    return scores


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file-path", required=True, help="Path to a test dataset JSON file with 'output' and 'model_response' keys")
    parser.add_argument("--model", type=str, default="llama3", help="Model name to use with Ollama API")


def run_ollama_evaluation_flow(file_path: str = None, model: str = "llama3") -> None:
    ollama_running = is_ollama_running()
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

    with open(file_path, "r") as file:
        test_data = json.load(file)

    scores = generate_model_scores(test_data, "model_response", model)
    avg_score = sum(scores) / len(scores) if scores else 0

    _logger.info(f"Number of scores: {len(scores)} of {len(test_data)}")
    _logger.info(f"Average score: {avg_score:.2f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model responses using Ollama API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arguments(parser)
    args = parser.parse_args()
    run_ollama_evaluation_flow(args.file_path, args.model)


if __name__ == "__main__":
    main()
