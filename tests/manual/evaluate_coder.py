#!/usr/bin/env python3
"""
Script to evaluate coder model responses using Ollama.
"""
import argparse
import json
import ollama

from tqdm import tqdm
from src.utils.ollama import coding_format_input
from src.utils.logger import g_logger


CODE_EVALUATION_PROMPT = """You are a code reviewer evaluating Python code solutions.

**Task/Instruction:** {instruction}
{input_section}
**Expected Solution:**
```python
{output}
```

**Student's Solution:**
```python
{response}
```

Evaluate the student's solution based on:
1. **Correctness**: Does it solve the problem correctly?
2. **Syntax**: Is the code syntactically valid Python?
3. **Logic**: Does it use reasonable approach (even if different from expected)?
4. **Completeness**: Does it handle the requirements?

Score from 0-100 where:
- 100: Perfect or equivalent solution
- 80-99: Works correctly with minor style differences
- 60-79: Works but has minor issues
- 40-59: Partially works or has logical errors
- 20-39: Major issues but shows understanding
- 0-19: Completely wrong or invalid syntax

Respond with ONLY the integer score (0-100)."""


def evaluate_code_responses(results_file: str, tester_model: str = "llama3", seed: int = 123):
    """
    Evaluate code responses using a code-specific prompt.

    Args:
        results_file: Path to JSON file with model responses
        tester_model: Ollama model to use for evaluation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (average_score, list of individual scores)
    """
    g_logger.info(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        test_responses = json.load(f)

    g_logger.info(f"Evaluating {len(test_responses)} responses using {tester_model}")

    scores = []
    for entry in tqdm(test_responses, desc="Scoring code responses"):
        if not entry.get("model_response"):
            scores.append(0)
            continue

        # Build input section
        input_section = ""
        if entry.get("input"):
            input_section = f"**Input:** {entry['input']}\n"

        # Create evaluation prompt
        prompt = CODE_EVALUATION_PROMPT.format(
            instruction=entry["instruction"],
            input_section=input_section,
            output=entry["output"],
            response=entry["model_response"]
        )

        # Query Ollama
        try:
            response = ollama.chat(
                model=tester_model,
                messages=[{"role": "user", "content": prompt}],
                options={"seed": seed, "temperature": 0.0, "num_ctx": 4096}
            )

            # Parse score
            score_text = response["message"]["content"].strip()
            # Extract first number found
            import re
            match = re.search(r'\b(\d{1,3})\b', score_text)
            score = int(match.group(1)) if match else 0
            scores.append(score)

        except Exception as e:
            g_logger.warning(f"Error evaluating response: {e}")
            scores.append(0)

    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score, scores


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coder model responses using Ollama",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="models/coder/coder_results.json",
        help="Path to the JSON file containing model responses"
    )
    parser.add_argument(
        "--tester-model",
        type=str,
        default="llama3",
        help="Ollama model to use for evaluation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to save detailed scores as JSON"
    )
    args = parser.parse_args()

    # Evaluate using improved code-specific evaluation
    avg_score, scores = evaluate_code_responses(
        results_file=args.results_file,
        tester_model=args.tester_model,
        seed=args.seed
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Total responses: {len(scores)}")
    print(f"Average score: {avg_score:.2f}%")
    print(f"Min score: {min(scores) if scores else 0}")
    print(f"Max score: {max(scores) if scores else 0}")
    print(f"{'='*60}\n")

    # Save detailed scores if requested
    if args.output_file:
        output_data = {
            "average_score": avg_score,
            "total_responses": len(scores),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "scores": scores
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Detailed scores saved to {args.output_file}")


if __name__ == "__main__":
    main()
