#!/usr/bin/env python3
"""Append verbalized-confidence instructions to parquet training prompts."""

import argparse
import copy
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "deepscaler_uniform_train.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "deepscaler_uniform_train_with_confidence.parquet"

CONFIDENCE_INSTRUCTION = (
    "\n\nPlease put your final answer within \\boxed{}.\n"
    "\nAlso output a singal line at the end of the answer：CONFIDENCE: <float number between 0 and 1>\n"
    "e.g. ：CONFIDENCE: 0.83\n"
    "Please make sure CONFIDENCE part is in a singal line with the exact same format。\n"
)


def append_instruction_to_prompt(prompt, instruction):
    """Return (updated_prompt, changed) for either string or chat-list prompts."""
    marker = instruction.strip()

    if isinstance(prompt, str):
        if marker in prompt:
            return prompt, False
        return prompt + instruction, True

    if isinstance(prompt, list):
        updated = copy.deepcopy(prompt)

        target_idx = None
        for idx in range(len(updated) - 1, -1, -1):
            message = updated[idx]
            if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                target_idx = idx
                break

        if target_idx is None:
            for idx in range(len(updated) - 1, -1, -1):
                message = updated[idx]
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    target_idx = idx
                    break

        if target_idx is None:
            raise TypeError(f"Unsupported prompt message format: {prompt!r}")

        content = updated[target_idx]["content"]
        if marker in content:
            return updated, False

        updated[target_idx]["content"] = content + instruction
        return updated, True

    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append final-answer and CONFIDENCE output instructions to a parquet prompt column."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input parquet path. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output parquet path.",
    )
    parser.add_argument("--prompt-key", default="prompt", help="Prompt column name.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires the `datasets` package. Install DCPO dependencies first, "
            "for example: pip install -r requirements.txt"
        ) from exc

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite to replace it.")

    dataset = load_dataset("parquet", data_files=input_path, split="train")
    if args.prompt_key not in dataset.column_names:
        raise KeyError(f"Column {args.prompt_key!r} not found. Available columns: {dataset.column_names}")

    stats = {"changed": 0, "unchanged": 0}

    def update_example(example):
        updated_prompt, changed = append_instruction_to_prompt(example[args.prompt_key], CONFIDENCE_INSTRUCTION)
        example[args.prompt_key] = updated_prompt
        stats["changed" if changed else "unchanged"] += 1
        return example

    updated_dataset = dataset.map(update_example, desc="Appending confidence instruction")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    updated_dataset.to_parquet(output_path)

    print(f"Loaded rows: {len(dataset)}")
    print(f"Updated prompts: {stats['changed']}")
    print(f"Already contained instruction: {stats['unchanged']}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
