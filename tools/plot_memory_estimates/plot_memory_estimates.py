#!/usr/bin/env python3
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Unified memory estimation plotting tool.

This tool generates plots for KV-cache memory usage across different attention mechanisms
(MHA, GQA, MLA, SWA) and MoE FFN parameter counts.
"""

from src.parser import parse_args, validate_args
from src.plotters import plot_gqa, plot_mla, plot_swa, plot_moe


def main() -> int:
    """
    Main entry point for the plotting tool.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.mode == "gqa":
        plot_gqa(args)
    elif args.mode == "mla":
        plot_mla(args)
    elif args.mode == "swa":
        plot_swa(args)
    elif args.mode == "moe":
        plot_moe(args)

    return 0


if __name__ == "__main__":
    exit(main())
