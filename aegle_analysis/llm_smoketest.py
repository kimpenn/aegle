"""Minimal smoke test for the OpenAI chat completions client."""
"""
python 0-phenocycler-penntmc-pipeline/aegle_analysis/llm_smoketest.py --model gpt-5-2025-08-07 --prompt "Give me a two-sentence summary of the Phenocycler analysis pipeline." --max-completion-tokens 600 --temperature 0.2
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - convenience script
    raise SystemExit(
        "openai package is required for this smoke test (pip install openai)."
    ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly exercise client.chat.completions.create to inspect responses."
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name to query (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--system",
        default="You are a concise assistant used for smoke testing.",
        help="System prompt text.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with a short greeting so we can confirm the client works.",
        help="User prompt text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=256,
        help="Cap on completion tokens (default: 256).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is required.")

    client = OpenAI(api_key=api_key)

    logging.info("Sending test chat completion request to model %s", args.model)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ],
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    choice = response.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)

    logging.info("finish_reason: %s", finish_reason)

    if getattr(message, "refusal", None):
        logging.warning("refusal: %s", message.refusal)

    content = (message.content or "").strip()
    if content:
        logging.info("content:\n%s", content)
    else:
        logging.warning("No content returned by the model.")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
