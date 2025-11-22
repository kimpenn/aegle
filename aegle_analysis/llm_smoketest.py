"""Minimal smoke test for the OpenAI chat completions client."""
"""
python 0-phenocycler-penntmc-pipeline/aegle_analysis/llm_smoketest.py --model gpt-5.1-2025-11-13 --prompt "Give me a two-sentence summary of the Phenocycler analysis pipeline." --max-completion-tokens 600 --temperature 0.2
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, List, Sequence

from aegle_analysis.analysis_annotator import DEFAULT_MODEL as DEFAULT_LLM_MODEL

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
        default=DEFAULT_LLM_MODEL,
        help=f"Model name to query (default: {DEFAULT_LLM_MODEL}).",
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


def _normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        fragments: List[str] = []
        for chunk in content:
            text = None
            if isinstance(chunk, dict):
                text = chunk.get("text")
            else:
                text = getattr(chunk, "text", None)
            if text:
                fragments.append(str(text))
        return "\n".join(fragments).strip()
    return str(content).strip()


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

    content = _normalize_message_content(getattr(message, "content", None))
    if content:
        logging.info("content:\n%s", content)
    else:
        logging.warning("No content returned by the model.")


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()
