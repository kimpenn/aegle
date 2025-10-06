"""Utilities and CLI helpers for LLM-assisted cluster annotation."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Optional dependency – only needed when annotations are enabled
    from openai import OpenAI
except ImportError:  # pragma: no cover - clearer error when invoked without dependency
    OpenAI = None  # type: ignore

try:  # Optional: load environment variables from .env if python-dotenv is available
    from dotenv import load_dotenv

    # Load .env from repo root and module directory if present
    load_dotenv()  # current working directory / parents
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover - silently ignore if dependency missing
    pass

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert in single-cell analysis and tissue biology. "
    "Provide detailed, accurate cell type annotations based on marker expression patterns."
)
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4000

DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "You are an expert reviewer of Phenocycler/CODEX single-cell experiments. "
    "Provide concise, evidence-based conclusions about the biological signal present in the run."
)


def load_json_file(filepath: str | Path) -> Dict[str, Any]:
    """Load JSON data from a file, returning an empty dict on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logging.error("File %s not found.", filepath)
    except json.JSONDecodeError:
        logging.error("Invalid JSON in %s", filepath)
    return {}


def create_prompt(prior_knowledge: Dict[str, Any], cluster_data: Dict[str, Any]) -> str:
    """Create the LLM prompt describing prior knowledge and observed clusters."""

    n_clusters = len(cluster_data)
    prompt = f"""You are an expert in single-cell analysis and histopathology. I have performed phenocycler scanning on fallopian tube tissue samples, followed by cell segmentation and clustering analysis. This resulted in {n_clusters} distinct cell clusters, each characterized by specific marker expression patterns.

**Task**: Annotate each cluster with the most likely cell type(s) based on the marker expression and provided prior knowledge.

**Prior Knowledge - Expected Cell Types and Markers in Uterus Tissue**:
```json
{json.dumps(prior_knowledge, indent=2)}
```

**Cluster Data - Marker Summary for Each Cluster**:
```json
{json.dumps(cluster_data, indent=2)}
```

**Instructions**:
1. For each cluster (1-{n_clusters}), analyse the marker expression pattern.
2. Cross-reference with the provided prior knowledge of expected cell types and their canonical markers.
3. Assign the most likely cell type annotation(s) for each cluster.
4. Each cluster may be:
   - **Single cell type**: Dominated by one clear cell type.
   - **Mixed cell type**: Contains two distinct cell types.
   - **Undetermined**: Mixed population that cannot be clearly classified.

**Output Format**:
For each cluster, provide:
- **Cluster ID**: [Number]
- **Predicted Cell Type(s)**: [Cell type name(s)]
- **Confidence Level**: High/Medium/Low
- **Key Supporting Markers**: [List of 3-5 most relevant markers]
- **Reasoning**: [Brief explanation of the annotation decision]
- **Alternative Possibilities**: [Other potential cell types if confidence is not high]

Please analyse all {n_clusters} clusters systematically."""
    return prompt


def query_llm(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Query the LLM with the annotation prompt."""

    if OpenAI is None:  # pragma: no cover - trigger informative error for missing dependency
        raise RuntimeError(
            "openai package is required to run LLM annotations. Install it or disable the step."
        )

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # pragma: no cover - caller decides how to handle failure
        logging.error("Error querying LLM: %s", exc)
        return ""


def annotate_clusters(
    prior_knowledge: Dict[str, Any],
    cluster_data: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Return LLM-generated annotations given prior knowledge and cluster summaries."""

    if not prior_knowledge:
        raise ValueError("prior_knowledge dictionary is empty")
    if not cluster_data:
        raise ValueError("cluster_data dictionary is empty")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    prompt = create_prompt(prior_knowledge, cluster_data)
    return query_llm(
        prompt,
        api_key,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_summary_prompt(
    prior_knowledge: Dict[str, Any],
    cluster_data: Dict[str, Any],
    annotation_text: str,
) -> str:
    """Create the LLM prompt asking for an overall interpretation summary."""

    prompt = (
        "You are assisting with quality control of a Phenocycler (CODEX) experiment on fallopian tube tissue.\n"
        "Review the following information and respond in English with a structured assessment.\n\n"
        "Prior knowledge (expected cell types and markers):\n"
        f"```json\n{json.dumps(prior_knowledge, indent=2)}\n```\n\n"
        "Cluster summaries (top markers and sizes):\n"
        f"```json\n{json.dumps(cluster_data, indent=2)}\n```\n\n"
        "Preliminary cluster annotations (verbatim output from an earlier step):\n"
        f"""{annotation_text}\n\n"""
        "Tasks:\n"
        "1. Determine whether the expected biological signal is present in this experiment.\n"
        "2. Highlight supporting evidence (clusters, markers, or spatial hints).\n"
        "3. Flag any missing or ambiguous signals that require follow-up.\n"
        "4. Provide recommendations or next steps.\n\n"
        "Output format:\n"
        "Summary:\n- <two to three sentence overview>\n"
        "Evidence:\n- bullet list linking clusters to biological expectations\n"
        "MissingSignals:\n- bullet list (use 'None' if nothing is missing)\n"
        "Confidence: <High|Medium|Low> – short justification\n"
        "Recommendations:\n- bullet list of follow-up actions\n"
        "FinalConclusion: <Yes|No> – succinct statement on whether the run shows the expected biological signal."
    )
    return prompt


def summarize_annotation(
    prior_knowledge: Dict[str, Any],
    cluster_data: Dict[str, Any],
    annotation_text: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    system_prompt: str = DEFAULT_SUMMARY_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Return a high-level summary describing biological signal interpretation."""

    if not annotation_text:
        raise ValueError("annotation_text is empty")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    prompt = create_summary_prompt(prior_knowledge, cluster_data, annotation_text)
    return query_llm(
        prompt,
        api_key,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def save_results(results: str, output_file: str | Path) -> None:
    """Persist annotation results to disk."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(results)
    logging.info("Results saved to %s", output_path)


def _cli() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Cell type annotation using an LLM")
    parser.add_argument("--prior", type=str, required=True, help="Path to prior knowledge JSON file")
    parser.add_argument("--cluster", type=str, required=True, help="Path to cluster data JSON file")
    parser.add_argument("--output", type=str, default="cell_annotation_results.txt", help="Output result file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Response token limit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    prior = load_json_file(args.prior)
    cluster = load_json_file(args.cluster)
    if not prior or not cluster:
        raise SystemExit("Missing prior knowledge or cluster summary input")

    try:
        results = annotate_clusters(
            prior,
            cluster,
            model=args.model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        logging.error("Annotation failed: %s", exc)
        raise SystemExit(1) from exc

    if not results:
        logging.error("No annotation result returned from the LLM")
        raise SystemExit(1)

    save_results(results, args.output)
    print(results)


if __name__ == "__main__":  # pragma: no cover - script usage
    _cli()
