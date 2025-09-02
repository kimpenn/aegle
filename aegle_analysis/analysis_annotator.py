#!/usr/bin/env python3
"""
Cell Type Annotation Script - Version 2 (Dynamic Cluster Count)
Uses LLM API to annotate cell clusters based on marker expression and prior knowledge.
"""

import json
import os
import argparse
import logging
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File {filepath} not found.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in {filepath}")
        return {}


def create_prompt_v1(prior_knowledge: Dict, cluster_data: Dict) -> str:
    """Create the annotation prompt with prior knowledge and dynamic cluster count."""

    n_clusters = len(cluster_data)
    prompt = f"""You are an expert in single-cell analysis and histopathology. I have performed phenocycler scanning on uterus tissue samples, followed by cell segmentation and clustering analysis. This resulted in {n_clusters} distinct cell clusters, each characterized by specific marker expression patterns.

**Task**: Annotate each cluster with the most likely cell type(s) based on the marker expression and provided prior knowledge.

**Prior Knowledge - Expected Cell Types and Markers in Uterus Tissue**:
```json
{json.dumps(prior_knowledge, indent=2)}
```

**Cluster Data - Top Markers for Each Cluster**:
```json
{json.dumps(cluster_data, indent=2)}
```

**Instructions**:
1. For each cluster (1-{n_clusters}), analyze the marker expression pattern
2. Cross-reference with the provided prior knowledge of expected cell types and their canonical markers
3. Assign the most likely cell type annotation(s) for each cluster
4. Each cluster may be:
   - **Single cell type**: Dominated by one clear cell type
   - **Mixed cell type**: Contains two distinct cell types
   - **Undetermined**: Mixed population that cannot be clearly classified

**Output Format**:
For each cluster, provide:
- **Cluster ID**: [Number]
- **Predicted Cell Type(s)**: [Cell type name(s)]
- **Confidence Level**: High/Medium/Low
- **Key Supporting Markers**: [List of 3-5 most relevant markers]
- **Reasoning**: [Brief explanation of the annotation decision]
- **Alternative Possibilities**: [Other potential cell types if confidence is not high]

Please analyze all {n_clusters} clusters systematically."""
    return prompt


def query_llm(prompt: str, api_key: str, model: str = "gpt-4o") -> str:
    """Query the LLM with the annotation prompt."""
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in single-cell analysis and tissue biology. Provide detailed, accurate cell type annotations based on marker expression patterns."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return ""


def save_results(results: str, output_file: str):
    """Save annotation results to file."""
    try:
        with open(output_file, 'w') as f:
            f.write(results)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Cell Type Annotation using LLM")
    parser.add_argument("--prior", type=str, required=True, help="Path to prior knowledge JSON file")
    parser.add_argument("--cluster", type=str, required=True, help="Path to cluster data JSON file")
    parser.add_argument("--output", type=str, default="cell_annotation_results_v1.txt", help="Output result file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    logging.basicConfig(level=logging.INFO)
    logging.info("Loading data files...")

    prior_knowledge = load_json_file(args.prior)
    cluster_data = load_json_file(args.cluster)

    if not prior_knowledge or not cluster_data:
        return

    logging.info("Creating annotation prompt...")
    prompt = create_prompt_v1(prior_knowledge, cluster_data)

    logging.info("Querying LLM...")
    results = query_llm(prompt, api_key, args.model)

    if results:
        logging.info("Annotation completed!")
        print("\n" + "=" * 50)
        print("RESULTS:")
        print("=" * 50)
        print(results)
        save_results(results, args.output)
    else:
        logging.error("No annotation result returned.")


if __name__ == "__main__":
    main()
