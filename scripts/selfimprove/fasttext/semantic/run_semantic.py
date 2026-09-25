#!/usr/bin/env python3
"""Classify text with a candidate Docling semantic model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from docling_nlp.nlp_utils import init_nlp_model

DEFAULT_MODEL_PATH = Path("artifacts/semantic/fst_semantic.bin")


def load_model(model_path: Path) -> Any:
    return init_nlp_model(
        f"semantic({model_path})",
        filters=["properties"],
    )


def classify(model: Any, text: str) -> dict[str, Any]:
    result = model.apply_on_text(text)
    properties = result.get("properties")
    if not isinstance(properties, dict):
        raise RuntimeError("semantic model returned no properties")
    headers = properties.get("headers", [])
    rows = properties.get("data", [])
    if "label" not in headers or "confidence" not in headers or not rows:
        raise RuntimeError("semantic model returned an invalid property table")
    label_index = headers.index("label")
    confidence_index = headers.index("confidence")
    row = rows[-1]
    return {
        "label": row[label_index],
        "confidence": row[confidence_index],
        "text": text,
    }


def input_texts(text: str | None, stream: Iterable[str]) -> list[str]:
    if text is not None:
        return [text]
    return [line.rstrip("\n") for line in stream if line.strip()]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "text", nargs="?", help="Text to classify; reads stdin when omitted."
    )
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"semantic model does not exist: {model_path}")
    texts = input_texts(args.text, sys.stdin)
    if not texts:
        raise SystemExit(
            "provide text as an argument or newline-delimited text on stdin"
        )

    model = load_model(model_path)
    for text in texts:
        print(json.dumps(classify(model, text), ensure_ascii=False))


if __name__ == "__main__":
    main()
