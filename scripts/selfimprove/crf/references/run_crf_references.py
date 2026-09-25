#!/usr/bin/env python3
"""Classify the tokens in a reference string with a trained CRF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tabulate import tabulate

from docling_nlp.nlp_utils import init_nlp_model

CRF_TAG_PREFIX = "__CUSTOM_CRF__"
DEFAULT_MODEL_PATH = Path("artifacts/grobid-crf-references/model/crf_reference.bin")
DEFAULT_VALIDATION_PATH = Path("artifacts/grobid-crf-references/validation.jsonl")
BIBTEX_FIELDS = {
    "authors": "author",
    "conference": "booktitle",
    "date": "year",
    "issue": "number",
    "location": "address",
}


def load_model(model_path: Path) -> Any:
    """Load the language tokenizer and trained reference CRF."""

    return init_nlp_model(
        f"language;custom_crf(references:{model_path})",
        filters=["word_tokens"],
    )


def classify_tokens(model: Any, text: str) -> list[list[Any]]:
    """Return token text, offsets, and predicted labels for one string."""

    result = model.apply_on_text(text)
    word_tokens = result["word_tokens"]
    headers = word_tokens["headers"]

    required = ("char_i", "char_j", "original", "tag")
    missing = [header for header in required if header not in headers]
    if missing:
        raise RuntimeError(
            "CRF output omitted required token fields: " + ", ".join(missing)
        )

    indices = {header: headers.index(header) for header in required}
    rows = []
    for index, token in enumerate(word_tokens["data"]):
        tag = token[indices["tag"]]
        label = tag.removeprefix(CRF_TAG_PREFIX) if isinstance(tag, str) else str(tag)
        rows.append(
            [
                index,
                token[indices["original"]],
                token[indices["char_i"]],
                token[indices["char_j"]],
                label,
            ]
        )
    return rows


def prediction_to_bibtex(text: str, rows: list[list[Any]]) -> dict[str, str]:
    """Convert labelled token spans into JSON-serializable BibTeX fields."""

    spans: dict[str, tuple[int, int]] = {}
    for _, _, start, end, label in rows:
        if label == "null":
            continue
        field = BIBTEX_FIELDS.get(label, label)
        if field in spans:
            start = min(start, spans[field][0])
            end = max(end, spans[field][1])
        spans[field] = (start, end)

    return {
        field: text[start:end].strip()
        for field, (start, end) in spans.items()
    }


def load_validation_texts(path: Path) -> list[str]:
    """Read reference text values from a validation JSONL file."""

    texts = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"invalid validation record at {path}:{line_number}")
            texts.append(record["text"])
    if not texts:
        raise ValueError(f"validation dataset is empty: {path}")
    return texts


def print_prediction(text: str, rows: list[list[Any]]) -> None:
    """Print the source reference, token predictions, and BibTeX JSON."""

    print(f"Original text:\n{text}\n")
    print(
        tabulate(
            rows,
            headers=["index", "token", "character start", "character end", "label"],
            tablefmt="github",
        )
    )
    print()
    print(json.dumps(prediction_to_bibtex(text, rows), indent=2, ensure_ascii=False))


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained CRF .bin file.",
    )
    parser.add_argument("text", nargs="?", help="Reference text to classify.")
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Validation JSONL used when no reference text is supplied.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"CRF model does not exist: {model_path}")

    if args.text is None:
        validation_path = args.validation_file.expanduser().resolve()
        if not validation_path.is_file():
            raise SystemExit(f"validation dataset does not exist: {validation_path}")
        texts = load_validation_texts(validation_path)
    else:
        texts = [args.text]

    model = load_model(model_path)
    for index, text in enumerate(texts):
        if index:
            print("\n" + "=" * 80 + "\n")
        rows = classify_tokens(model, text)
        print_prediction(text, rows)


if __name__ == "__main__":
    main()
