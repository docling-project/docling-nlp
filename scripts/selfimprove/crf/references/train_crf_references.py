#!/usr/bin/env python3
"""Prepare, train, and validate Docling's native reference CRF from YAML."""

from __future__ import annotations

import argparse
import difflib
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from docling_nlp.andromeda_nlp import nlp_model

from docling_nlp.nlp_utils import init_nlp_model

LOG = logging.getLogger(__name__)
REQUIRED_RECORD_FIELDS = {"text", "annotation"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a YAML mapping")
    for section in ("data", "training", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration mapping: {section}")
    for key in ("train_file", "validation_file"):
        if not config["data"].get(key):
            raise ValueError(f"missing configuration value: data.{key}")
    epochs = config["training"].get("epochs", 20)
    sigma = config["training"].get("gaussian_sigma", 2.0)
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("training.epochs must be a positive integer")
    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("training.gaussian_sigma must be positive")
    return config


def configured_path(value: str) -> Path:
    """Resolve configuration paths relative to the current working directory."""

    return Path(value).expanduser().resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not REQUIRED_RECORD_FIELDS.issubset(row):
                raise ValueError(f"invalid record at {path}:{line_number}")
            if not isinstance(row["text"], str) or not isinstance(
                row["annotation"], list
            ):
                raise ValueError(f"invalid text or annotation at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def validate_annotations(row: dict[str, Any]) -> None:
    text = row["text"]
    previous_end = 0
    for annotation in sorted(
        row["annotation"], key=lambda item: (item["start"], item["end"])
    ):
        try:
            label = annotation["label"]
            start = annotation["start"]
            end = annotation["end"]
        except (KeyError, TypeError) as exc:
            raise ValueError("each annotation needs label, start, and end") from exc
        if (
            not isinstance(label, str)
            or not label
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(f"invalid annotation: {annotation!r}")
        if start < previous_end or start < 0 or start >= end or end > len(text):
            raise ValueError(f"invalid or overlapping annotation: {annotation!r}")
        previous_end = end


def align_offsets(source: str, target: str) -> list[int]:
    """Map source character boundaries onto tokenizer-normalized text."""

    offsets = [0] * (len(source) + 1)
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    for (
        tag,
        source_start,
        source_end,
        target_start,
        target_end,
    ) in matcher.get_opcodes():
        if tag == "insert":
            offsets[source_start] = target_end
            continue

        source_width = source_end - source_start
        target_width = target_end - target_start
        for position in range(source_width + 1):
            offsets[source_start + position] = (
                target_start
                + (position * target_width + source_width // 2) // source_width
            )
    return offsets


def prepare_record(
    row: dict[str, Any], training_sample: bool, tokenizer: Any
) -> dict[str, Any]:
    """Tokenize one raw span record and attach native CRF token labels."""

    validate_annotations(row)
    prepared = tokenizer.apply_on_text(row["text"])
    word_tokens = prepared.get("word_tokens")
    if not isinstance(word_tokens, dict):
        raise RuntimeError("language tokenizer did not return word_tokens")
    headers = word_tokens.get("headers", [])
    data = word_tokens.get("data", [])
    for required in ("char_i", "char_j", "word"):
        if required not in headers:
            raise RuntimeError(f"language tokenizer omitted {required!r}")
    if "true-label" in headers:
        raise RuntimeError("language tokenizer unexpectedly returned true-label")
    headers.append("true-label")
    char_i_index = headers.index("char_i")
    char_j_index = headers.index("char_j")

    normalised_text = prepared["text"]
    aligned_offsets = align_offsets(row["text"], normalised_text)
    spans = []
    for annotation in row["annotation"]:
        start_char = aligned_offsets[annotation["start"]]
        end_char = aligned_offsets[annotation["end"]]
        start = len(normalised_text[:start_char].encode("utf-8"))
        end = len(normalised_text[:end_char].encode("utf-8"))
        spans.append(
            (start, end, annotation["label"].strip().lower().replace(" ", "_"))
        )

    for token in data:
        label = "null"
        for start, end, candidate in spans:
            if start <= token[char_i_index] and token[char_j_index] <= end:
                label = candidate
                break
        token.append(label)
    prepared["training-sample"] = training_sample
    prepared["source_file"] = row.get("source_file")
    prepared["source_index"] = row.get("source_index")
    return prepared


def write_prepared_dataset(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    tokenizer = init_nlp_model("language", filters=["properties", "word_tokens"])
    with output.open("w", encoding="utf-8") as stream:
        for is_training, rows in ((True, train_rows), (False, validation_rows)):
            for row in rows:
                prepared = prepare_record(row, is_training, tokenizer)
                stream.write(json.dumps(prepared, ensure_ascii=False) + "\n")


def native_crf_config(
    prepared_file: Path,
    model_file: Path,
    metrics_file: Path,
    training: dict[str, Any],
) -> dict[str, Any]:
    model_name = training.get("model_name", "references")
    return {
        "args": {
            "epoch": training.get("epochs", 20),
            "gaussian-sigma": float(training.get("gaussian_sigma", 2.0)),
        },
        "files": {
            "metrics-file": str(metrics_file),
            "model-file": str(model_file),
            "test-file": str(prepared_file),
            "train-file": str(prepared_file),
            "validate-file": str(prepared_file),
            "model-name": model_name,
        },
        "mode": "train",
        "model": "custom_crf(:)",
        "verbose": bool(training.get("verbose", False)),
    }


def train(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    train_file = configured_path(config["data"]["train_file"])
    validation_file = configured_path(config["data"]["validation_file"])
    output_dir = configured_path(
        config["output"].get("directory", "artifacts/crf-reference-model")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_file = output_dir / config["output"].get(
        "prepared_file", "references.annot.jsonl"
    )
    model_file = output_dir / config["output"].get("model_file", "crf_reference.bin")
    metrics_file = output_dir / config["output"].get(
        "metrics_file", "validation_metrics.txt"
    )

    train_rows = load_jsonl(train_file)
    validation_rows = load_jsonl(validation_file)
    LOG.info(
        "Preparing %d training and %d validation references",
        len(train_rows),
        len(validation_rows),
    )
    write_prepared_dataset(train_rows, validation_rows, prepared_file)

    model = nlp_model()
    model.set_loglevel(config["training"].get("loglevel", "WARNING"))
    native_config = native_crf_config(
        prepared_file, model_file, metrics_file, config["training"]
    )
    LOG.info("Training CRF model at %s", model_file)
    model.train(native_config)
    LOG.info("Evaluating held-out validation records")
    model.evaluate(native_config)
    if not model_file.is_file():
        raise RuntimeError(f"CRF trainer did not create model: {model_file}")
    if not metrics_file.is_file():
        raise RuntimeError(f"CRF evaluator did not create metrics: {metrics_file}")
    return prepared_file, model_file, metrics_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        default=argparse.SUPPRESS,
        help="YAML training configuration.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    prepared, model, metrics = train(load_config(args.config))
    LOG.info("Prepared data: %s", prepared)
    LOG.info("Model: %s", model)
    LOG.info("Validation metrics: %s", metrics)
    print(metrics.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
