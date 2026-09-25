#!/usr/bin/env python3
"""Apply a candidate semantic model to every text item in a DCLX or PDF."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from docling_nlp.andromeda_doclang import DocLangXDocument
from tabulate import tabulate

from docling_nlp.nlp_utils import init_nlp_model

LOG = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path("artifacts/semantic/fst_semantic.bin")
DEFAULT_DOCUMENT_DIRECTORY = Path("artifacts/semantic/documents")


def text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def docling_converter() -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires Docling; run: uv sync --extra docling"
        ) from exc

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = False
    options.generate_page_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_pdf(source: Path, destination: Path) -> Path:
    LOG.info("Converting PDF to DCLX: %s", source)
    converted = docling_converter().convert(source).document
    destination.parent.mkdir(parents=True, exist_ok=True)
    converted.save_as_doclang_archive(destination)
    return destination


def read_dclx_text_items(path: Path) -> list[tuple[str, str]]:
    document = DocLangXDocument()
    if not document.read(str(path)):
        raise RuntimeError(
            f"could not read DCLX document {path}: {document.last_error()}"
        )

    items = []
    for xpath, item, _, _ in document.iterate_items():
        if item["name"] in ["picture", "table"]:
            continue
        item_text = text(item["text"])
        if item_text:
            items.append((xpath, item_text))
    if not items:
        raise RuntimeError(f"DCLX contains no text items: {path}")
    return items


def load_model(model_path: Path) -> Any:
    model = init_nlp_model(
        f"semantic({model_path})",
        filters=["properties"],
    )
    try:
        classify(model, "Semantic model load check.", require_prediction=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "could not load the candidate semantic model; rebuild the native "
            "extension with: uv run python local_build.py"
        ) from exc
    return model


def classify(
    model: Any,
    item_text: str,
    *,
    require_prediction: bool = False,
) -> tuple[str, float]:
    result = model.apply_on_text(item_text)
    properties = result.get("properties")
    if not isinstance(properties, dict):
        if require_prediction:
            raise RuntimeError("semantic model returned no properties")
        return "no-prediction", 0.0
    headers = properties.get("headers", [])
    rows = properties.get("data", [])
    if not rows:
        if require_prediction:
            raise RuntimeError("semantic model returned no prediction")
        return "no-prediction", 0.0
    if "label" not in headers or "confidence" not in headers:
        raise RuntimeError("semantic model returned an invalid property table")
    row = rows[-1]
    return row[headers.index("label")], row[headers.index("confidence")]


def predict_items(model: Any, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    predictions = []
    for xpath, item_text in items:
        label, confidence = classify(model, item_text)
        predictions.append(
            {
                "xpath": xpath,
                "label": label,
                "confidence": confidence,
                "text": item_text,
            }
        )
    return predictions


def print_predictions(predictions: list[dict[str, Any]], output_format: str) -> None:
    if output_format == "jsonl":
        for prediction in predictions:
            print(json.dumps(prediction, ensure_ascii=False))
        return

    rows = [
        [row["xpath"], row["label"], row["confidence"], row["text"]]
        for row in predictions
    ]
    print(
        tabulate(
            rows,
            headers=["xpath", "label", "confidence", "text"],
            tablefmt="github",
            floatfmt=".4f",
            maxcolwidths=[None, None, None, 100],
        )
    )


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input .dclx or .pdf document.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Candidate semantic model.",
    )
    parser.add_argument(
        "--output-dclx",
        type=Path,
        help=(
            "DCLX destination for PDF input; defaults to "
            "artifacts/semantic/documents/<PDF-name>.dclx."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("table", "jsonl"),
        default="table",
        help="Prediction output format.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    source = args.input.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"input document does not exist: {source}")
    if source.suffix.lower() not in {".dclx", ".pdf"}:
        raise SystemExit("input document must have a .dclx or .pdf extension")

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"semantic model does not exist: {model_path}")
    model = load_model(model_path)

    if source.suffix.lower() == ".pdf":
        destination = args.output_dclx or (
            DEFAULT_DOCUMENT_DIRECTORY / source.with_suffix(".dclx").name
        )
        dclx_path = convert_pdf(source, destination.expanduser().resolve())
        LOG.info("DCLX document: %s", dclx_path)
    else:
        if args.output_dclx is not None:
            raise SystemExit("--output-dclx is only valid for PDF input")
        dclx_path = source

    items = read_dclx_text_items(dclx_path)
    LOG.info("Classifying %d text items with %s", len(items), model_path)
    print_predictions(predict_items(model, items), args.format)


if __name__ == "__main__":
    main()
