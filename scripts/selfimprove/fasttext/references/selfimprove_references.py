#!/usr/bin/env python3
"""Bounded local-LLM self-improvement for FastText reference classification.

PDFs always use Docling's layout pipeline with table structure disabled and OCR opt-in.
Install the optional dependency first: uv sync --extra docling
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import random
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ElementTree
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

LABELS = {"meta-data", "reference", "text"}
TRAINABLE_LAYOUT_LABELS = {"text", "list_item"}
LOG = logging.getLogger(__name__)
REFERENCE_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited)\s*$",
    re.IGNORECASE,
)


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("docling").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def arguments() -> argparse.Namespace:
    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example:\n"
            "  uv run python scripts/selfimprove/fasttext/references/"
            "selfimprove_references.py --input-dir corpus --output-dir runs/references"
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, default=argparse.SUPPRESS,
        help="Root searched recursively for PDF, JSON, and JSONL inputs; omit for arXiv PDFs.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=argparse.SUPPRESS,
        help=(
            "Directory for DCLX documents, reviews, training data, models, and metrics. "
            "When omitted, creates scratch-references-YYYY-MM-DD-HH-MM-SS."
        ),
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of apply-review-train-evaluate iterations to run.",
    )
    parser.add_argument(
        "--max-files", type=int, default=25,
        help="Maximum number of discovered input files processed in this run.",
    )
    parser.add_argument(
        "--selection-seed", type=int, default=0,
        help="Seed used to deterministically rank files before selecting the batch.",
    )
    parser.add_argument(
        "--arxiv-batch-size", type=int, default=10,
        help="Number of random arXiv PDFs fetched initially and per later iteration.",
    )
    parser.add_argument(
        "--arxiv-seed", type=int, default=0,
        help="Seed used to reproducibly select arXiv API result pages and categories.",
    )
    parser.add_argument(
        "--arxiv-categories",
        default="cs.CL,cs.IR,cs.LG,stat.ML,math.ST,physics.geo-ph,q-bio.QM,econ.EM",
        help="Comma-separated arXiv categories used when --input-dir is omitted.",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Base URL of local Ollama; the script calls its /api/chat endpoint.",
    )
    parser.add_argument(
        "--llm-model", default="qwen3.6:27b",
        help="Installed Ollama model used to review uncertain candidates.",
    )
    parser.add_argument(
        "--llm-thinking", action="store_true",
        help="Enable Ollama reasoning; disabled by default for faster label reviews.",
    )
    parser.add_argument(
        "--ollama-timeout", type=float, default=60.0,
        help="Maximum seconds to wait for one Ollama review response.",
    )
    parser.add_argument(
        "--llm-context-window", type=int, default=8192,
        help="Maximum Ollama context tokens per isolated review request.",
    )
    parser.add_argument(
        "--llm-batch-size", type=int, default=32,
        help="Uncertain items classified in each Ollama request.",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.90,
        help="FastText confidence below which a candidate is sent to Ollama.",
    )
    parser.add_argument(
        "--review-limit", type=int, default=500,
        help="Maximum uncertain candidates reviewed in one iteration; use -1 to review all.",
    )
    parser.add_argument(
        "--validation-ratio", type=float, default=0.10,
        help="Deterministic fraction of approved labels reserved for evaluation.",
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Enable Docling OCR for scanned or image-only PDFs (disabled by default).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse DCLX and candidate artifacts, then continue after the latest iteration.",
    )
    parser.add_argument(
        "--compact-run", action="store_true",
        help="Migrate a retained legacy iteration to the compact layout, then exit (requires --resume).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Log each Ollama review prompt and raw response to stderr.",
    )
    parser.add_argument(
        "--autotune-duration", type=int, default=3600,
        help="FastText autotuning budget in seconds when --autotune is enabled.",
    )
    parser.add_argument(
        "--autotune-modelsize", default="100M",
        help="Target FastText model-size budget passed to autotuning.",
    )
    parser.add_argument(
        "--autotune", action="store_true",
        help="Enable native FastText autotuning (disabled by default for reliability).",
    )
    parser.add_argument(
        "--fasttext-retries", type=int, default=2,
        help="Number of retries after a FastText artifact/evaluation failure.",
    )
    parser.add_argument(
        "--ngram", type=int, default=0,
        help="FastText word n-gram order; zero disables word n-grams.",
    )
    args = parser.parse_args()
    if not hasattr(args, "output_dir"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = Path(f"scratch-references-{timestamp}")
    args.arxiv_mode = not hasattr(args, "input_dir")
    if args.arxiv_mode:
        args.input_dir = args.output_dir / "arxiv-pdfs"
    elif not args.input_dir.is_dir():
        parser.error(f"missing input directory: {args.input_dir}")
    if (args.iterations < 1 or args.review_limit == 0 or args.review_limit < -1
            or args.max_files < 1 or args.arxiv_batch_size < 1
            or args.fasttext_retries < 0 or args.llm_context_window < 1
            or args.llm_batch_size < 1):
        parser.error(
            "iterations, max-files, arxiv-batch-size, llm-context-window, and "
            "llm-batch-size must be positive; review-limit must be positive or -1; "
            "fasttext-retries cannot be negative"
        )
    if not 0 < args.min_confidence <= 1 or not 0 < args.validation_ratio < 1:
        parser.error("confidence and validation ratio must be in (0, 1)")
    return args


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_dataset_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write an inspection-friendly mirror of a derived FastText dataset."""

    columns = [
        "candidate_id", "source_path", "split", "label", "applied_label",
        "confidence", "label_source", "text",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row.get("candidate-id"),
                    "source_path": row.get("source-path"),
                    "split": "train" if row.get("training-sample") else "validation",
                    "label": row.get("label"),
                    "applied_label": row.get("applied-label"),
                    "confidence": row.get("confidence"),
                    "label_source": row.get("label-source"),
                    "text": row.get("text"),
                }
            )


def write_persistent_datasets(
    output_dir: Path,
    iteration: int,
    rows: list[dict[str, Any]],
    model_expression: str,
) -> None:
    """Overwrite compact accumulated train/validation CSVs using the latest model."""

    LOG.info("Scoring %d accumulated rows with the latest model for persistent CSVs", len(rows))
    evaluated = apply(model_expression, rows)
    confidence = {
        item["candidate-id"]: item.get("prediction_confidence")
        for item in evaluated
    }
    fields = ["label", "conf", "LLM verified", "iteration", "filename", "xpath", "text"]
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)
    for split, selected in (
        ("train", [row for row in rows if row["training-sample"]]),
        ("validation", [row for row in rows if not row["training-sample"]]),
    ):
        with (data_dir / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                writer.writerow({
                    "label": row["label"],
                    "conf": confidence.get(row["candidate-id"]),
                    "LLM verified": "yes" if row.get("label-source") == "llm" else "no",
                    "iteration": (
                        row["introduced_iteration"]
                        if row.get("introduced_iteration") is not None else iteration
                    ),
                    "filename": row["source-path"],
                    "xpath": row.get("xpath", ""),
                    "text": row["text"],
                })


METRIC_LABELS = ("meta-data", "reference", "text")


def metric_label(label: str) -> str:
    """Make a classifier label safe and consistent in CSV column names."""

    return label.replace("-", "_")


METRICS_FIELDS = [
    "iteration", "candidates", "uncertain", "reviewed", "training_rows",
]
METRICS_FIELDS += [
    f"{split}_{metric_label(label)}"
    for split in ("train", "validation", "llm_validation")
    for label in METRIC_LABELS
]
METRICS_FIELDS += [
    f"{model}_{metric_label(label)}_{metric}"
    for model in ("baseline", "corrected")
    for label in METRIC_LABELS
    for metric in ("precision", "recall", "f1")
]
METRICS_FIELDS.insert(METRICS_FIELDS.index("baseline_meta_data_precision"), "baseline_accuracy")
METRICS_FIELDS.insert(METRICS_FIELDS.index("corrected_meta_data_precision"), "corrected_accuracy")


def metrics_row(summary: dict[str, Any], comparison: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten one iteration's retained counts and held-out evaluation results."""

    counts = summary.get("training_label_counts", {})
    train_counts = counts.get("train", {})
    validation_counts = counts.get("validation", {})
    held_out = summary.get("validation_labels", {})
    row: dict[str, Any] = {
        "iteration": summary.get("iteration"),
        "candidates": summary.get("candidates"),
        "uncertain": summary.get("uncertain"),
        "reviewed": summary.get("reviewed"),
        "training_rows": sum(train_counts.values()) + sum(validation_counts.values()),
    }
    for label in METRIC_LABELS:
        key = metric_label(label)
        row[f"train_{key}"] = train_counts.get(label)
        row[f"validation_{key}"] = validation_counts.get(label)
        row[f"llm_validation_{key}"] = held_out.get(label)
    for model in ("baseline", "corrected"):
        evaluation = (comparison or {}).get(model, {})
        metrics = evaluation.get("metrics", {})
        row[f"{model}_accuracy"] = evaluation.get("accuracy")
        for label in METRIC_LABELS:
            for metric in ("precision", "recall", "f1"):
                row[f"{model}_{metric_label(label)}_{metric}"] = metrics.get(label, {}).get(metric)
    return row


def write_iteration_metrics(
    output_dir: Path, summary: dict[str, Any], comparison: dict[str, Any] | None
) -> Path:
    """Keep a tiny, plot-ready metrics history instead of full iteration artifacts."""

    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    iteration_file = metrics_dir / f"iteration-{int(summary['iteration']):03d}.json"
    write_json(iteration_file, {"summary": summary, "comparison": comparison or {}})
    history_file = metrics_dir / "iterations.csv"
    existing: dict[str, dict[str, str]] = {}
    if history_file.exists():
        with history_file.open(encoding="utf-8", newline="") as stream:
            existing = {row["iteration"]: row for row in csv.DictReader(stream)}
    row = metrics_row(summary, comparison)
    existing[str(summary["iteration"])] = {
        field: "" if row.get(field) is None else str(row.get(field))
        for field in METRICS_FIELDS
    }
    with history_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        for _, value in sorted(existing.items(), key=lambda item: int(item[0])):
            writer.writerow(value)
    return iteration_file


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def xml_compatible(value: str | None) -> str | None:
    """Remove characters forbidden by XML 1.0 while retaining document text."""

    if value is None:
        return None
    cleaned = "".join(
        character
        for character in value
        if character in {"\t", "\n", "\r"} or ord(character) >= 0x20
    )
    return text(cleaned)


def json_texts(path: Path) -> list[tuple[str, str | None]]:
    loaded = read_jsonl(path) if path.suffix == ".jsonl" else json.loads(
        path.read_text(encoding="utf-8")
    )
    roots = loaded if isinstance(loaded, list) else [loaded]
    output: list[tuple[str, str | None]] = []

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        item_text = text(value.get("text"))
        if item_text:
            output.append((item_text, text(value.get("type"))))
        for key in ("main-text", "texts"):
            for child in value.get(key, []) if isinstance(value.get(key), list) else []:
                visit(child)

    for root in roots:
        visit(root)
    return output


def docling_converter(ocr: bool) -> Any:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires Docling; run: uv sync --extra docling"
        ) from exc
    options = PdfPipelineOptions()
    options.do_table_structure = False
    options.do_ocr = ocr
    LOG.debug(
        "Initializing Docling layout pipeline (OCR=%s, table structure disabled)",
        ocr,
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def select_sources(
    args: argparse.Namespace,
    batch_number: int = 0,
    selected_file: Path | None = None,
) -> list[Path]:
    """Inventory inputs and select one deterministic, non-overlapping batch."""

    LOG.info("Listing PDF, JSON, and JSONL inputs under %s", args.input_dir)
    paths = sorted(
        path
        for path in args.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".json", ".jsonl"}
    )
    if not paths:
        raise RuntimeError("no PDF, JSON, or JSONL files found")
    LOG.info("Found %d supported input files", len(paths))

    inventory = [
        {
            "source_path": str(path.relative_to(args.input_dir)),
            "source_kind": path.suffix.lower().lstrip("."),
        }
        for path in paths
    ]
    write_json(args.output_dir / "input-files.json", inventory)
    write_json(
        args.output_dir / "pdf-files.json",
        [record for record in inventory if record["source_kind"] == "pdf"],
    )

    def rank(path: Path) -> str:
        relative = str(path.relative_to(args.input_dir))
        return digest(f"{args.selection_seed}\\0{relative}")

    first = batch_number * args.max_files
    selected = sorted(paths, key=rank)[first:first + args.max_files]
    selected_file = selected_file or args.output_dir / "selected-files.json"
    write_json(
        selected_file,
        [
            {
                "source_path": str(path.relative_to(args.input_dir)),
                "source_kind": path.suffix.lower().lstrip("."),
                "selection_rank": rank(path),
            }
            for path in selected
        ],
    )
    LOG.info(
        "Selected batch %d: %d files (%d-%d of %d; seed=%d)",
        batch_number + 1,
        len(selected),
        first + 1 if selected else 0,
        first + len(selected),
        len(paths),
        args.selection_seed,
    )
    return selected


def arxiv_manifest_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "arxiv-manifest.json"


def fetch_arxiv_batch(args: argparse.Namespace) -> list[Path]:
    """Download one reproducible, de-duplicated batch of arXiv PDFs."""

    manifest_path = arxiv_manifest_path(args)
    manifest = read_json(manifest_path) if manifest_path.exists() else []
    known_ids = {record["arxiv_id"] for record in manifest}
    batch_number = len(manifest) // args.arxiv_batch_size
    categories = [category.strip() for category in args.arxiv_categories.split(",") if category.strip()]
    if not categories:
        raise RuntimeError("arxiv-categories must contain at least one category")
    randomizer = random.Random(f"{args.arxiv_seed}:{batch_number}")
    args.input_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    attempts = 0
    while len(downloaded) < args.arxiv_batch_size and attempts < args.arxiv_batch_size * 4:
        attempts += 1
        category = randomizer.choice(categories)
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"cat:{category}",
                "start": randomizer.randrange(0, 10_000),
                "max_results": 10,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            headers={"User-Agent": "docling-nlp-reference-selfimprove/1.0"},
            timeout=args.ollama_timeout,
        )
        response.raise_for_status()
        try:
            feed = ElementTree.fromstring(response.content)
        except ElementTree.ParseError as exc:
            raise RuntimeError("arXiv returned invalid Atom XML") from exc
        namespace = "{http://www.w3.org/2005/Atom}"
        entries = feed.findall(f"{namespace}entry")
        if not entries:
            continue
        randomizer.shuffle(entries)
        for entry in entries:
            identifier = text(entry.findtext(f"{namespace}id"))
            if not identifier:
                continue
            arxiv_id = identifier.split("/abs/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            if arxiv_id in known_ids:
                continue
            destination = args.input_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            LOG.info("Downloading arXiv PDF %d/%d: %s", len(downloaded) + 1, args.arxiv_batch_size, arxiv_id)
            pdf_response = requests.get(
                f"https://arxiv.org/pdf/{arxiv_id}",
                headers={"User-Agent": "docling-nlp-reference-selfimprove/1.0"},
                timeout=args.ollama_timeout,
            )
            if not pdf_response.ok:
                LOG.warning(
                    "Skipping arXiv PDF %s: HTTP %s",
                    arxiv_id,
                    pdf_response.status_code,
                )
                continue
            if not pdf_response.content.startswith(b"%PDF"):
                raise RuntimeError(f"arXiv did not return a PDF for {arxiv_id}")
            destination.write_bytes(pdf_response.content)
            manifest.append(
                {
                    "arxiv_id": arxiv_id,
                    "category": category,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                    "path": str(destination.relative_to(args.output_dir)),
                    "sha256": hashlib.sha256(pdf_response.content).hexdigest(),
                }
            )
            known_ids.add(arxiv_id)
            downloaded.append(destination)
            if len(downloaded) == args.arxiv_batch_size:
                break
    if len(downloaded) != args.arxiv_batch_size:
        raise RuntimeError(
            f"arXiv supplied only {len(downloaded)} new PDFs after {attempts} attempts"
        )
    write_json(manifest_path, manifest)
    LOG.info("Downloaded %d arXiv PDFs into %s", len(downloaded), args.input_dir)
    return downloaded


def dclx_path(args: argparse.Namespace, source: Path) -> Path:
    """Return the DCLX cache path while preserving the input directory layout."""

    relative = source.relative_to(args.input_dir)
    return (args.output_dir / "documents" / relative).with_suffix(".dclx")


def docling_text_items(document: Any) -> list[tuple[str, int | None, str | None]]:
    """Extract text/list items outside Docling picture and table containers."""

    items = []
    containers: list[tuple[int, str]] = []
    excluded_containers = {"PictureItem", "FigureItem", "TableItem"}
    for item, level in document.iterate_items():
        while containers and containers[-1][0] >= level:
            containers.pop()
        item_kind = type(item).__name__
        in_excluded_container = any(
            ancestor_kind in excluded_containers
            for _, ancestor_kind in containers
        )
        if item_kind in excluded_containers:
            containers.append((level, item_kind))
            continue
        if item_kind not in {"TextItem", "ListItem"} or in_excluded_container:
            continue
        item_text = xml_compatible(text(getattr(item, "text", None)))
        if not item_text:
            continue
        provenance = getattr(item, "prov", None) or []
        page = getattr(provenance[0], "page_no", None) if provenance else None
        label = str(getattr(item, "label", "")) or None
        items.append((item_text, page, label))
    return items


def write_dclx(
    path: Path,
    source_path: str,
    items: list[tuple[str, int | None, str | None]],
) -> None:
    """Store Docling layout text and its provenance in a project-native DCLX archive."""

    from docling_nlp.andromeda_doclang import DocLangXDocument

    root = ElementTree.Element(
        "doclang",
        {
            "version": "0.7",
            "source-path": source_path,
            "source-kind": "pdf",
            "selfimprove-schema": "2",
        },
    )
    for position, (item_text, page, layout) in enumerate(items):
        attributes = {"position": str(position)}
        if page is not None:
            attributes["page-no"] = str(page)
        if layout:
            attributes["layout-label"] = layout
        element = ElementTree.SubElement(root, "text", attributes)
        element.text = item_text
    dclx = DocLangXDocument()
    if not dclx.read_xml(ElementTree.tostring(root, encoding="unicode")):
        raise RuntimeError(
            f"could not create DCLX document for {source_path}: {dclx.last_error()}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not dclx.write(str(path)):
        raise RuntimeError(f"could not write DCLX document: {path}")
    LOG.debug("Wrote DCLX with %d text items: %s", len(items), path)


def dclx_text_items(path: Path) -> list[tuple[str, int | None, str | None]]:
    """Read direct candidate items from DCLX using its native iterator."""

    from docling_nlp.andromeda_doclang import DocLangXDocument

    dclx = DocLangXDocument()
    if not dclx.read(str(path)):
        raise RuntimeError(f"could not read DCLX document: {path}")
    try:
        root = ElementTree.fromstring(dclx.xml())
    except ElementTree.ParseError as exc:
        raise RuntimeError(f"DCLX contains invalid DocLang XML: {path}") from exc
    if root.get("selfimprove-schema") != "2":
        raise ValueError(f"legacy flattened DCLX cache: {path}")
    items = []
    for _, item, _, _ in dclx.iterate_items():
        if item["name"] != "text":
            continue
        try:
            element = ElementTree.fromstring(item["xml"])
        except ElementTree.ParseError as exc:
            raise RuntimeError(f"DCLX contains invalid DocLang XML: {path}") from exc
        item_text = text(item["text"])
        if not item_text:
            continue
        page_value = element.get("page-no")
        try:
            page = int(page_value) if page_value is not None else None
        except ValueError:
            page = None
        items.append((item_text, page, text(element.get("layout-label"))))
    return items


def _convert_source_batch_to_dclx(
    args: argparse.Namespace, paths: list[Path]
) -> list[dict[str, Any]]:
    converter = None
    output: list[dict[str, Any]] = []
    for index, path in enumerate(
        tqdm(paths, desc="Converting documents", unit="file", file=sys.stderr, dynamic_ncols=True),
        start=1,
    ):
        relative = str(path.relative_to(args.input_dir))
        if path.suffix.lower() == ".pdf":
            cache_path = dclx_path(args, path)
            if cache_path.exists():
                try:
                    LOG.debug("Reusing DCLX %d/%d: %s", index, len(paths), relative)
                    items = dclx_text_items(cache_path)
                except ValueError as exc:
                    LOG.info("Recreating legacy DCLX %d/%d: %s", index, len(paths), relative)
                    LOG.debug("%s", exc)
                    converter = converter or docling_converter(args.ocr)
                    document = converter.convert(path).document
                    items = docling_text_items(document)
                    write_dclx(cache_path, relative, items)
            else:
                LOG.debug("Converting PDF %d/%d: %s", index, len(paths), relative)
                converter = converter or docling_converter(args.ocr)
                document = converter.convert(path).document
                items = docling_text_items(document)
                write_dclx(cache_path, relative, items)
            kind = "pdf"
        else:
            LOG.debug("Reading JSON input %d/%d: %s", index, len(paths), relative)
            items = [(item_text, None, label) for item_text, label in json_texts(path)]
            kind = "json"
        in_references = False
        for position, (item_text, page, layout) in enumerate(items):
            if REFERENCE_HEADING.match(item_text):
                in_references = True
            output.append(
                {
                    "candidate_id": digest(f"{kind}\\0{relative}\\0{position}\\0{item_text}"),
                    "source_path": relative,
                    "source_kind": kind,
                    "position": position,
                    "xpath": f"/doclang[1]/text[{position + 1}]",
                    "page_no": page,
                    "layout_label": layout,
                    "in_reference_section": in_references,
                    "text": item_text,
                }
            )
    LOG.info("Prepared %d text candidates from %d input files", len(output), len(paths))
    return output


def trainable_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only non-numeric Docling text/list items for reference training."""

    return [
        item for item in items
        if item["layout_label"] in TRAINABLE_LAYOUT_LABELS and not item["numval_only"]
    ]


def convert_documents_to_dclx(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create/reuse initial DCLX documents and their SEMANTIC candidates."""

    candidates_file = args.output_dir / "reference-candidates.json"
    semantic_file = args.output_dir / "semantic-predictions.json"
    if candidates_file.exists():
        if not args.resume:
            raise RuntimeError(
                f"{candidates_file} exists; use --resume or a new output directory"
            )
        semantic_predictions = (
            read_json(semantic_file) if semantic_file.exists() else read_json(candidates_file)
        )
        if any("numval_only" not in item for item in semantic_predictions):
            semantic_predictions = annotate_numval(semantic_predictions)
            write_json(semantic_file, semantic_predictions)
        candidates = trainable_candidates(semantic_predictions)
        write_json(candidates_file, candidates)
        LOG.info("Reusing %d accumulated candidates from %s", len(candidates), candidates_file)
        return semantic_predictions, candidates

    selected_sources = fetch_arxiv_batch(args) if args.arxiv_mode else select_sources(args)
    extracted = _convert_source_batch_to_dclx(args, selected_sources)
    semantic_predictions = annotate_numval(apply("semantic", extracted))
    for item in semantic_predictions:
        item["introduced_iteration"] = 0
    candidates = trainable_candidates(semantic_predictions)
    write_json(semantic_file, semantic_predictions)
    write_json(candidates_file, candidates)
    LOG.info("Created %d initial trainable candidates", len(candidates))
    return semantic_predictions, candidates


def property_prediction(result: dict[str, Any]) -> tuple[str | None, float | None]:
    table = result.get("properties", {})
    headers, rows = table.get("headers", []), table.get("data", [])
    try:
        label_index = headers.index("label")
        confidence_index = headers.index("confidence")
    except ValueError:
        return None, None
    for row in rows:
        if len(row) > max(label_index, confidence_index):
            label, confidence = row[label_index], row[confidence_index]
            if isinstance(label, str) and isinstance(confidence, (int, float)):
                return label.lower(), float(confidence)
    return None, None


def deterministic_text_fragment(value: str) -> bool:
    """Identify fragments that cannot be bibliographic or author metadata."""

    compact = "".join(value.split())
    return len(compact) <= 1 or not any(character.isalnum() for character in compact)


def apply(expression: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from docling_nlp.nlp_utils import init_nlp_model

    LOG.info("Applying model %r to %d candidates", expression, len(candidates))
    model = None
    output = []
    for item in candidates:
        if deterministic_text_fragment(item["text"]):
            output.append({
                **item,
                "prediction_label": "text",
                "prediction_confidence": 1.0,
                "error": None,
                "deterministic_text": True,
            })
            continue
        try:
            model = model or init_nlp_model(expression, filters=["properties"])
            label, confidence = property_prediction(model.apply_on_text(item["text"]))
            output.append({**item, "prediction_label": label,
                           "prediction_confidence": confidence, "error": None,
                           "deterministic_text": False})
        except Exception as exc:
            output.append({**item, "prediction_label": None,
                           "prediction_confidence": None, "error": str(exc),
                           "deterministic_text": False})
    failures = sum(item["error"] is not None for item in output)
    LOG.info("Model application complete (%d failures)", failures)
    return output


def numval_coverage(result: dict[str, Any], value: str) -> float:
    """Return the fraction of meaningful characters covered by NUMVAL spans."""

    table = result.get("instances", {})
    headers, rows = table.get("headers", []), table.get("data", [])
    try:
        type_index = headers.index("type")
        char_i_index = headers.index("char_i")
        char_j_index = headers.index("char_j")
    except ValueError:
        return 0.0
    covered: set[int] = set()
    for row in rows:
        if len(row) <= max(type_index, char_i_index, char_j_index):
            continue
        if str(row[type_index]).lower() != "numval":
            continue
        try:
            start, end = int(row[char_i_index]), int(row[char_j_index])
        except (TypeError, ValueError):
            continue
        covered.update(range(max(0, start), min(len(value), end)))
    meaningful = {index for index, character in enumerate(value) if character.isalnum()}
    return len(meaningful & covered) / len(meaningful) if meaningful else 0.0


def annotate_numval(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark values wholly recognised by NUMVAL; failures are retained."""

    from docling_nlp.nlp_utils import init_nlp_model

    LOG.info("Applying model 'numval' to %d candidates", len(candidates))
    model = init_nlp_model("numval")
    output = []
    for item in candidates:
        try:
            coverage = numval_coverage(model.apply_on_text(item["text"]), item["text"])
            output.append({
                **item,
                "numval_coverage": coverage,
                "numval_only": coverage == 1.0,
                "numval_error": None,
            })
        except Exception as exc:
            output.append({
                **item,
                "numval_coverage": 0.0,
                "numval_only": False,
                "numval_error": str(exc),
            })
    rejected = [item for item in output if item["numval_only"]]
    by_layout = Counter(str(item.get("layout_label")) for item in rejected)
    by_source = Counter(str(item.get("source_path")) for item in rejected)
    LOG.info("NUMVAL excluded %d numeric-only candidates by layout: %s", len(rejected), dict(by_layout))
    LOG.info("NUMVAL numeric-only sources: %s", dict(by_source.most_common(10)))
    return output


def uncertain(item: dict[str, Any], minimum: float) -> bool:
    return (
        item["error"] is not None
        or item["prediction_label"] not in LABELS
        or item["prediction_confidence"] is None
        or item["prediction_confidence"] < minimum
    )


def local_review(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Classify one text item in a fresh, isolated Ollama context."""

    return local_reviews([item], args)[item["candidate_id"]]


def local_reviews(
    items: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    """Classify multiple text items in one Ollama request."""

    if not items:
        return {}
    prompt = {
        "task": "Classify each document text item independently.",
        "labels": {
            "meta-data": (
                "An author name or author list, affiliation, email address, "
                "postal address, or a combination of those author details."
            ),
            "reference": (
                "A clean bibliographic reference entry: a cited work with such "
                "signals as author(s), title, year, venue, publisher, or identifier."
            ),
            "text": "Any other prose, heading, caption, keyword list, or text item.",
        },
        "items": [
            {
                "text": item["text"],
                "baseline": {
                    "label": item["prediction_label"],
                    "confidence": item["prediction_confidence"],
                    "in_reference_section": item["in_reference_section"],
                },
            }
            for item in items
        ],
        "instruction": "Return one label per item, in the same order as the input items.",
    }
    response_schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(LABELS)},
                "minItems": len(items),
                "maxItems": len(items),
            },
        },
        "required": ["labels"],
    }
    started = time.monotonic()
    response = requests.post(
        f"{args.ollama_url.rstrip('/')}/api/chat",
        headers={"Content-Type": "application/json"},
        json={
            "model": args.llm_model,
            "stream": False,
            "think": args.llm_thinking,
            "format": response_schema,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_ctx": args.llm_context_window,
                "num_predict": max(32, 8 * len(items)),
            },
            "messages": [
                {"role": "system", "content": "You are a careful document reviewer."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        },
        timeout=args.ollama_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        parsed = json.loads(payload["message"]["content"].strip())
    except (KeyError, AttributeError, json.JSONDecodeError):
        parsed = {}
    labels = parsed.get("labels")
    accepted = (
        isinstance(labels, list)
        and len(labels) == len(items)
        and all(label in LABELS for label in labels)
    )
    if not accepted and len(items) > 1:
        midpoint = len(items) // 2
        LOG.warning(
            "Ollama returned an invalid %d-item batch; retrying as %d and %d items",
            len(items), midpoint, len(items) - midpoint,
        )
        return {
            **local_reviews(items[:midpoint], args),
            **local_reviews(items[midpoint:], args),
        }
    if not accepted:
        labels = [None] * len(items)
    if LOG.isEnabledFor(logging.DEBUG):
        timing = {
            key: payload.get(key)
            for key in (
                "total_duration", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration",
            )
            if key in payload
        }
        LOG.debug(
            "Ollama batch: %d samples in %.2fs; timing=%s",
            len(items), time.monotonic() - started, timing,
        )
    reviews = {}
    for index, (item, label) in enumerate(zip(items, labels)):
        confidence = item["prediction_confidence"]
        compact_text = " ".join(item["text"].split())
        if len(compact_text) > 180:
            compact_text = f"{compact_text[:177]}..."
        if LOG.isEnabledFor(logging.DEBUG):
            tqdm.write(
                "conf: %s, orig: %s -> pred: %s, text: %s" % (
                    f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a",
                    f"{item['prediction_label'] or 'unknown':>9}",
                    f"{label or 'invalid':>9}",
                    compact_text,
                ),
                file=sys.stderr,
            )
        item_parsed = {"label": label} if label in LABELS else None
        item_request = {
            "task": prompt["task"],
            "labels": prompt["labels"],
            "item": prompt["items"][index],
            "batch_position": index,
            "batch_size": len(items),
            "instruction": prompt["instruction"],
        }
        reviews[item["candidate_id"]] = {
            "accepted": item_parsed is not None,
            "parsed": item_parsed,
            "request": item_request,
            "response": payload,
        }
    return reviews


def is_training_sample(candidate_id: str, ratio: float) -> bool:
    return int(candidate_id[:8], 16) / 0xFFFFFFFF >= ratio


def structural_reference_seed(item: dict[str, Any]) -> bool:
    return (
        item["in_reference_section"]
        and item["layout_label"] in TRAINABLE_LAYOUT_LABELS
    )


def bootstrap_rows(
    semantic_predictions: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Create seeds from confident SEMANTIC labels and Docling reference sections."""

    rows = []
    for item in semantic_predictions:
        if item["layout_label"] not in TRAINABLE_LAYOUT_LABELS:
            continue
        if item.get("numval_only"):
            continue
        if structural_reference_seed(item):
            label, source = "reference", "references-section"
        elif not uncertain(item, args.min_confidence):
            label, source = item["prediction_label"], "semantic"
        else:
            continue
        rows.append(
            {
                "candidate-id": item["candidate_id"],
                "source-path": item["source_path"],
                "label": label,
                "label-source": source,
                "applied-label": item["prediction_label"],
                "confidence": item["prediction_confidence"],
                "text": item["text"],
                "training-sample": is_training_sample(
                    item["candidate_id"], args.validation_ratio
                ),
            }
        )
    return rows


def label_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {row["candidate-id"]: row["label"] for row in rows}


def accepted_labels(reviews: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        candidate_id: review["parsed"]["label"]
        for candidate_id, review in reviews.items()
        if review.get("accepted") and review.get("parsed", {}).get("label") in LABELS
    }


def reviewed_candidate_ids(output_dir: Path) -> set[str]:
    """Return candidates with a completed Ollama response in an earlier iteration."""

    reviewed = set()
    review_paths = list(output_dir.glob("iteration-*/reviews.json"))
    review_paths.extend((output_dir / "metadata" / "reviews").glob("*.json"))
    for path in sorted(review_paths):
        for review in read_json(path):
            if review.get("response") is not None:
                reviewed.add(review["candidate-id"])
    return reviewed


def confusion_report(observations: list[tuple[str, str]]) -> dict[str, Any]:
    """Return a JSON-serialisable confusion matrix and per-label PRF metrics."""

    matrix = {actual: {predicted: 0 for predicted in sorted(LABELS)} for actual in sorted(LABELS)}
    for actual, predicted in observations:
        if actual in LABELS and predicted in LABELS:
            matrix[actual][predicted] += 1
    metrics = {}
    for label in sorted(LABELS):
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in LABELS if actual != label)
        false_negative = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[label].values())}
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in LABELS)
    return {"samples": total, "accuracy": correct / total if total else 0.0, "matrix": matrix, "metrics": metrics}


def format_confusion_report(report: dict[str, Any]) -> str:
    """Render a compact matrix and per-label precision, recall, and F1 table."""

    labels = sorted(LABELS)
    matrix = report["matrix"]
    matrix_format = "%-11s | " + " | ".join("%9s" for _ in labels)
    matrix_row_format = "%-11s | " + " | ".join("%9d" for _ in labels)
    lines = [
        "rows=orig, columns=pred; samples=%d" % report["samples"],
        matrix_format % ("orig \\ pred", *labels),
        "-" * (14 + 13 * len(labels)),
    ]
    lines.extend(
        matrix_row_format % (label, *(matrix[label][predicted] for predicted in labels))
        for label in labels
    )
    lines.extend([
        "metrics:",
        "%-11s %9s %9s %9s %9s" % (
            "label", "precision", "recall", "F1", "support"
        ),
    ])
    lines.extend(
        "%-11s %9.3f %9.3f %9.3f %9d" % (
            label,
            report["metrics"][label]["precision"],
            report["metrics"][label]["recall"],
            report["metrics"][label]["f1"],
            report["metrics"][label]["support"],
        )
        for label in labels
    )
    return "\n".join(lines)


def paired_training_rows(
    predictions: list[dict[str, Any]],
    seed_labels: dict[str, str],
    llm_labels: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build matched baseline/corrected data and LLM-only held-out evaluation rows."""

    baseline, corrected, validation = [], [], []
    for item in predictions:
        candidate_id = item["candidate_id"]
        llm_label = llm_labels.get(candidate_id)
        seed_label = seed_labels.get(candidate_id)
        base_label = item["prediction_label"] if item["prediction_label"] in LABELS else seed_label
        training_sample = is_training_sample(candidate_id, args.validation_ratio)
        if llm_label and not training_sample:
            base_label = corrected_label = llm_label
            validation.append({**item, "label": llm_label})
        elif llm_label:
            corrected_label = llm_label
        elif seed_label:
            base_label = corrected_label = seed_label
        elif not uncertain(item, args.min_confidence):
            base_label = corrected_label = item["prediction_label"]
        else:
            continue
        if base_label not in LABELS or corrected_label not in LABELS:
            continue
        common = {
            "candidate-id": candidate_id,
            "source-path": item["source_path"],
            "xpath": item.get("xpath", f"/doclang[1]/text[{int(item.get('position', 0)) + 1}]"),
            "introduced_iteration": item.get("introduced_iteration"),
            "text": item["text"],
            "training-sample": training_sample,
            "applied-label": item["prediction_label"],
            "confidence": item["prediction_confidence"],
        }
        baseline.append({
            **common,
            "label": base_label,
            "label-source": "latest-model" if base_label == item["prediction_label"] else "bootstrap",
        })
        corrected.append({
            **common,
            "label": corrected_label,
            "label-source": "llm" if llm_label else ("bootstrap" if seed_label else "latest-model"),
        })
    return baseline, corrected, validation


def training_label_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count each label in the explicit train and validation partitions."""

    counts = {
        "train": {label: 0 for label in sorted(LABELS)},
        "validation": {label: 0 for label in sorted(LABELS)},
    }
    for row in rows:
        label = row.get("label")
        if label not in LABELS:
            continue
        split = "train" if row.get("training-sample") else "validation"
        counts[split][label] += 1
    return counts


def train(
    data_file: Path,
    directory: Path,
    model_stem: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    from docling_nlp.nlp_utils import eval_fst, prepare_data_for_fst_training, train_fst

    model_file = directory / f"{model_stem}.bin"
    metrics_file = directory / f"{model_stem}.metrics.txt"
    counts = training_label_counts(rows)
    LOG.info(
        "%s training data: %d rows; train=%s; validation=%s",
        model_stem,
        len(rows),
        counts["train"],
        counts["validation"],
    )
    LOG.info("Preparing FastText training data: %s", data_file)
    prepare_data_for_fst_training(data_file=str(data_file))
    for attempt in range(args.fasttext_retries + 1):
        if attempt:
            LOG.warning(
                "Retrying FastText %s after incomplete artifacts (%d/%d)",
                model_stem, attempt, args.fasttext_retries,
            )
            time.sleep(1)
        LOG.info("Training FastText model in %s", directory)
        train_fst(str(data_file), str(model_file), str(metrics_file), args.autotune,
                  args.autotune_duration, args.autotune_modelsize, args.ngram)
        LOG.info("Evaluating FastText model")
        eval_fst(str(data_file), str(model_file), str(metrics_file))
        if model_file.is_file() and model_file.stat().st_size and metrics_file.is_file():
            break
    else:
        raise RuntimeError(
            f"FastText training/evaluation did not create expected artifacts after "
            f"{args.fasttext_retries + 1} attempt(s): {model_file}"
        )
    metrics_text = metrics_file.read_text(encoding="utf-8")
    match = re.search(r"%-perfect:\s+([0-9.]+)\s+\[(\d+)/(\d+)\]", metrics_text)
    if match:
        score, correct, total = match.groups()
        LOG.info(
            "Native validation complete: %.2f%% (%s/%s); metrics: %s",
            float(score) * 100,
            correct,
            total,
            metrics_file,
        )
    else:
        LOG.info("Native validation complete; metrics: %s", metrics_file)
    return model_file, metrics_file


def persist_latest_model(output_dir: Path, model_file: Path, metrics_file: Path) -> tuple[Path, Path]:
    """Atomically replace the single model retained for the next iteration."""

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    latest_model = models_dir / "latest-reference.bin"
    latest_metrics = models_dir / "latest-reference.metrics.txt"
    for source, target in ((model_file, latest_model), (metrics_file, latest_metrics)):
        staged = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, staged)
        staged.replace(target)
    return latest_model, latest_metrics


def persist_iteration(
    args: argparse.Namespace,
    number: int,
    directory: Path,
    summary: dict[str, Any],
    corrected_rows: list[dict[str, Any]],
    corrected_model: Path,
    corrected_metrics: Path,
    comparison: dict[str, Any],
    reviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Retain compact run state and make the per-iteration workspace disposable."""

    latest_model, latest_metrics = persist_latest_model(
        args.output_dir, corrected_model, corrected_metrics
    )
    write_persistent_datasets(
        args.output_dir,
        number,
        corrected_rows,
        f"custom_fst(references:{latest_model})",
    )
    reviews_dir = args.output_dir / "metadata" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        reviews_dir / f"iteration-{number:03d}.json",
        [{"candidate-id": identifier, **review} for identifier, review in reviews.items()],
    )
    summary = {
        **summary,
        "model_file": str(latest_model),
        "metrics_file": str(latest_metrics),
        "comparison_file": str(args.output_dir / "metrics" / f"iteration-{number:03d}.json"),
    }
    write_iteration_metrics(args.output_dir, summary, comparison)
    return summary


def add_candidate_locations(rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    """Backfill source locations for data created before persistent CSVs existed."""

    locations = {
        item["candidate_id"]: item for item in candidates
    }
    for row in rows:
        source = locations.get(row["candidate-id"], {})
        row.setdefault("xpath", source.get("xpath", f"/doclang[1]/text[{int(source.get('position', 0)) + 1}]"))


def migrate_legacy_run(args: argparse.Namespace) -> None:
    """Compact the final retained legacy iteration before resuming a run."""

    if not args.resume:
        return
    summary_file = args.output_dir / "run-summary.json"
    if not summary_file.exists():
        return
    summaries = read_json(summary_file)
    if not summaries:
        return
    latest_model = args.output_dir / "models" / "latest-reference.bin"
    if latest_model.exists():
        return
    last_summary = summaries[-1]
    number = int(last_summary["iteration"])
    legacy_dir = args.output_dir / f"iteration-{number:03d}"
    corrected_data = legacy_dir / "reference-corrected.jsonl"
    corrected_model = Path(last_summary["model_file"])
    corrected_metrics = Path(last_summary["metrics_file"])
    if not (legacy_dir.is_dir() and corrected_data.is_file() and corrected_model.is_file()
            and corrected_metrics.is_file()):
        return

    LOG.info("Migrating retained legacy iteration %d into compact resumable artifacts", number)
    corrected_rows = read_jsonl(corrected_data)
    candidates_file = args.output_dir / "reference-candidates.json"
    if candidates_file.exists():
        add_candidate_locations(corrected_rows, read_json(candidates_file))
    comparison_file = legacy_dir / "model-comparison.json"
    comparison = read_json(comparison_file) if comparison_file.exists() else {}
    reviews_file = legacy_dir / "reviews.json"
    reviews = {
        row["candidate-id"]: {key: value for key, value in row.items() if key != "candidate-id"}
        for row in read_json(reviews_file)
    } if reviews_file.exists() else {}

    for historic_summary in summaries[:-1]:
        historic_comparison_path = Path(historic_summary.get("comparison_file", ""))
        historic_comparison = (
            read_json(historic_comparison_path)
            if historic_comparison_path.is_file() else {}
        )
        write_iteration_metrics(args.output_dir, historic_summary, historic_comparison)
    summaries[-1] = persist_iteration(
        args, number, legacy_dir, last_summary, corrected_rows, corrected_model,
        corrected_metrics, comparison, reviews,
    )
    write_json(summary_file, summaries)
    required = [
        latest_model,
        args.output_dir / "data" / "train.csv",
        args.output_dir / "data" / "validation.csv",
        args.output_dir / "metrics" / "iterations.csv",
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("legacy migration did not create all compact resume artifacts")
    shutil.rmtree(legacy_dir)
    LOG.info("Removed migrated temporary workspace %s", legacy_dir)


def create_bootstrap_training_data(
    args: argparse.Namespace,
    semantic_predictions: list[dict[str, Any]],
) -> tuple[Path, dict[str, str]]:
    """Create/reuse the confident SEMANTIC and REFERENCES-section seed model."""

    summary_file = args.output_dir / "bootstrap-summary.json"
    labels_file = args.output_dir / "bootstrap-labels.json"
    if summary_file.exists():
        summary = read_json(summary_file)
        model = Path(summary["model_file"])
        if not model.exists():
            raise RuntimeError(f"bootstrap model is missing: {model}")
        LOG.info("Reusing bootstrap model %s", model)
        return model, read_json(labels_file)

    seeds = bootstrap_rows(semantic_predictions, args)
    seed_counts = {label: sum(row["label"] == label for row in seeds) for label in LABELS}
    seed_validation = sum(not row["training-sample"] for row in seeds)
    if (not all(seed_counts[label] for label in {"reference", "text"})
            or seed_validation == 0 or seed_validation == len(seeds)):
        raise RuntimeError(
            f"insufficient bootstrap labels/splits: {seed_counts}, validation={seed_validation}"
        )
    directory = args.output_dir / "bootstrap"
    if directory.exists() and not args.resume:
        raise FileExistsError(
            f"{directory} already exists; use --resume to retry this bootstrap run"
        )
    directory.mkdir(exist_ok=True)
    data_file = directory / "reference-bootstrap.jsonl"
    write_jsonl(data_file, seeds)
    write_dataset_csv(directory / "reference-bootstrap.csv", seeds)
    seed_labels = label_map(seeds)
    write_json(labels_file, seed_labels)
    LOG.info("Bootstrapping from %d seeds (%s; %d validation)", len(seeds), seed_counts, seed_validation)
    model, metrics = train(data_file, directory, "bootstrap-classifier", args, seeds)
    bootstrap_evaluated = apply(
        f"custom_fst(references:{model})",
        [row for row in seeds if not row["training-sample"]],
    )
    write_json(
        directory / "bootstrap-evaluation.json",
        {
            "ground_truth": "bootstrap seed labels (not LLM labels)",
            **confusion_report([
                (item["label"], item["prediction_label"])
                for item in bootstrap_evaluated if item["prediction_label"] in LABELS
            ]),
        },
    )
    write_json(
        summary_file,
        {"model_file": str(model), "metrics_file": str(metrics), "seeds": len(seeds), "labels": seed_counts},
    )
    return model, seed_labels


def refine_training_data(
    args: argparse.Namespace,
    number: int,
    directory: Path,
    expression: str,
    candidates: list[dict[str, Any]],
    seed_labels: dict[str, str],
    llm_labels: dict[str, str],
) -> dict[str, Any]:
    """Review low-confidence predictions, retrain, and evaluate one iteration."""

    predictions_file = directory / "predictions.json"
    reviews_file = directory / "reviews.json"
    if args.resume and predictions_file.exists():
        predictions = read_json(predictions_file)
        LOG.info("Resuming iteration %d from %s (%d predictions)", number, predictions_file, len(predictions))
    else:
        predictions = apply(expression, candidates)
        write_json(predictions_file, predictions)
    pending = [item for item in predictions if uncertain(item, args.min_confidence)]
    already_reviewed = reviewed_candidate_ids(args.output_dir)
    unseen_pending = [item for item in pending if item["candidate_id"] not in already_reviewed]
    skipped = len(pending) - len(unseen_pending)
    if args.review_limit == -1:
        LOG.info(
            "Found %d uncertain candidates; skipped %d previously reviewed; reviewing all %d with %s",
            len(pending), skipped, len(unseen_pending), args.llm_model,
        )
        review_items = unseen_pending
    else:
        LOG.info(
            "Found %d uncertain candidates; skipped %d previously reviewed; reviewing up to %d of %d with %s",
            len(pending), skipped, args.review_limit, len(unseen_pending), args.llm_model,
        )
        review_items = unseen_pending[:args.review_limit]
    if args.resume and reviews_file.exists():
        reviews = {
            row["candidate-id"]: {key: value for key, value in row.items() if key != "candidate-id"}
            for row in read_json(reviews_file)
        }
        LOG.info("Reusing %d completed Ollama reviews", len(reviews))
    else:
        reviews = {}
        progress = tqdm(
            total=len(review_items), desc="Ollama review", unit="sample",
            file=sys.stderr, dynamic_ncols=True,
        )
        for start in range(0, len(review_items), args.llm_batch_size):
            batch = review_items[start:start + args.llm_batch_size]
            try:
                reviews.update(local_reviews(batch, args))
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                reviews.update({
                    item["candidate_id"]: {"accepted": False, "error": str(exc)}
                    for item in batch
                })
            progress.update(len(batch))
        progress.close()
        write_json(reviews_file, [
            {"candidate-id": identifier, **review}
            for identifier, review in reviews.items()
        ])
    new_labels = accepted_labels(reviews)
    review_predictions = {item["candidate_id"]: item["prediction_label"] for item in predictions}
    review_report = {
        "row_label": "original_model_label",
        "column_label": "llm_label",
        **confusion_report([
            (review_predictions[candidate_id], label)
            for candidate_id, label in new_labels.items()
            if review_predictions.get(candidate_id) in LABELS
        ]),
    }
    write_json(directory / "llm-review-confusion-matrix.json", review_report)
    LOG.info("LLM review confusion matrix and metrics:\n%s", format_confusion_report(review_report))
    llm_labels.update(new_labels)
    write_json(args.output_dir / "llm-corrections.json", llm_labels)
    baseline_rows, corrected_rows, validation_rows = paired_training_rows(predictions, seed_labels, llm_labels, args)
    validation_labels = {label: sum(row["label"] == label for row in validation_rows) for label in LABELS}
    if sum(validation_labels.values()) == 0:
        raise RuntimeError(f"held-out LLM validation has no labelled rows: {validation_labels}")
    baseline_data = directory / "reference-baseline.jsonl"
    corrected_data = directory / "reference-corrected.jsonl"
    write_jsonl(baseline_data, baseline_rows)
    write_jsonl(corrected_data, corrected_rows)
    write_dataset_csv(directory / "reference-baseline.csv", baseline_rows)
    write_dataset_csv(directory / "reference-corrected.csv", corrected_rows)
    LOG.info("Wrote accumulated baseline/corrected datasets (%d rows; %d LLM validation)", len(corrected_rows), len(validation_rows))
    baseline_model, baseline_metrics = train(baseline_data, directory, "baseline-classifier", args, baseline_rows)
    model_file, metrics_file = train(corrected_data, directory, "corrected-classifier", args, corrected_rows)
    comparison = {}
    for name, model_path in (("baseline", baseline_model), ("corrected", model_file)):
        evaluated = apply(f"custom_fst(references:{model_path})", validation_rows)
        comparison[name] = confusion_report([
            (item["label"], item["prediction_label"])
            for item in evaluated if item["prediction_label"] in LABELS
        ])
        write_json(directory / f"{name}-evaluation.json", comparison[name])
    write_json(directory / "model-comparison.json", comparison)
    baseline_evaluation = comparison["baseline"]
    corrected_evaluation = comparison["corrected"]
    LOG.info(
        "LLM held-out evaluation (n=%d): accuracy %.1f%% -> %.1f%%; reference F1 %.3f -> %.3f; text F1 %.3f -> %.3f",
        corrected_evaluation["samples"], baseline_evaluation["accuracy"] * 100, corrected_evaluation["accuracy"] * 100,
        baseline_evaluation["metrics"]["reference"]["f1"], corrected_evaluation["metrics"]["reference"]["f1"],
        baseline_evaluation["metrics"]["text"]["f1"], corrected_evaluation["metrics"]["text"]["f1"],
    )
    summary = {
        "iteration": number, "baseline": expression, "candidates": len(candidates),
        "uncertain": len(pending), "reviewed": len(new_labels), "validation_labels": validation_labels,
        "baseline_model_file": str(baseline_model), "baseline_metrics_file": str(baseline_metrics),
        "model_file": str(model_file), "metrics_file": str(metrics_file),
        "comparison_file": str(directory / "model-comparison.json"),
        "training_label_counts": training_label_counts(corrected_rows),
    }
    summary = persist_iteration(
        args, number, directory, summary, corrected_rows, model_file, metrics_file,
        comparison, reviews,
    )
    write_json(directory / "summary.json", summary)
    return summary


def iterate_over_training(
    args: argparse.Namespace,
    bootstrap_model: Path,
    seed_labels: dict[str, str],
    semantic_predictions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Accumulate a new source batch per iteration, then refine and retrain."""

    summary_file = args.output_dir / "run-summary.json"
    summaries = read_json(summary_file) if args.resume and summary_file.exists() else []
    if summaries:
        last_model = Path(summaries[-1]["model_file"])
        if not last_model.exists():
            raise RuntimeError(f"cannot resume: previous model is missing: {last_model}")
        expression = f"custom_fst(references:{last_model})"
        start_iteration = int(summaries[-1]["iteration"]) + 1
    else:
        expression = f"custom_fst(references:{bootstrap_model})"
        start_iteration = 1
    corrections_file = args.output_dir / "llm-corrections.json"
    llm_labels = read_json(corrections_file) if corrections_file.exists() else {}

    for number in range(start_iteration, start_iteration + args.iterations):
        directory = args.output_dir / "tmp" / f"iteration-{number:03d}"
        if directory.exists() and not args.resume:
            raise RuntimeError(f"{directory} already exists; use --resume or a new output directory")
        directory.mkdir(parents=True, exist_ok=True)
        if number > 1:
            if args.arxiv_mode:
                batch_file = directory / "arxiv-batch.json"
                if batch_file.exists():
                    sources = []
                    LOG.info("Reusing arXiv batch recorded in %s", batch_file)
                else:
                    sources = fetch_arxiv_batch(args)
                    write_json(batch_file, [str(source.relative_to(args.output_dir)) for source in sources])
            else:
                sources = select_sources(args, number - 1, directory / "selected-files.json")
            if sources:
                new_semantic_predictions = annotate_numval(
                    apply("semantic", _convert_source_batch_to_dclx(args, sources))
                )
                for item in new_semantic_predictions:
                    item["introduced_iteration"] = number
                known_prediction_ids = {item["candidate_id"] for item in semantic_predictions}
                new_semantic_predictions = [
                    item for item in new_semantic_predictions
                    if item["candidate_id"] not in known_prediction_ids
                ]
                semantic_predictions.extend(new_semantic_predictions)
                known_candidate_ids = {item["candidate_id"] for item in candidates}
                new_candidates = [
                    item for item in trainable_candidates(new_semantic_predictions)
                    if item["candidate_id"] not in known_candidate_ids
                ]
                candidates.extend(new_candidates)
                write_json(args.output_dir / "semantic-predictions.json", semantic_predictions)
                write_json(args.output_dir / "reference-candidates.json", candidates)
                seed_labels.update(label_map(bootstrap_rows(new_semantic_predictions, args)))
                write_json(args.output_dir / "bootstrap-labels.json", seed_labels)
                LOG.info(
                    "Added %d files / %d candidates; accumulated %d candidates",
                    len(sources), len(new_candidates), len(candidates),
                )
            else:
                LOG.info("No additional input files remain for iteration %d; reusing %d candidates", number, len(candidates))
        LOG.info("Starting iteration %d with baseline %r", number, expression)
        summary = refine_training_data(
            args, number, directory, expression, candidates, seed_labels, llm_labels
        )
        summaries.append(summary)
        expression = f"custom_fst(references:{summary['model_file']})"
        write_json(summary_file, summaries)
        shutil.rmtree(directory)
        LOG.info("Removed completed temporary workspace %s", directory)
    LOG.info("Completed %d iteration(s)", len(summaries))
    return summaries


def main() -> int:
    args = arguments()
    configure_logging(args.debug)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Writing run artifacts to %s", args.output_dir)
    migrate_legacy_run(args)
    if args.compact_run:
        if not args.resume:
            raise RuntimeError("--compact-run requires --resume")
        return 0
    semantic_predictions, candidates = convert_documents_to_dclx(args)
    if not candidates:
        raise RuntimeError("no Docling text/list-item candidates found")
    bootstrap_model, seed_labels = create_bootstrap_training_data(args, semantic_predictions)
    summaries = iterate_over_training(
        args, bootstrap_model, seed_labels, semantic_predictions, candidates
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
