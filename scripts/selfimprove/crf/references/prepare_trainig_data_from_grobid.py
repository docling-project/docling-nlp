#!/usr/bin/env python3
"""Download GROBID citation XML files and create CRF annotation splits.

Only regular XML files from GROBID's citation corpus directory are downloaded.
The generated JSONL records use end-exclusive Python character offsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)
REPOSITORY = "grobidOrg/grobid"
CORPUS_PATH = "grobid-trainer/resources/dataset/citation/corpus"
GITHUB_API = "https://api.github.com"


def local_name(name: str) -> str:
    """Return an XML name without its namespace."""

    return name.rsplit("}", 1)[-1]


def normalised_attribute(element: ET.Element, *names: str) -> str:
    for name in names:
        value = element.attrib.get(name)
        if value:
            return value.strip().lower()
    return ""


def citation_label(element: ET.Element, inherited: str | None) -> str | None:
    """Map a GROBID TEI element to the reference model's label vocabulary."""

    tag = local_name(element.tag).lower()
    if tag in {"author", "editor"}:
        return "authors"
    if tag == "title":
        level = normalised_attribute(element, "level")
        if level == "j":
            return "journal"
        if level == "s":
            return "conference"
        return "title"
    if tag in {"biblscope", "biblstruct"}:
        kind = normalised_attribute(element, "unit", "type", "level")
        return {
            "vol": "volume",
            "volume": "volume",
            "issue": "issue",
            "page": "pages",
            "pages": "pages",
            "pp": "pages",
        }.get(kind, inherited)
    if tag == "date":
        return "date"
    if tag in {"publisher", "orgname", "institution"}:
        return "publisher"
    if tag == "pubplace":
        return "location"
    if tag in {"ptr", "ref"}:
        return "url"
    if tag == "idno":
        kind = normalised_attribute(element, "type")
        return {
            "doi": "doi",
            "isbn": "isbn",
            "issn": "issn",
        }.get(kind, "identifier")
    if tag == "note":
        return "note"
    return inherited


def flatten_bibliography(element: ET.Element) -> tuple[str, list[dict[str, Any]]]:
    """Flatten mixed XML content while retaining labelled character ranges."""

    pieces: list[tuple[str, str | None]] = []

    def visit(node: ET.Element, inherited: str | None = None) -> None:
        label = citation_label(node, inherited)
        if node.text:
            pieces.append((node.text, label))
        for child in node:
            visit(child, label)
            if child.tail:
                pieces.append((child.tail, label))

    visit(element)
    raw_text = "".join(value for value, _ in pieces)
    left = len(raw_text) - len(raw_text.lstrip())
    right = len(raw_text.rstrip())
    text = raw_text[left:right]

    annotations: list[dict[str, Any]] = []
    cursor = 0
    for value, label in pieces:
        start = cursor
        cursor += len(value)
        clipped_start = max(start, left)
        clipped_end = min(cursor, right)
        if label and clipped_start < clipped_end:
            begin = clipped_start - left
            end = clipped_end - left
            if (
                annotations
                and annotations[-1]["label"] == label
                and annotations[-1]["end"] == begin
            ):
                annotations[-1]["end"] = end
            else:
                annotations.append({"label": label, "start": begin, "end": end})

    return text, annotations


def parse_xml(path: Path) -> list[dict[str, Any]]:
    """Extract one raw CRF record per bibl element."""

    root = ET.parse(path).getroot()
    records: list[dict[str, Any]] = []
    for index, element in enumerate(root.iter()):
        if local_name(element.tag).lower() != "bibl":
            continue
        text, annotations = flatten_bibliography(element)
        if not text:
            continue
        records.append(
            {
                "source_file": path.name,
                "source_index": index,
                "text": text,
                "annotation": annotations,
            }
        )
    return records


def split_files(
    paths: list[Path], validation_ratio: float, seed: int
) -> tuple[set[Path], set[Path]]:
    """Split at file level so references from one source cannot leak across splits."""

    ordered = sorted(paths, key=lambda path: path.name)
    random.Random(seed).shuffle(ordered)
    validation_count = round(len(ordered) * validation_ratio)
    if len(ordered) > 1:
        validation_count = min(max(validation_count, 1), len(ordered) - 1)
    return set(ordered[validation_count:]), set(ordered[:validation_count])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def request_json(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def raw_file_url(commit_sha: str, name: str) -> str:
    """Build a raw URL that preserves literal percent escapes in filenames."""

    encoded_name = quote(name, safe="")
    return (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{commit_sha}/"
        f"{CORPUS_PATH}/{encoded_name}"
    )


def download_corpus(
    raw_dir: Path, ref: str, force: bool
) -> tuple[str, list[dict[str, Any]]]:
    """Download only regular XML data files from the configured GitHub directory."""

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "User-Agent": "docling-nlp-grobid-corpus-preparer",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    commit = request_json(session, f"{GITHUB_API}/repos/{REPOSITORY}/commits/{ref}")
    commit_sha = commit["sha"]
    listing = request_json(
        session,
        f"{GITHUB_API}/repos/{REPOSITORY}/contents/{CORPUS_PATH}?ref={commit_sha}",
    )
    files = sorted(
        (
            item
            for item in listing
            if item.get("type") == "file"
            and item.get("name", "").lower().endswith(".xml")
        ),
        key=lambda item: item["name"],
    )
    if not files:
        raise RuntimeError(
            "GitHub returned no XML files for the GROBID citation corpus"
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    for number, item in enumerate(files, start=1):
        target = raw_dir / item["name"]
        if target.is_file() and not force:
            LOG.info("[%d/%d] Reusing %s", number, len(files), target.name)
            continue
        LOG.info("[%d/%d] Downloading %s", number, len(files), target.name)
        response = session.get(
            raw_file_url(commit_sha, item["name"]), timeout=60, stream=True
        )
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as stream:
            temporary = Path(stream.name)
            for chunk in response.iter_content(chunk_size=64 * 1024):
                stream.write(chunk)
        temporary.replace(target)
    return commit_sha, files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    output_dir: Path, ref: str, validation_ratio: float, seed: int, force: bool
) -> None:
    raw_dir = output_dir / "raw"
    commit_sha, remote_files = download_corpus(raw_dir, ref, force)
    paths = [raw_dir / item["name"] for item in remote_files]
    training_files, validation_files = split_files(paths, validation_ratio, seed)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    label_counts: Counter[str] = Counter()
    for path in paths:
        records = parse_xml(path)
        split = "train" if path in training_files else "validation"
        splits[split].extend(records)
        label_counts.update(
            annotation["label"]
            for record in records
            for annotation in record["annotation"]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", splits["train"])
    write_jsonl(output_dir / "validation.jsonl", splits["validation"])
    manifest = {
        "source": f"https://github.com/{REPOSITORY}/tree/{commit_sha}/{CORPUS_PATH}",
        "commit": commit_sha,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "files": [
            {
                "name": path.name,
                "sha256": sha256(path),
                "split": "train" if path in training_files else "validation",
            }
            for path in paths
        ],
        "records": {name: len(rows) for name, rows in splits.items()},
        "labels": dict(sorted(label_counts.items())),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOG.info(
        "Wrote %d training and %d validation references to %s",
        len(splits["train"]),
        len(splits["validation"]),
        output_dir,
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/grobid-crf-references"),
        help="Directory receiving raw XML, JSONL splits, and the manifest.",
    )
    parser.add_argument(
        "--ref", default="master", help="GROBID branch, tag, or commit."
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of source files assigned to validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used for splitting."
    )
    parser.add_argument(
        "--force", action="store_true", help="Redownload existing XML files."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be between 0 and 1")
    return args


def main() -> None:
    args = arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    prepare(args.output_dir, args.ref, args.validation_ratio, args.seed, args.force)


if __name__ == "__main__":
    main()
