#!/usr/bin/env python3
"""Print a DocLangX document's content tree."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from docling_core.types.doc import DoclingDocument
from docling_nlp.andromeda_doclang import DocLangXDocument
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

LIBRARY = Path(__file__).resolve().parents[2] / "artifacts" / "library"
METADATA = {"location", "layer", "src", "fcel", "lcel", "nl", "ched", "marker"}
PREVIEW_LENGTH = 100


def is_url(source: str) -> bool:
    parsed = urlsplit(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def archive_path(source: str) -> Path:
    name = unquote(urlsplit(source).path) if is_url(source) else source
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return LIBRARY / f"{stem or 'document'}-{digest}.dclx"


def convert(source: str) -> Path:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Conversion requires Docling; run: uv sync --extra docling"
        ) from exc

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False
    pdf_options.generate_page_images = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )
    result = converter.convert(source)
    if result.document is None:
        raise RuntimeError(f"Docling did not produce a document for {source}")

    destination = archive_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.document.save_as_doclang_archive(destination)
    return destination


def resolve_input(source: str) -> tuple[Path, bool]:
    if is_url(source):
        return convert(source), True

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"input file does not exist: {path}")
    if path.suffix.lower() == ".dclx":
        return path, False
    return convert(str(path)), True


def iter_content(document: DocLangXDocument, parent_xpath: str | None = None):
    """Yield native items recursively, retaining their native XPaths."""
    for xpath, item, page_no, bbox in document.iterate_items(xpath=parent_xpath):
        if item["name"] in METADATA:
            continue
        yield xpath, item, page_no, bbox
        if item["name"] != "list_item":
            yield from iter_content(document, xpath)


def label(xpath: str, item: dict) -> Text:
    segment = xpath.rsplit("/", 1)[-1]
    index = segment[segment.find("[") :] if "[" in segment else ""
    result = Text(f"{item['name']}{index}", style="bold cyan")

    content = " ".join(item["text"].split())
    if content:
        preview = (
            content[: PREVIEW_LENGTH - 3] + "..."
            if len(content) > PREVIEW_LENGTH
            else content
        )
        result.append(f"  {preview}")
    return result


def build_tree(path: Path, document: DocLangXDocument) -> Tree:
    tree = Tree(Text(path.name, style="bold"))
    root_xpath = "/doclang[1]"
    nodes: dict[str, Tree] = {root_xpath: tree}

    def parent_node(xpath: str) -> Tree:
        if xpath in nodes:
            return nodes[xpath]
        parent_xpath, _, segment = xpath.rpartition("/")
        parent = parent_node(parent_xpath)
        node = parent.add(Text(segment, style="bold cyan"))
        nodes[xpath] = node
        return node

    for xpath, item, _, _ in iter_content(document):
        parent_xpath = xpath.rpartition("/")[0]
        nodes[xpath] = parent_node(parent_xpath).add(label(xpath, item))
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="A .dclx file, local file, or HTTP(S) URL")
    parser.add_argument(
        "--json",
        nargs="?",
        const="",
        metavar="PATH",
        help="Also save the DoclingDocument as JSON (default: beside the .dclx)",
    )
    args = parser.parse_args()

    try:
        path, converted = resolve_input(args.input)
        json_path = None
        if args.json is not None:
            json_path = (
                Path(args.json).expanduser().resolve()
                if args.json
                else path.with_suffix(".json")
            )
            docling_document = DoclingDocument.load_from_doclang_archive(path)
            docling_document.save_as_json(json_path)

        document = DocLangXDocument()
        if not document.read(str(path)):
            raise RuntimeError(f"could not read {path}: {document.last_error()}")
        tree = build_tree(path, document)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")

    console = Console()
    if converted:
        console.print(f"Saved DCLX: {path}")
    if json_path is not None:
        console.print(f"Saved JSON: {json_path}")
    console.print(tree)


if __name__ == "__main__":
    main()
