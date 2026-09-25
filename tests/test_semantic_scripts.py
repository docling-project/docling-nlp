import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "scripts/selfimprove/semantic"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_semantic")
document_runner = load_script("run_semantic_document")
trainer = load_script("train_semantic")


def test_download_file_reuses_existing_copy(tmp_path: Path, monkeypatch):
    destination = tmp_path / "train.csv"
    destination.write_text("local", encoding="utf-8")

    def unexpected_download(**kwargs):
        raise AssertionError(f"unexpected download: {kwargs}")

    monkeypatch.setattr(trainer, "hf_hub_download", unexpected_download)

    assert (
        trainer.download_file("repo", "remote/train.csv", "main", destination)
        == destination
    )
    assert destination.read_text(encoding="utf-8") == "local"


def test_download_file_fetches_missing_copy(tmp_path: Path, monkeypatch):
    cached = tmp_path / "cached.csv"
    cached.write_text("downloaded", encoding="utf-8")
    destination = tmp_path / "data" / "train.csv"

    monkeypatch.setattr(trainer, "hf_hub_download", lambda **kwargs: str(cached))

    assert (
        trainer.download_file("repo", "remote/train.csv", "main", destination)
        == destination
    )
    assert destination.read_text(encoding="utf-8") == "downloaded"


def test_convert_csv_omits_header_and_preserves_split(tmp_path: Path):
    source = tmp_path / "train.csv"
    source.write_text(
        "label,text,filename,xpath\n"
        "header,Introduction,paper.pdf,/text/1\n"
        "text,Body text,paper.pdf,/text/2\n"
        "reference,A. Author (2024),paper.pdf,/text/3\n",
        encoding="utf-8",
    )
    output = io.StringIO()

    counts, omitted = trainer.convert_csv(source, output, True)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]

    assert counts == {"text": 1, "reference": 1}
    assert omitted == 1
    assert {row["label"] for row in rows} == {"text", "reference"}
    assert all(row["training-sample"] is True for row in rows)


def test_convert_csv_rejects_unknown_label(tmp_path: Path):
    source = tmp_path / "train.csv"
    source.write_text("label,text\ncaption,A caption\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown label"):
        trainer.convert_csv(source, io.StringIO(), True)


def test_semantic_config_uses_native_semantic_model(tmp_path: Path):
    config = trainer.semantic_config(
        tmp_path / "semantic.jsonl",
        tmp_path / "fst_semantic.bin",
        tmp_path / "metrics.txt",
        {
            "autotune": True,
            "duration": 60,
            "model_size": "10M",
            "metric": "f1:reference",
            "predictions": 2,
            "learning_rate": 0.2,
            "epochs": 8,
            "dimension": 128,
            "context_window": 4,
            "ngram": 2,
            "loss": "ova",
            "min_count": 2,
            "min_char_ngram": 2,
            "max_char_ngram": 5,
            "bucket": 1000000,
            "threads": 1,
            "seed": 42,
        },
    )

    assert config["model"] == "semantic"
    assert config["hpo"] == {
        "autotune": True,
        "duration": 60,
        "modelsize": "10M",
        "metric": "f1:reference",
        "predictions": 2,
    }
    assert config["args"] == {
        "learning-rate": 0.2,
        "epoch": 8,
        "dim": 128,
        "ws": 4,
        "n-gram": 2,
        "loss": "ova",
        "min-count": 2,
        "min-char-ngram": 2,
        "max-char-ngram": 5,
        "bucket": 1000000,
        "thread": 1,
        "seed": 42,
    }


def test_semantic_config_omits_null_training_parameters(tmp_path: Path):
    training = dict.fromkeys(trainer.TRAINING_ARGUMENTS)
    config = trainer.semantic_config(
        tmp_path / "semantic.jsonl",
        tmp_path / "fst_semantic.bin",
        tmp_path / "metrics.txt",
        training,
    )

    assert config["args"] == {}


def test_runner_extracts_label_and_confidence():
    class Model:
        def apply_on_text(self, text):
            return {
                "properties": {
                    "headers": ["type", "label", "confidence"],
                    "data": [["semantic", "meta-data", 0.91]],
                }
            }

    assert runner.classify(Model(), "IBM Research") == {
        "label": "meta-data",
        "confidence": 0.91,
        "text": "IBM Research",
    }


def test_input_texts_ignores_blank_stdin_lines():
    assert runner.input_texts(None, ["first\n", "\n", "second\n"]) == [
        "first",
        "second",
    ]


def test_document_runner_round_trips_text_items_with_xpaths(tmp_path: Path):
    path = tmp_path / "document.dclx"
    document = document_runner.DocLangXDocument()
    assert document.read_xml(
        '<doclang version="0.7">'
        "<text>First item</text>"
        "<picture><text>Picture text</text></picture>"
        "<table><text>Table text</text></table>"
        "<text>Second item</text>"
        "</doclang>"
    )
    assert document.write(str(path))

    assert document_runner.read_dclx_text_items(path) == [
        ("/doclang[1]/text[1]", "First item"),
        ("/doclang[1]/text[2]", "Second item"),
    ]


def test_document_runner_uses_native_docling_dclx_export(tmp_path, monkeypatch):
    source = tmp_path / "document.pdf"
    destination = tmp_path / "nested" / "document.dclx"
    exported = []

    class Document:
        def save_as_doclang_archive(self, path):
            exported.append(path)
            path.write_bytes(b"native dclx")

    class Conversion:
        document = Document()

    class Converter:
        def convert(self, path):
            assert path == source
            return Conversion()

    monkeypatch.setattr(document_runner, "docling_converter", Converter)

    assert document_runner.convert_pdf(source, destination) == destination
    assert exported == [destination]
    assert destination.read_bytes() == b"native dclx"


def test_document_runner_disables_pdf_ocr_and_table_structure():
    from docling.datamodel.base_models import InputFormat
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    converter = document_runner.docling_converter()
    format_option = converter.format_to_options[InputFormat.PDF]
    options = format_option.pipeline_options

    assert format_option.pipeline_cls is StandardPdfPipeline
    assert options.do_ocr is False
    assert options.do_table_structure is False
    assert options.generate_page_images is True


def test_document_runner_predicts_each_item():
    class Model:
        def apply_on_text(self, item_text):
            label = "reference" if item_text.startswith("[") else "text"
            return {
                "properties": {
                    "headers": ["type", "label", "confidence"],
                    "data": [["semantic", label, 0.95]],
                }
            }

    assert document_runner.predict_items(
        Model(),
        [
            ("/doclang[1]/text[1]", "Body"),
            ("/doclang[1]/text[2]", "[1] Citation"),
        ],
    ) == [
        {
            "xpath": "/doclang[1]/text[1]",
            "label": "text",
            "confidence": 0.95,
            "text": "Body",
        },
        {
            "xpath": "/doclang[1]/text[2]",
            "label": "reference",
            "confidence": 0.95,
            "text": "[1] Citation",
        },
    ]


def test_document_runner_keeps_items_without_a_prediction():
    class Model:
        def apply_on_text(self, item_text):
            return {}

    assert document_runner.classify(Model(), "1952") == (
        "no-prediction",
        0.0,
    )


def test_document_runner_reports_stale_native_extension(tmp_path, monkeypatch):
    class InvalidModel:
        def apply_on_text(self, item_text):
            return {}

    monkeypatch.setattr(
        document_runner, "init_nlp_model", lambda *args, **kwargs: InvalidModel()
    )

    with pytest.raises(RuntimeError, match=r"uv run python local_build\.py"):
        document_runner.load_model(tmp_path / "fst_semantic.bin")


def test_document_runner_arguments_have_document_defaults():
    args = document_runner.arguments(["document.pdf"])

    assert args.input == Path("document.pdf")
    assert args.model == document_runner.DEFAULT_MODEL_PATH
    assert args.output_dclx is None
    assert args.format == "table"
    assert not hasattr(args, "ocr")
