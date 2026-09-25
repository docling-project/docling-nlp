import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "scripts/selfimprove/crf/references"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_trainig_data_from_grobid")
runner = load_script("run_crf_references")
trainer = load_script("train_crf_references")


def test_parse_namespaced_mixed_content(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><back><listBibl>
<bibl><author>A. Writer</author>. <title level="a">An Article</title>. <title level="j">A Journal</title> <biblScope unit="volume">4</biblScope>:<biblScope unit="page">10-12</biblScope> (<date>2024</date>).</bibl>
</listBibl></back></text></TEI>""",
        encoding="utf-8",
    )

    records = prepare.parse_xml(xml)

    assert len(records) == 1
    record = records[0]
    assert record["text"] == "A. Writer. An Article. A Journal 4:10-12 (2024)."
    assert [annotation["label"] for annotation in record["annotation"]] == [
        "authors",
        "title",
        "journal",
        "volume",
        "pages",
        "date",
    ]
    for annotation in record["annotation"]:
        assert record["text"][annotation["start"] : annotation["end"]].strip()


def test_split_files_is_deterministic_and_disjoint(tmp_path: Path):
    paths = [tmp_path / f"{number}.xml" for number in range(10)]
    first = prepare.split_files(paths, validation_ratio=0.2, seed=7)
    second = prepare.split_files(list(reversed(paths)), validation_ratio=0.2, seed=7)

    assert first == second
    assert not first[0] & first[1]
    assert len(first[0]) == 8
    assert len(first[1]) == 2


def test_raw_file_url_preserves_literal_percent_escape():
    url = prepare.raw_file_url(
        "abc123", "10.1007%2Fexample.training.references.tei.xml"
    )

    assert url.endswith("/10.1007%252Fexample.training.references.tei.xml")


def test_load_config_validates_training_values(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """data:
  train_file: train.jsonl
  validation_file: validation.jsonl
training:
  epochs: 0
output: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="epochs"):
        trainer.load_config(config)


def test_validate_annotations_rejects_overlap():
    row = {
        "text": "abcdef",
        "annotation": [
            {"label": "title", "start": 0, "end": 4},
            {"label": "date", "start": 3, "end": 6},
        ],
    }

    with pytest.raises(ValueError, match="overlapping"):
        trainer.validate_annotations(row)


def test_prepare_record_aligns_offsets_after_tokenizer_deletion():
    class NormalizingTokenizer:
        def apply_on_text(self, text):
            normalized = text.replace("¨", "")
            start = len(normalized[: normalized.index("title")].encode("utf-8"))
            end = len(normalized.encode("utf-8"))
            return {
                "text": normalized,
                "word_tokens": {
                    "headers": ["char_i", "char_j", "word"],
                    "data": [[start, end, "title"]],
                },
            }

    text = "É. ¨ title"
    row = {
        "text": text,
        "annotation": [
            {"label": "Title", "start": text.index("title"), "end": len(text)}
        ],
    }

    prepared = trainer.prepare_record(row, True, NormalizingTokenizer())

    assert prepared["text"] == "É.  title"
    assert prepared["word_tokens"]["data"][0][-1] == "title"


def test_prediction_to_bibtex_maps_fields_and_unicode_character_offsets():
    text = "É. Writer. A title. Journal 4 (2024)."

    def row(token: str, label: str):
        start = text.index(token)
        return [0, token, start, start + len(token), label]

    rows = [
        row("É.", "authors"),
        row("Writer", "authors"),
        row("A title", "title"),
        row("Journal", "journal"),
        row("4", "volume"),
        row("(", "null"),
        row("2024", "date"),
    ]

    assert runner.prediction_to_bibtex(text, rows) == {
        "author": "É. Writer",
        "title": "A title",
        "journal": "Journal",
        "volume": "4",
        "year": "2024",
    }


def test_load_validation_texts(tmp_path: Path):
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        '\n{"text": "First reference"}\n{"text": "Second reference"}\n',
        encoding="utf-8",
    )

    assert runner.load_validation_texts(validation) == [
        "First reference",
        "Second reference",
    ]


def test_arguments_default_to_model_and_validation_dataset():
    args = runner.arguments([])

    assert args.model == runner.DEFAULT_MODEL_PATH
    assert args.text is None
    assert args.validation_file == runner.DEFAULT_VALIDATION_PATH
