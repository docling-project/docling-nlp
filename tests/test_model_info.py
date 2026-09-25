import json
import struct

import pytest

from docling_nlp.utils.load_pretrained_models import (
    read_models_lock,
    write_models_lock,
)
from docling_nlp.utils.model_info import (
    describe_nlp_model,
    format_size,
    model_info_path,
    read_crf_labels,
    read_fasttext_labels,
    read_model_info,
    read_model_labels,
    read_rgx_labels,
)


def label_names(labels):
    return [label.name for label in labels]


def write_model_info(path, labels, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": labels, **fields}), encoding="utf-8")


REPO_ID = "docling-project/docling-nlp-models"

MODEL_FILES = {
    "language": (
        "fst_language.bin",
        "models/fasttext/language/fst_language.bin",
    ),
    "material": ("crf_material.bin", "models/crf/ucmi/crf_material.bin"),
    "geoloc": ("rgx_geoloc.json", "models/rgx/geoloc/rgx_geoloc.json"),
}


def write_fasttext_model(path, labels, words=("hello", "world")):
    """Write a minimal fastText model file: magic, args and the dictionary."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fw:
        fw.write(struct.pack("<ii", 793712314, 12))
        # dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn,
        # maxn, lrUpdateRate
        fw.write(struct.pack("<12i", 16, 5, 10, 1, 5, 1, 3, 3, 2000, 2, 4, 100))
        fw.write(struct.pack("<d", 0.0001))

        entries = [(word, 0) for word in words] + [
            (f"__label__{label}", 1) for label in labels
        ]

        fw.write(struct.pack("<iii", len(entries), len(words), len(labels)))
        fw.write(struct.pack("<qq", 123, -1))

        for word, entry_type in entries:
            fw.write(word.encode("utf-8") + b"\x00")
            fw.write(struct.pack("<qb", 7, entry_type))


def write_crf_model(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{label}\tW0_word-{i}\t1.5" for i, label in enumerate(labels)]
    lines.append("!BOS!\tW0_word\t0.5")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rgx_model(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "Geographic Locations",
                "description": "a list of locations",
                "headers": ["type", "subtype", "expression", "related-to"],
                "data": [[row[0], row[1], row[2], {}] for row in rows],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def resources_dir(tmp_path):
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "huggingface": {"repo-id": REPO_ID, "revision": "main"},
                "nlp": {"trained-models": MODEL_FILES},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_read_fasttext_labels(tmp_path):
    path = tmp_path / "fst.bin"
    write_fasttext_model(path, ["en", "de", "fr"])

    labels, details = read_fasttext_labels(path)

    assert label_names(labels) == ["en", "de", "fr"]
    assert labels[0].count == 7
    assert details["training mode"] == "supervised"
    assert details["dimension"] == "16"
    assert details["vocabulary"] == "2 word(s), 3 label(s)"


def test_read_fasttext_labels_rejects_other_files(tmp_path):
    path = tmp_path / "fst.bin"
    path.write_bytes(b"not a fasttext model at all")

    with pytest.raises(ValueError):
        read_fasttext_labels(path)


def test_read_crf_labels_drops_the_sentinels(tmp_path):
    path = tmp_path / "crf.bin"
    write_crf_model(path, ["title", "authors", "title"])

    labels, details = read_crf_labels(path)

    assert label_names(labels) == ["authors", "title"]
    assert [label.count for label in labels] == [1, 2]
    assert details["feature weights"] == "4"


def test_read_rgx_labels_counts_the_expressions(tmp_path):
    path = tmp_path / "rgx.json"
    write_rgx_model(
        path,
        [
            ("geoloc", "country", "Belgium"),
            ("geoloc", "country", "France"),
            ("geoloc", "continent", "Europe"),
        ],
    )

    labels, details = read_rgx_labels(path)

    assert label_names(labels) == ["geoloc/continent", "geoloc/country"]
    assert [label.count for label in labels] == [1, 2]
    assert details["expressions"] == "3"
    assert details["asset"] == "a list of locations"


def test_read_model_labels_reports_unreadable_artifacts(tmp_path):
    path = tmp_path / "rgx.json"
    path.write_text("{not json", encoding="utf-8")

    labels, details = read_model_labels("rgx", path)

    assert labels == []
    assert "warning" in details


def test_read_model_labels_ignores_unknown_kinds(tmp_path):
    assert read_model_labels("nosuchkind", tmp_path / "missing") == ([], {})


def test_describe_reads_the_artifact(resources_dir):
    artifact = resources_dir / MODEL_FILES["material"][1]
    write_crf_model(artifact, ["B_simple_chemical", "I_simple_chemical"])
    write_model_info(
        model_info_path(artifact),
        {"B_simple_chemical": "First word-token of a simple chemical."},
        summary="Detector of chemical and material mentions in text.",
        task="sequence labelling (CRF, BIO tagging)",
    )

    info = describe_nlp_model("material", resources_dir=resources_dir)

    assert info.kind == "crf"
    assert info.downloaded
    assert label_names(info.labels) == ["B_simple_chemical", "I_simple_chemical"]
    assert info.label_count_header == "feature weights"
    assert info.size_bytes > 0
    assert info.summary == "Detector of chemical and material mentions in text."
    assert info.task == "sequence labelling (CRF, BIO tagging)"
    assert info.repo_id == "docling-project/docling-nlp-models"


def test_describe_of_a_missing_artifact_stays_offline(resources_dir):
    info = describe_nlp_model("language", resources_dir=resources_dir, remote=False)

    assert not info.downloaded
    assert info.labels == ()
    assert info.size_bytes is None
    assert info.remote_size_bytes is None


def test_describe_looks_up_the_artifact_in_the_models_dir(resources_dir, tmp_path):
    models_dir = tmp_path / "artifacts"
    write_fasttext_model(models_dir / MODEL_FILES["language"][1], ["en", "de"])

    info = describe_nlp_model(
        "language", resources_dir=resources_dir, models_dir=models_dir
    )

    assert info.downloaded
    assert label_names(info.labels) == ["en", "de"]
    assert info.path == models_dir / MODEL_FILES["language"][1]


def test_describe_rejects_unknown_models(resources_dir):
    with pytest.raises(ValueError, match="Unknown NLP model"):
        describe_nlp_model("nosuchmodel", resources_dir=resources_dir, remote=False)


def test_describe_serialises_to_json(resources_dir):
    write_rgx_model(
        resources_dir / MODEL_FILES["geoloc"][1], [("geoloc", "country", "Belgium")]
    )

    data = describe_nlp_model("geoloc", resources_dir=resources_dir).to_dict()

    assert json.loads(json.dumps(data))["name"] == "geoloc"
    assert data["labels"] == [
        {"label": "geoloc/country", "description": "", "count": 1}
    ]
    assert data["huggingface"]["repo-id"] == "docling-project/docling-nlp-models"


@pytest.mark.parametrize(
    "size_bytes,expected",
    [
        (None, "unknown"),
        (0, "0 B"),
        (999, "999 B"),
        (2048, "2.0 KB"),
        (10**7, "9.5 MB"),
    ],
)
def test_format_size(size_bytes, expected):
    assert format_size(size_bytes) == expected


def test_label_descriptions_sit_next_to_the_artifact(tmp_path):
    assert model_info_path(tmp_path / "models" / "fst_language.bin") == (
        tmp_path / "models" / "fst_language.info.json"
    )


def test_read_model_info_accepts_both_forms(tmp_path):
    path = tmp_path / "labels.info.json"
    path.write_text(
        json.dumps(
            {
                "labels": {
                    "en": "English",
                    "de": {"description": "German", "note": "ignored"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert read_model_info(path)["labels"] == {"en": "English", "de": "German"}


@pytest.mark.parametrize("content", ["{not json", "[]"])
def test_read_model_info_tolerates_broken_files(tmp_path, content):
    path = tmp_path / "labels.info.json"
    path.write_text(content, encoding="utf-8")

    assert read_model_info(path) == {}


def test_read_model_info_of_a_missing_file(tmp_path):
    assert read_model_info(tmp_path / "missing.info.json") == {}


def test_describe_merges_the_label_descriptions(resources_dir):
    artifact = resources_dir / MODEL_FILES["language"][1]
    write_fasttext_model(artifact, ["en", "de"])
    write_model_info(
        model_info_path(artifact),
        {"en": "English", "de": "German"},
        **{"label-description": "the languages of the classifier"},
    )

    info = describe_nlp_model("language", resources_dir=resources_dir)

    assert [(label.name, label.description) for label in info.labels] == [
        ("en", "English"),
        ("de", "German"),
    ]
    assert info.label_description == "the languages of the classifier"
    assert info.info_path == model_info_path(artifact)


def test_describe_keeps_labels_that_only_the_descriptions_know(resources_dir):
    artifact = resources_dir / MODEL_FILES["language"][1]
    write_fasttext_model(artifact, ["en"])
    write_model_info(model_info_path(artifact), {"en": "English", "de": "German"})

    info = describe_nlp_model("language", resources_dir=resources_dir)

    assert [(label.name, label.count) for label in info.labels] == [
        ("en", 7),
        ("de", None),
    ]


def test_describe_falls_back_to_the_descriptions_in_the_resources_dir(
    resources_dir, tmp_path
):
    models_dir = tmp_path / "artifacts"
    write_fasttext_model(models_dir / MODEL_FILES["language"][1], ["en"])
    write_model_info(
        model_info_path(resources_dir / MODEL_FILES["language"][1]),
        {"en": "English"},
    )

    info = describe_nlp_model(
        "language", resources_dir=resources_dir, models_dir=models_dir
    )

    assert info.labels[0].description == "English"


def test_describe_without_label_descriptions(resources_dir):
    write_fasttext_model(resources_dir / MODEL_FILES["language"][1], ["en"])

    info = describe_nlp_model("language", resources_dir=resources_dir)

    assert info.info_path is None
    assert info.labels[0].description == ""


def stamp_revision(resources_dir, name, revision, repo_id=REPO_ID):
    """Record that an artifact was downloaded from `revision`, as the loader does."""

    lock = read_models_lock(resources_dir) or {"models": {}}
    lock.setdefault("models", {})[name] = {
        "repo-id": repo_id,
        "revision": revision,
        "relative-path": MODEL_FILES[name][1],
        "size-bytes": None,
    }
    write_models_lock(resources_dir, lock)


def test_describe_reports_the_revision_the_artifact_came_from(resources_dir):
    write_fasttext_model(resources_dir / MODEL_FILES["language"][1], ["en"])
    stamp_revision(resources_dir, "language", "main")

    info = describe_nlp_model("language", resources_dir=resources_dir, remote=False)

    assert info.downloaded
    assert info.revision == "main"
    assert info.local_revision == "main"
    assert not info.stale
    assert info.to_dict()["huggingface"]["local-revision"] == "main"
    assert info.to_dict()["stale"] is False


def test_describe_flags_an_artifact_of_another_revision_as_stale(resources_dir):
    write_fasttext_model(resources_dir / MODEL_FILES["language"][1], ["en"])
    stamp_revision(resources_dir, "language", "an-older-revision")

    info = describe_nlp_model("language", resources_dir=resources_dir, remote=False)

    assert info.local_revision == "an-older-revision"
    assert info.stale
    assert info.to_dict()["stale"] is True


def test_describe_flags_an_unstamped_artifact_as_stale(resources_dir):
    write_fasttext_model(resources_dir / MODEL_FILES["language"][1], ["en"])

    info = describe_nlp_model("language", resources_dir=resources_dir, remote=False)

    assert info.downloaded
    assert info.local_revision is None
    assert info.stale


def test_describe_does_not_call_an_artifact_that_is_missing_stale(resources_dir):
    info = describe_nlp_model("language", resources_dir=resources_dir, remote=False)

    assert not info.downloaded
    assert info.local_revision is None
    assert not info.stale


def test_describe_reads_the_lock_of_the_models_dir(resources_dir, tmp_path):
    models_dir = tmp_path / "elsewhere"
    write_fasttext_model(models_dir / MODEL_FILES["language"][1], ["en"])
    stamp_revision(models_dir, "language", "main")
    # a lock in the resources directory says nothing about the artifact used
    stamp_revision(resources_dir, "language", "an-older-revision")

    info = describe_nlp_model(
        "language", resources_dir=resources_dir, models_dir=models_dir, remote=False
    )

    assert info.local_revision == "main"
    assert not info.stale


def test_describe_ignores_a_stamp_of_another_repository(resources_dir):
    write_fasttext_model(resources_dir / MODEL_FILES["language"][1], ["en"])
    stamp_revision(resources_dir, "language", "main", repo_id="someone-else/models")

    info = describe_nlp_model("language", resources_dir=resources_dir, remote=False)

    assert info.local_revision is None
    assert info.stale
