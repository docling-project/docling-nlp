import json

import pytest
from test_model_info import (
    write_crf_model,
    write_fasttext_model,
    write_model_info,
    write_rgx_model,
)
from typer.testing import CliRunner

import docling_nlp.cli.models as cli
import docling_nlp.utils.load_pretrained_models as loader
import docling_nlp.utils.model_info as model_info

MODEL_FILES = {
    "language": (
        "fst_language.bin",
        "models/fasttext/language/fst_language.bin",
    ),
    "material": ("crf_material.bin", "models/crf/ucmi/crf_material.bin"),
    "geoloc": ("rgx_geoloc.json", "models/rgx/geoloc/rgx_geoloc.json"),
}

runner = CliRunner()


@pytest.fixture
def resources_dir(tmp_path, monkeypatch):
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "huggingface": {
                    "repo-id": "docling-project/docling-nlp-models",
                    "revision": "main",
                },
                "nlp": {"trained-models": MODEL_FILES},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(cli, "get_resources_dir", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    calls = []

    def fake_download(**kwargs):
        # the repository holds no label descriptions (yet), as in the default case
        if kwargs["filename"].endswith(loader.MODEL_INFO_SUFFIX):
            raise FileNotFoundError(kwargs["filename"])
        calls.append(kwargs)
        cached_file = tmp_path / f"cached-{len(calls)}"
        cached_file.write_bytes(kwargs["filename"].encode())
        return str(cached_file)

    monkeypatch.setattr(loader, "hf_hub_download", fake_download)
    return calls


def test_list_shows_models_and_download_state(resources_dir):
    target = resources_dir / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0
    assert "language" in result.output
    assert "fasttext" in result.output


def test_download_without_arguments_fetches_all_models(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download"])

    assert result.exit_code == 0
    assert [call["filename"] for call in downloads] == [
        MODEL_FILES["language"][1],
        MODEL_FILES["material"][1],
        MODEL_FILES["geoloc"][1],
    ]


def test_download_selects_named_models(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "geoloc"])

    assert result.exit_code == 0
    assert [call["filename"] for call in downloads] == [MODEL_FILES["geoloc"][1]]
    assert (resources_dir / MODEL_FILES["geoloc"][1]).exists()


def test_download_selects_by_kind(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "--kind", "crf"])

    assert result.exit_code == 0
    assert [call["filename"] for call in downloads] == [MODEL_FILES["material"][1]]


def test_download_writes_to_output_dir(resources_dir, downloads, tmp_path):
    output_dir = tmp_path / "artifacts"

    result = runner.invoke(
        cli.app, ["download", "geoloc", "--output-dir", str(output_dir)]
    )

    assert result.exit_code == 0
    assert (output_dir / MODEL_FILES["geoloc"][1]).exists()
    assert "DOCLING_NLP_RESOURCES_DIR" in result.output


def test_download_quiet_prints_only_the_directory(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "geoloc", "--quiet"])

    assert result.exit_code == 0
    assert result.output.strip() == str(resources_dir)


def test_download_rejects_unknown_model(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "nosuchmodel"])

    assert result.exit_code != 0
    assert downloads == []


def test_download_rejects_unknown_kind(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "--kind", "nosuchkind"])

    assert result.exit_code != 0
    assert downloads == []


def test_download_rejects_models_together_with_all(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "geoloc", "--all"])

    assert result.exit_code != 0
    assert downloads == []


def test_download_rejects_models_together_with_kind(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "geoloc", "--kind", "rgx"])

    assert result.exit_code != 0
    assert downloads == []


def test_download_copies_support_resources_into_output_dir(
    resources_dir, downloads, tmp_path
):
    (resources_dir / "confusables").mkdir()
    (resources_dir / "confusables" / "confusablesRestricted.txt").write_text("x")
    output_dir = tmp_path / "artifacts"

    result = runner.invoke(
        cli.app, ["download", "geoloc", "--output-dir", str(output_dir)]
    )

    assert result.exit_code == 0
    assert (output_dir / "confusables" / "confusablesRestricted.txt").exists()
    assert (output_dir / "models.json").exists()


def test_download_no_standalone_skips_support_resources(
    resources_dir, downloads, tmp_path
):
    (resources_dir / "confusables").mkdir()
    (resources_dir / "confusables" / "confusablesRestricted.txt").write_text("x")
    output_dir = tmp_path / "artifacts"

    result = runner.invoke(
        cli.app,
        ["download", "geoloc", "--output-dir", str(output_dir), "--no-standalone"],
    )

    assert result.exit_code == 0
    assert (output_dir / MODEL_FILES["geoloc"][1]).exists()
    assert not (output_dir / "confusables").exists()


def test_download_into_resources_dir_does_not_copy_support_resources(
    resources_dir, downloads
):
    result = runner.invoke(cli.app, ["download", "geoloc"])

    assert result.exit_code == 0
    assert "DOCLING_NLP_RESOURCES_DIR" not in result.output


def test_info_describes_a_downloaded_model(resources_dir):
    write_crf_model(
        resources_dir / MODEL_FILES["material"][1],
        ["B_simple_chemical", "I_simple_chemical"],
    )

    result = runner.invoke(cli.app, ["info", "material"])

    assert result.exit_code == 0
    assert "material" in result.output
    assert "crf" in result.output
    assert "B_simple_chemical" in result.output
    assert "yes" in result.output


def test_info_prints_json(resources_dir):
    write_rgx_model(
        resources_dir / MODEL_FILES["geoloc"][1], [("geoloc", "country", "Belgium")]
    )

    result = runner.invoke(cli.app, ["info", "geoloc", "--json"])

    assert result.exit_code == 0

    data = json.loads(result.output)
    assert data["name"] == "geoloc"
    assert data["kind"] == "rgx"
    assert data["downloaded"] is True
    assert data["labels"] == [
        {"label": "geoloc/country", "description": "", "count": 1}
    ]


def test_info_of_a_missing_artifact_points_at_the_download(resources_dir):
    write_model_info(
        model_info.model_info_path(resources_dir / MODEL_FILES["language"][1]),
        {"en": "English"},
        summary="Language identification of a paragraph or a table.",
    )

    result = runner.invoke(cli.app, ["info", "language", "--offline"])

    assert result.exit_code == 0
    assert "English" in result.output
    assert "the artifact is not downloaded" in result.output
    assert "models download language" in result.output


def test_info_reads_the_artifact_from_the_output_dir(resources_dir, tmp_path):
    models_dir = tmp_path / "artifacts"
    write_fasttext_model(models_dir / MODEL_FILES["language"][1], ["en", "de"])

    result = runner.invoke(
        cli.app, ["info", "language", "--output-dir", str(models_dir), "--json"]
    )

    assert result.exit_code == 0
    assert [label["label"] for label in json.loads(result.output)["labels"]] == [
        "en",
        "de",
    ]


def test_info_rejects_unknown_models(resources_dir):
    result = runner.invoke(cli.app, ["info", "nosuchmodel", "--offline"])

    assert result.exit_code != 0


def test_info_shows_the_label_descriptions(resources_dir):
    artifact = resources_dir / MODEL_FILES["language"][1]
    write_fasttext_model(artifact, ["en", "de"])
    write_model_info(
        model_info.model_info_path(artifact),
        {"en": "English", "de": "German"},
    )

    result = runner.invoke(cli.app, ["info", "language"])

    assert result.exit_code == 0
    assert "English" in result.output
    assert "German" in result.output
    assert "no label descriptions were found" not in result.output


def test_info_without_label_descriptions_says_where_they_go(resources_dir):
    write_fasttext_model(resources_dir / MODEL_FILES["language"][1], ["en"])

    result = runner.invoke(cli.app, ["info", "language"])

    assert result.exit_code == 0
    assert "no label descriptions were found" in result.output
    assert "fst_language.info.json" in result.output


def test_info_of_a_missing_artifact_without_a_description(resources_dir):
    result = runner.invoke(cli.app, ["info", "language", "--offline"])

    assert result.exit_code == 0
    assert "the artifact is not downloaded" in result.output
    assert "download" in result.output


def stamp_revision(resources_dir, name, revision):
    """Record that an artifact was downloaded from `revision`, as the loader does."""

    lock = loader.read_models_lock(resources_dir) or {"models": {}}
    lock.setdefault("models", {})[name] = {
        "repo-id": "docling-project/docling-nlp-models",
        "revision": revision,
        "relative-path": MODEL_FILES[name][1],
        "size-bytes": None,
    }
    loader.write_models_lock(resources_dir, lock)


def test_list_marks_an_artifact_of_another_revision_as_stale(resources_dir):
    target = resources_dir / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    stamp_revision(resources_dir, "geoloc", "an-older-revision")

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0
    assert "stale" in result.output
    assert "models download geoloc" in result.output.replace("\n", "")


def test_list_marks_an_unstamped_artifact_as_stale(resources_dir):
    target = resources_dir / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0
    assert "stale" in result.output


def test_list_does_not_mark_a_stamped_artifact_as_stale(resources_dir):
    target = resources_dir / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    stamp_revision(resources_dir, "geoloc", "main")

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0
    assert "stale" not in result.output


def test_info_reports_the_local_revision(resources_dir):
    write_rgx_model(resources_dir / MODEL_FILES["geoloc"][1], [("city", "name", "x")])
    stamp_revision(resources_dir, "geoloc", "main")

    result = runner.invoke(cli.app, ["info", "geoloc"])

    assert result.exit_code == 0
    assert "local revision" in result.output
    assert "stale" not in result.output


def test_info_flags_a_stale_artifact(resources_dir):
    write_rgx_model(resources_dir / MODEL_FILES["geoloc"][1], [("city", "name", "x")])
    stamp_revision(resources_dir, "geoloc", "an-older-revision")

    result = runner.invoke(cli.app, ["info", "geoloc"])

    assert result.exit_code == 0
    assert "an-older-revision" in result.output
    assert "stale" in result.output


def test_download_stamps_the_revision(resources_dir, downloads):
    result = runner.invoke(cli.app, ["download", "geoloc"])

    assert result.exit_code == 0
    lock = loader.read_models_lock(resources_dir)
    assert lock["models"]["geoloc"]["revision"] == "main"


def test_download_refreshes_a_stale_artifact_without_force(resources_dir, downloads):
    target = resources_dir / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    stamp_revision(resources_dir, "geoloc", "an-older-revision")

    result = runner.invoke(cli.app, ["download", "geoloc"])

    assert result.exit_code == 0
    assert [call["filename"] for call in downloads] == [MODEL_FILES["geoloc"][1]]
    assert target.read_bytes() == MODEL_FILES["geoloc"][1].encode()
    assert loader.read_models_lock(resources_dir)["models"]["geoloc"]["revision"] == (
        "main"
    )
