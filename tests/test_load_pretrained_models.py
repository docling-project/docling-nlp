import json
from pathlib import Path

import pytest

import docling_nlp.utils.load_pretrained_models as loader

MODEL_FILES = {
    "language": (
        "fst_language.bin",
        "models/fasttext/language/fst_language.bin",
    ),
    "geoloc": ("rgx_geoloc.json", "models/rgx/geoloc/rgx_geoloc.json"),
}


def info_path(relative_path: str) -> str:
    """Path of the description belonging to a model artifact."""

    return Path(relative_path).with_suffix(loader.MODEL_INFO_SUFFIX).as_posix()


def stamp_revision(resources_dir: Path, name: str, revision: str):
    """Record that an artifact was downloaded from `revision`, as the loader does."""

    lock = loader.read_models_lock(resources_dir) or {"models": {}}
    lock.setdefault("models", {})[name] = {
        "repo-id": "docling-project/docling-nlp-models",
        "revision": revision,
        "relative-path": MODEL_FILES[name][1],
        "size-bytes": None,
    }
    loader.write_models_lock(resources_dir, lock)


def write_models_config(resources_dir: Path):
    (resources_dir / "models.json").write_text(
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


def test_load_pretrained_models_downloads_to_configured_paths(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    calls = []

    def fake_download(**kwargs):
        # the repository holds no label descriptions (yet), as in the default case
        if kwargs["filename"].endswith(loader.MODEL_INFO_SUFFIX):
            raise FileNotFoundError(kwargs["filename"])
        calls.append(kwargs)
        cached_file = tmp_path / f"cached-{len(calls)}"
        cached_file.write_bytes(kwargs["filename"].encode())
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    result = loader.load_pretrained_nlp_models()

    assert result == ["language", "geoloc"]
    assert (tmp_path / MODEL_FILES["language"][1]).read_bytes() == (
        MODEL_FILES["language"][1].encode()
    )
    assert calls[0]["repo_id"] == "docling-project/docling-nlp-models"
    assert calls[0]["revision"] == "main"
    assert [call["filename"] for call in calls] == [
        MODEL_FILES["language"][1],
        MODEL_FILES["geoloc"][1],
    ]


def test_load_pretrained_models_reuses_existing_files(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    target = tmp_path / MODEL_FILES["language"][1]
    geoloc_target = tmp_path / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    geoloc_target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    geoloc_target.write_bytes(b"existing")
    stamp_revision(tmp_path, "language", "main")
    stamp_revision(tmp_path, "geoloc", "main")
    calls = []

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        loader, "hf_hub_download", lambda **kwargs: calls.append(kwargs)
    )

    result = loader.load_pretrained_nlp_models()

    assert result == ["language", "geoloc"]
    # the artifacts are stamped with the configured revision, so only their
    # descriptions are looked up
    assert [call["filename"] for call in calls] == [
        info_path(MODEL_FILES["language"][1]),
        info_path(MODEL_FILES["geoloc"][1]),
    ]
    assert target.read_bytes() == b"existing"


def test_load_pretrained_models_force_refreshes_existing_files(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    target = tmp_path / MODEL_FILES["language"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"refreshed")
        assert kwargs["force_download"] is True
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.load_pretrained_nlp_models(force=True)

    assert target.read_bytes() == b"refreshed"


def test_load_pretrained_models_reports_download_failures(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        loader,
        "hf_hub_download",
        lambda **kwargs: (_ for _ in ()).throw(OSError("no network")),
    )

    with pytest.raises(RuntimeError, match="Failed to download NLP model 'language'"):
        loader.load_pretrained_nlp_models()


def test_list_pretrained_models_reports_name_kind_and_path(tmp_path):
    write_models_config(tmp_path)

    specs = loader.list_pretrained_nlp_models(tmp_path)

    assert [(spec.name, spec.kind) for spec in specs] == [
        ("language", "fasttext"),
        ("geoloc", "rgx"),
    ]
    assert specs[0].filename == "fst_language.bin"
    assert specs[0].target(tmp_path) == tmp_path / MODEL_FILES["language"][1]


def test_download_pretrained_models_selects_by_name(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    calls = []

    def fake_download(**kwargs):
        # the repository holds no label descriptions (yet), as in the default case
        if kwargs["filename"].endswith(loader.MODEL_INFO_SUFFIX):
            raise FileNotFoundError(kwargs["filename"])
        calls.append(kwargs)
        cached_file = tmp_path / f"cached-{len(calls)}"
        cached_file.write_bytes(kwargs["filename"].encode())
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    result = loader.download_pretrained_nlp_models(names=["geoloc"])

    assert result == ["geoloc"]
    assert [call["filename"] for call in calls] == [MODEL_FILES["geoloc"][1]]
    assert not (tmp_path / MODEL_FILES["language"][1]).exists()


def test_download_pretrained_models_writes_to_output_dir(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    output_dir = tmp_path / "elsewhere"

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"payload")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"], output_dir=output_dir)

    assert (output_dir / MODEL_FILES["geoloc"][1]).read_bytes() == b"payload"
    assert not (tmp_path / MODEL_FILES["geoloc"][1]).exists()


def test_download_pretrained_models_rejects_unknown_names(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))

    with pytest.raises(ValueError, match="Unknown NLP model\\(s\\): nosuchmodel"):
        loader.download_pretrained_nlp_models(names=["nosuchmodel"])


def test_copy_support_resources_skips_training_data_and_downloadables(tmp_path):
    write_models_config(tmp_path)
    (tmp_path / "confusables").mkdir()
    (tmp_path / "confusables" / "confusablesRestricted.txt").write_text("x")
    (tmp_path / "models" / "rgx" / "vau").mkdir(parents=True)
    (tmp_path / "models" / "rgx" / "vau" / "units.jsonl").write_text("{}")
    (tmp_path / "data" / "text").mkdir(parents=True)
    (tmp_path / "data" / "text" / "huge.jsonl").write_text("{}")
    downloadable = tmp_path / MODEL_FILES["language"][1]
    downloadable.parent.mkdir(parents=True)
    downloadable.write_bytes(b"model")

    output_dir = tmp_path / "standalone"
    copied = loader.copy_support_resources(output_dir, resources_dir=tmp_path)

    assert {path.relative_to(output_dir).as_posix() for path in copied} == {
        "models.json",
        "confusables/confusablesRestricted.txt",
        "models/rgx/vau/units.jsonl",
    }
    assert not (output_dir / "data" / "text" / "huge.jsonl").exists()
    assert not (output_dir / MODEL_FILES["language"][1]).exists()


def test_copy_support_resources_is_a_noop_for_the_resources_dir(tmp_path):
    write_models_config(tmp_path)

    assert loader.copy_support_resources(tmp_path, resources_dir=tmp_path) == []


def test_download_fetches_the_label_descriptions_along(tmp_path, monkeypatch):
    write_models_config(tmp_path)

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_text(kwargs["filename"], encoding="utf-8")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])

    sidecar = tmp_path / "models/rgx/geoloc/rgx_geoloc.info.json"
    assert (
        sidecar.read_text(encoding="utf-8") == "models/rgx/geoloc/rgx_geoloc.info.json"
    )


def test_download_tolerates_missing_label_descriptions(tmp_path, monkeypatch):
    write_models_config(tmp_path)

    def fake_download(**kwargs):
        if kwargs["filename"].endswith(loader.MODEL_INFO_SUFFIX):
            raise FileNotFoundError(kwargs["filename"])
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"payload")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    result = loader.download_pretrained_nlp_models(names=["geoloc"])

    assert result == ["geoloc"]
    assert (tmp_path / MODEL_FILES["geoloc"][1]).read_bytes() == b"payload"
    assert not (tmp_path / "models/rgx/geoloc/rgx_geoloc.info.json").exists()


def test_download_keeps_the_label_descriptions_of_a_reused_artifact(
    tmp_path, monkeypatch
):
    write_models_config(tmp_path)
    target = tmp_path / MODEL_FILES["geoloc"][1]
    sidecar = tmp_path / info_path(MODEL_FILES["geoloc"][1])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    sidecar.write_text("{}", encoding="utf-8")
    stamp_revision(tmp_path, "geoloc", "main")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["filename"])
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"payload")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])

    # nothing is fetched: the artifact is stamped and its description is there
    assert calls == []
    assert target.read_bytes() == b"existing"
    assert sidecar.read_text(encoding="utf-8") == "{}"


def test_model_spec_points_at_its_label_descriptions(tmp_path):
    write_models_config(tmp_path)

    spec = {spec.name: spec for spec in loader.list_pretrained_nlp_models(tmp_path)}[
        "language"
    ]

    assert spec.info_relative_path == (
        "models/fasttext/language/fst_language.info.json"
    )
    assert spec.info_target(tmp_path) == (
        tmp_path / "models/fasttext/language/fst_language.info.json"
    )


def test_download_stamps_the_revision_it_downloaded_from(tmp_path, monkeypatch):
    write_models_config(tmp_path)

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"payload")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])

    lock = loader.read_models_lock(tmp_path)
    assert lock["models"]["geoloc"] == {
        "repo-id": "docling-project/docling-nlp-models",
        "revision": "main",
        "relative-path": MODEL_FILES["geoloc"][1],
        "size-bytes": len(b"payload"),
    }
    # the models that were not downloaded are not stamped
    assert "language" not in lock["models"]


def test_download_skips_the_artifacts_stamped_with_the_configured_revision(
    tmp_path, monkeypatch
):
    write_models_config(tmp_path)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs["filename"])
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"payload")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])
    calls.clear()

    loader.download_pretrained_nlp_models(names=["geoloc"])

    # the artifact is stamped with the configured revision and its description is
    # already there, so nothing is fetched at all
    assert calls == []


def test_download_refreshes_the_artifacts_of_another_revision(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    target = tmp_path / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old")
    loader.write_models_lock(
        tmp_path,
        {
            "models": {
                "geoloc": {
                    "repo-id": "docling-project/docling-nlp-models",
                    "revision": "an-older-revision",
                    "relative-path": MODEL_FILES["geoloc"][1],
                    "size-bytes": 3,
                }
            }
        },
    )

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"new")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])

    assert target.read_bytes() == b"new"
    assert loader.read_models_lock(tmp_path)["models"]["geoloc"]["revision"] == "main"


def test_download_refreshes_the_artifacts_that_are_not_stamped(tmp_path, monkeypatch):
    write_models_config(tmp_path)
    target = tmp_path / MODEL_FILES["geoloc"][1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"of an unknown revision")

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_bytes(b"new")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])

    assert target.read_bytes() == b"new"
    assert loader.read_models_lock(tmp_path)["models"]["geoloc"]["revision"] == "main"


def test_download_refreshes_the_description_along_with_the_artifact(
    tmp_path, monkeypatch
):
    write_models_config(tmp_path)
    target = tmp_path / MODEL_FILES["geoloc"][1]
    sidecar = tmp_path / info_path(MODEL_FILES["geoloc"][1])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"of an unknown revision")
    sidecar.write_text("stale", encoding="utf-8")

    def fake_download(**kwargs):
        cached_file = tmp_path / "cached"
        cached_file.write_text(f"fresh {kwargs['filename']}", encoding="utf-8")
        return str(cached_file)

    monkeypatch.setattr(loader, "get_resources_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "hf_hub_download", fake_download)

    loader.download_pretrained_nlp_models(names=["geoloc"])

    assert sidecar.read_text(encoding="utf-8") == (
        f"fresh {info_path(MODEL_FILES['geoloc'][1])}"
    )


def test_locked_revision_ignores_a_stamp_of_another_repository(tmp_path):
    lock = {
        "models": {"geoloc": {"repo-id": "someone-else/models", "revision": "abcdef"}}
    }

    assert loader.locked_model_revision(lock, "geoloc", "someone-else/models") == (
        "abcdef"
    )
    assert loader.locked_model_revision(lock, "geoloc", "docling-project/x") is None
    assert loader.locked_model_revision(lock, "nosuchmodel", "someone-else/models") is (
        None
    )


def test_read_models_lock_tolerates_a_broken_or_foreign_lock(tmp_path):
    assert loader.read_models_lock(tmp_path) == {}

    loader.models_lock_path(tmp_path).write_text("not json", encoding="utf-8")
    assert loader.read_models_lock(tmp_path) == {}

    loader.models_lock_path(tmp_path).write_text(
        json.dumps({"version": 99, "models": {"geoloc": {}}}), encoding="utf-8"
    )
    assert loader.read_models_lock(tmp_path) == {}


def test_copy_support_resources_leaves_the_lock_behind(tmp_path):
    write_models_config(tmp_path)
    loader.write_models_lock(tmp_path, {"models": {"geoloc": {"revision": "main"}}})

    output_dir = tmp_path / "standalone"
    copied = loader.copy_support_resources(output_dir, resources_dir=tmp_path)

    assert {path.name for path in copied} == {"models.json"}
    assert not loader.models_lock_path(output_dir).exists()
