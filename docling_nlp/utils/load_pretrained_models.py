#!/usr/bin/env python
"""Module to load binary files of models and data"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from huggingface_hub import hf_hub_download


def get_resources_dir():
    """Function to obtain the resources directory"""

    if "DOCLING_NLP_RESOURCES_DIR" in os.environ:
        resources_dir = os.getenv("DOCLING_NLP_RESOURCES_DIR")
    else:
        from docling_nlp.andromeda_nlp import nlp_model

        model = nlp_model()
        resources_dir = model.get_resources_path()

    return os.path.normpath(resources_dir)


def list_training_data(key: str, force: bool = False, verbose: bool = False):
    """Function to list the training data"""

    return []


def download_items(
    items: Dict[str, Tuple[str, Path]], force: bool = False, verbose: bool = False
) -> Tuple[bool, Dict[str, str]]:
    """
    Iterate through all the items and downloads them.
    The return dictionary will contain the location where items are downloaded.
    If any error occur, the first return value will be false.
    """

    data = {}
    done = True
    for name, (source, target) in items.items():
        if force or (not target.exists()):
            if verbose:
                print(f" -> downloading {name} ... ", end="")

            target.parent.mkdir(exist_ok=True, parents=True)

            with target.open("wb") as fw:
                r = requests.get(source, stream=True)
                if r.ok:
                    for chunk in r.iter_content(chunk_size=8192):
                        fw.write(chunk)
                else:
                    print(f"Error downloading {name}: [{r.status_code}] {r.text}")
                    done = False
                if verbose:
                    print("done!")

            data[name] = target.name
        elif target.exists():
            if verbose:
                print(f" -> already downloaded {name}")
            data[name] = target.name
        else:
            print(f" -> missing {name}")
    return done, data


def load_training_data(
    data_type: str, data_name: str, force: bool = False, verbose: bool = False
):
    """Function to load data to train NLP models"""

    assert data_type in ["text", "crf", "fst"]

    resources_dir = Path(get_resources_dir())

    data_file_path = resources_dir / "data.json"
    with data_file_path.open(encoding="utf-8") as fr:
        training_data = json.load(fr)

    cos_url = training_data["object-store"]
    cos_prfx = training_data["data"]["prefix"]
    cos_path = f"{cos_url}/{cos_prfx}"

    downloads = {}
    for name, files in training_data["data"][data_type].items():
        print(name)
        if name == data_name:
            source = f"{cos_path}/{files[0]}"
            target = resources_dir / str(files[1])

            downloads[name] = (source, target)

    done, data = download_items(downloads)
    return done, data


# Suffix of the (optional) sidecar describing a model and what its labels mean. It
# sits next to the artifact, both in the HuggingFace repository and on disk.
MODEL_INFO_SUFFIX = ".info.json"


@dataclass(frozen=True)
class NlpModelSpec:
    """Description of a single pretrained NLP model artifact."""

    name: str
    kind: str
    filename: str
    relative_path: str

    def target(self, resources_dir) -> Path:
        """Location of the artifact below `resources_dir`."""

        return Path(resources_dir) / self.relative_path

    @property
    def info_relative_path(self) -> str:
        """Path of the description belonging to the artifact."""

        return Path(self.relative_path).with_suffix(MODEL_INFO_SUFFIX).as_posix()

    def info_target(self, resources_dir) -> Path:
        """Location of the description below `resources_dir`."""

        return Path(resources_dir) / self.info_relative_path


# Name of the file stamping which revision every artifact below a directory was
# downloaded from. It sits at the root of the directory the models are written to.
MODELS_LOCK_FILENAME = "models.lock.json"

# Version of the layout of the lock file, so a future change can be recognised.
_MODELS_LOCK_VERSION = 1


def models_lock_path(output_dir) -> Path:
    """Location of the lock file below `output_dir`."""

    return Path(output_dir) / MODELS_LOCK_FILENAME


def read_models_lock(output_dir) -> Dict:
    """
    Function to read the lock file stamping the downloaded artifacts.

    The lock file records, per model, the repository and the revision its artifact
    was downloaded from. It is written by the downloader and is not required to be
    present: a missing, unreadable or unknown-version lock file reads as empty,
    which makes every artifact count as not stamped.
    """

    path = models_lock_path(output_dir)
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as fr:
            lock = json.load(fr)
    except Exception:
        return {}

    if not isinstance(lock, dict):
        return {}
    if lock.get("version") != _MODELS_LOCK_VERSION:
        return {}
    if not isinstance(lock.get("models"), dict):
        return {}

    return lock


def write_models_lock(output_dir, lock: Dict) -> Path:
    """Function to write the lock file stamping the downloaded artifacts."""

    path = models_lock_path(output_dir)
    path.parent.mkdir(exist_ok=True, parents=True)

    lock = {"version": _MODELS_LOCK_VERSION, "models": lock.get("models", {})}

    with path.open("w", encoding="utf-8") as fw:
        json.dump(lock, fw, indent=4, sort_keys=True)
        fw.write("\n")

    return path


def locked_model_revision(lock: Dict, name: str, repo_id: str) -> str | None:
    """
    Revision an artifact was downloaded from, `None` when it is not stamped.

    An artifact stamped as coming from another repository counts as not stamped:
    its revision says nothing about the repository that is configured now.
    """

    entry = (lock.get("models") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    if entry.get("repo-id") != repo_id:
        return None

    revision = entry.get("revision")

    return revision if isinstance(revision, str) else None


def stamp_model_download(
    lock: Dict,
    spec: "NlpModelSpec",
    repo_id: str,
    revision: str | None,
    target: Path,
) -> Dict:
    """Function to record in `lock` which revision an artifact was downloaded from."""

    lock.setdefault("models", {})[spec.name] = {
        "repo-id": repo_id,
        "revision": revision,
        "relative-path": spec.relative_path,
        "size-bytes": target.stat().st_size if target.exists() else None,
    }

    return lock


def load_models_config(resources_dir=None) -> Dict:
    """Function to read the `models.json` describing the model artifacts"""

    if resources_dir is None:
        resources_dir = get_resources_dir()

    models_file_path = os.path.join(resources_dir, "models.json")
    with open(models_file_path, encoding="utf-8") as fr:
        return json.load(fr)


def list_pretrained_nlp_models(resources_dir=None) -> List[NlpModelSpec]:
    """Function to list the pretrained NLP models declared in `models.json`"""

    models = load_models_config(resources_dir)

    specs = []
    for name, files in models["nlp"]["trained-models"].items():
        relative_path = str(files[1])
        parts = Path(relative_path).parts
        kind = parts[1] if len(parts) > 2 else ""
        specs.append(
            NlpModelSpec(
                name=name,
                kind=kind,
                filename=str(files[0]),
                relative_path=relative_path,
            )
        )

    return specs


# Trees below the resources directory that hold (large) training data rather than
# resources the NLP models need at runtime.
_SKIPPED_RESOURCE_TREES = ("data", "data_nlp")

# Files that describe the resources directory they sit in, so they must not travel
# to another one.
_SKIPPED_RESOURCE_FILES = (MODELS_LOCK_FILENAME,)


def copy_support_resources(
    output_dir,
    resources_dir=None,
    force: bool = False,
    verbose: bool = False,
) -> List[Path]:
    """
    Function to copy the packaged resources that are not downloaded.

    Next to the model artifacts, the NLP models read a handful of files that ship
    with the package (the confusables tables, the regex data and the JSON
    configurations). Copying them makes `output_dir` a complete resources
    directory, usable as `DOCLING_NLP_RESOURCES_DIR`.
    """

    if resources_dir is None:
        resources_dir = get_resources_dir()

    resources_dir = Path(resources_dir)
    output_dir = Path(output_dir)

    if output_dir == resources_dir:
        return []

    downloadable = {
        spec.relative_path for spec in list_pretrained_nlp_models(resources_dir)
    }

    copied = []
    for source in sorted(resources_dir.rglob("*")):
        if not source.is_file():
            continue

        relative = source.relative_to(resources_dir)
        if relative.parts[0] in _SKIPPED_RESOURCE_TREES:
            continue
        if relative.as_posix() in _SKIPPED_RESOURCE_FILES:
            continue
        if relative.as_posix() in downloadable:
            continue

        target = output_dir / relative
        if target.exists() and not force:
            continue

        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(source, target)
        copied.append(target)

    if verbose and copied:
        print(f" -> copied {len(copied)} support resource(s) to {output_dir}")

    return copied


def download_model_info(
    spec: NlpModelSpec,
    output_dir,
    repo_id: str,
    revision: str | None = None,
    force: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Function to download the (optional) description of a model.

    The sidecar describes the model and what its labels mean. Not every model has
    one, so a sidecar that the repository does not hold is not an error.

    `overwrite` replaces a sidecar that is already there, which is what a freshly
    downloaded artifact needs; `force` also bypasses the HuggingFace cache.
    """

    target = spec.info_target(output_dir)

    if target.exists() and not (force or overwrite):
        return False

    try:
        cached_file = hf_hub_download(
            repo_id=repo_id,
            filename=spec.info_relative_path,
            revision=revision,
            force_download=force,
        )

        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(cached_file, target)
    except Exception:
        if verbose:
            print(f" -> no description for {spec.name}")
        return False

    if verbose:
        print(f" -> downloaded the description of {spec.name}")

    return True


def download_pretrained_nlp_models(
    names: Sequence[str] | None = None,
    output_dir=None,
    force: bool = False,
    verbose: bool = False,
) -> List[str]:
    """
    Function to download (a selection of) the pretrained NLP models.

    The models are taken from the HuggingFace repository declared in
    `models.json` and written below `output_dir` (the resources directory by
    default), preserving the layout the NLP models are looked up in.

    Which revision every artifact came from is stamped in a lock file next to
    them, so that moving the pin in `models.json` re-downloads the artifacts that
    are behind it. An artifact that is present but not stamped has an unknown
    revision and is downloaded again, once, to stamp it.
    """

    resources_dir = Path(get_resources_dir())
    if output_dir is None:
        output_dir = resources_dir
    output_dir = Path(output_dir)

    models = load_models_config(resources_dir)
    huggingface = models["huggingface"]
    repo_id = huggingface["repo-id"]
    revision = huggingface.get("revision")

    specs = {spec.name: spec for spec in list_pretrained_nlp_models(resources_dir)}
    lock = read_models_lock(output_dir)

    if names is None:
        selected = list(specs.values())
    else:
        unknown = [name for name in names if name not in specs]
        if unknown:
            raise ValueError(
                f"Unknown NLP model(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(specs)}"
            )
        selected = [specs[name] for name in names]

    downloaded_models = []
    stamped = False
    for spec in selected:
        target = spec.target(output_dir)
        locked_revision = locked_model_revision(lock, spec.name, repo_id)

        if target.exists() and not force and locked_revision == revision:
            if verbose:
                print(f" -> already downloaded {spec.name}")
            # the description may have been added to the repository after the
            # artifact was downloaded, so it is fetched independently
            download_model_info(
                spec,
                output_dir=output_dir,
                repo_id=repo_id,
                revision=revision,
                force=False,
                verbose=False,
            )
            downloaded_models.append(spec.name)
            continue

        if verbose:
            if target.exists() and not force:
                reason = (
                    f"revision {locked_revision[:8]} -> {(revision or 'main')[:8]}"
                    if locked_revision
                    else "unknown revision"
                )
                print(f" -> re-downloading {spec.name} ({reason}) ... ", end="")
            else:
                print(f" -> downloading {spec.name} ... ", end="")

        target.parent.mkdir(exist_ok=True, parents=True)
        try:
            cached_file = hf_hub_download(
                repo_id=repo_id,
                filename=spec.relative_path,
                revision=revision,
                force_download=force,
            )
            shutil.copyfile(cached_file, target)
        except Exception as exc:
            if verbose:
                print("failed!")
            raise RuntimeError(
                f"Failed to download NLP model {spec.name!r} from {repo_id!r} "
                f"({spec.relative_path!r})"
            ) from exc

        if verbose:
            print("done!")

        # the description travels with the artifact: a fresh artifact gets a fresh
        # description, so an older one is replaced even without `force`
        download_model_info(
            spec,
            output_dir=output_dir,
            repo_id=repo_id,
            revision=revision,
            force=force,
            overwrite=True,
            verbose=False,
        )

        stamp_model_download(lock, spec, repo_id, revision, target)
        stamped = True

        downloaded_models.append(spec.name)

    if stamped:
        write_models_lock(output_dir, lock)

    return downloaded_models


def load_pretrained_nlp_models(force: bool = False, verbose: bool = False):
    """Function to load pretrained NLP models"""

    return download_pretrained_nlp_models(force=force, verbose=verbose)
