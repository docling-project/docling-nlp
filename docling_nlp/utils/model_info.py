#!/usr/bin/env python
"""Module to describe the pretrained NLP model artifacts"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from docling_nlp.utils.load_pretrained_models import (
    MODEL_INFO_SUFFIX,
    NlpModelSpec,
    get_resources_dir,
    list_pretrained_nlp_models,
    load_models_config,
    locked_model_revision,
    read_models_lock,
)

# Sentinel labels the CRF models use internally, not part of their output.
_CRF_SENTINELS = ("!BOS!", "!BOE!")

_FASTTEXT_MAGIC = 793712314

_FASTTEXT_LOSSES = {1: "hierarchical-softmax", 2: "negative-sampling", 3: "softmax"}
_FASTTEXT_MODELS = {1: "cbow", 2: "skipgram", 3: "supervised"}


@dataclass(frozen=True)
class NlpModelLabel:
    """One label a model produces, with what it means and how often it occurs."""

    name: str
    description: str = ""
    count: int | None = None

    def to_dict(self) -> Dict:
        return {
            "label": self.name,
            "description": self.description,
            "count": self.count,
        }


@dataclass(frozen=True)
class NlpModelInfo:
    """Everything the CLI knows about one pretrained NLP model artifact."""

    name: str
    kind: str
    filename: str
    relative_path: str
    path: Path
    downloaded: bool
    repo_id: str
    revision: str | None = None
    local_revision: str | None = None
    size_bytes: int | None = None
    remote_size_bytes: int | None = None
    summary: str = ""
    task: str = ""
    applies_to: Tuple[str, ...] = ()
    dependencies: Tuple[str, ...] = ()
    upstream: str | None = None
    labels: Tuple[NlpModelLabel, ...] = ()
    label_description: str = ""
    label_count_header: str = ""
    info_path: Path | None = None
    details: Dict[str, str] = field(default_factory=dict)

    @property
    def stale(self) -> bool:
        """
        Whether the downloaded artifact is behind the revision that is configured.

        An artifact that is not downloaded is never stale. One that is downloaded
        but carries no stamp has an unknown revision, which counts as stale: it
        cannot be shown to be the configured one.
        """

        return self.downloaded and self.local_revision != self.revision

    def to_dict(self) -> Dict:
        """Representation of the model info as plain JSON-serialisable data."""

        return {
            "name": self.name,
            "kind": self.kind,
            "filename": self.filename,
            "relative-path": self.relative_path,
            "path": str(self.path),
            "downloaded": self.downloaded,
            "size-bytes": self.size_bytes,
            "remote-size-bytes": self.remote_size_bytes,
            "huggingface": {
                "repo-id": self.repo_id,
                "revision": self.revision,
                "local-revision": self.local_revision,
            },
            "stale": self.stale,
            "summary": self.summary,
            "task": self.task,
            "applies-to": list(self.applies_to),
            "dependencies": list(self.dependencies),
            "upstream": self.upstream,
            "label-description": self.label_description,
            "label-count-header": self.label_count_header,
            "info-path": (str(self.info_path) if self.info_path is not None else None),
            "labels": [label.to_dict() for label in self.labels],
            "details": dict(self.details),
        }


def format_size(size_bytes: int | None) -> str:
    """Function to render a number of bytes in human readable form"""

    if size_bytes is None:
        return "unknown"

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024

    return f"{size_bytes} B"  # pragma: no cover


def _read_cstring(fr) -> str:
    chars = bytearray()
    while True:
        char = fr.read(1)
        if not char or char == b"\x00":
            break
        chars += char

    return chars.decode("utf-8", errors="replace")


def read_fasttext_labels(path) -> Tuple[List[NlpModelLabel], Dict[str, str]]:
    """
    Function to read the labels and hyper-parameters of a fastText model.

    The `.bin` files start with a magic number and a version, followed by the
    training arguments and the dictionary. The dictionary entries carry a type,
    which is 1 for the labels.
    """

    labels: List[NlpModelLabel] = []
    details: Dict[str, str] = {}

    with open(path, "rb") as fr:
        magic, version = struct.unpack("<ii", fr.read(8))
        if magic != _FASTTEXT_MAGIC:
            raise ValueError(f"{path} is not a fastText model file")

        args = struct.unpack("<12i", fr.read(48))
        (
            dim,
            _ws,
            epoch,
            min_count,
            _neg,
            word_ngrams,
            loss,
            model,
            bucket,
            minn,
            maxn,
            _lr_update_rate,
        ) = args
        fr.read(8)  # the sampling threshold `t`, a double

        size, nwords, nlabels = struct.unpack("<iii", fr.read(12))
        ntokens, _pruneidx_size = struct.unpack("<qq", fr.read(16))

        for _ in range(size):
            word = _read_cstring(fr)
            count, entry_type = struct.unpack("<qb", fr.read(9))
            if entry_type == 1:
                # the labels are stored with the `__label__` prefix fastText uses
                labels.append(
                    NlpModelLabel(name=word.removeprefix("__label__"), count=count)
                )

        details = {
            "format": f"fastText (version {version})",
            "training mode": _FASTTEXT_MODELS.get(model, str(model)),
            "loss": _FASTTEXT_LOSSES.get(loss, str(loss)),
            "dimension": str(dim),
            "word n-grams": str(word_ngrams),
            "character n-grams": f"{minn}-{maxn}",
            "buckets": f"{bucket:,}",
            "vocabulary": f"{nwords:,} word(s), {nlabels} label(s)",
            "training tokens": f"{ntokens:,}",
            "epochs": str(epoch),
            "min-count": str(min_count),
        }

    return labels, details


def read_crf_labels(path) -> Tuple[List[NlpModelLabel], Dict[str, str]]:
    """
    Function to read the labels and the size of a CRF model.

    The CRF models are stored as tab-separated `label<TAB>feature<TAB>weight`
    lines, so the labels are the distinct values of the first column.
    """

    counts: Dict[str, int] = {}
    weights = 0

    with open(path, encoding="utf-8", errors="replace") as fr:
        for line in fr:
            label, _, rest = line.partition("\t")
            if not rest:
                continue
            counts[label] = counts.get(label, 0) + 1
            weights += 1

    for sentinel in _CRF_SENTINELS:
        counts.pop(sentinel, None)

    labels = [
        NlpModelLabel(name=label, count=count)
        for label, count in sorted(counts.items())
    ]

    details = {
        "format": "CRF (tab-separated label/feature/weight)",
        "feature weights": f"{weights:,}",
    }

    return labels, details


def read_rgx_labels(path) -> Tuple[List[NlpModelLabel], Dict[str, str]]:
    """
    Function to read the labels of a regex asset.

    The regex assets are JSON tables of `type`, `subtype` and `expression`, so
    the labels are the distinct type/subtype pairs.
    """

    with open(path, encoding="utf-8") as fr:
        asset = json.load(fr)

    headers = asset.get("headers", [])
    rows = asset.get("data", [])

    counts: Dict[str, int] = {}
    if "type" in headers and "subtype" in headers:
        type_ind = headers.index("type")
        subtype_ind = headers.index("subtype")
        for row in rows:
            label = f"{row[type_ind]}/{row[subtype_ind]}"
            counts[label] = counts.get(label, 0) + 1

    labels = [
        NlpModelLabel(name=label, count=count)
        for label, count in sorted(counts.items())
    ]

    details = {
        "format": "regex table (JSON)",
        "columns": ", ".join(headers),
        "expressions": f"{len(rows):,}",
    }
    if asset.get("description"):
        details["asset"] = asset["description"]

    return labels, details


def read_model_labels(kind: str, path) -> Tuple[List[NlpModelLabel], Dict[str, str]]:
    """Function to read the labels of a model artifact, dispatching on its kind"""

    readers = {
        "crf": read_crf_labels,
        "fasttext": read_fasttext_labels,
        "rgx": read_rgx_labels,
    }

    reader = readers.get(kind)
    if reader is None:
        return [], {}

    try:
        return reader(path)
    except Exception as error:
        return [], {"warning": f"could not read the artifact: {error}"}


# Header of the per-label count column, per kind of model.
_LABEL_COUNT_HEADERS = {
    "crf": "feature weights",
    "fasttext": "training examples",
    "rgx": "expressions",
}


def model_info_path(artifact_path) -> Path:
    """Location of the description belonging to a model artifact."""

    artifact_path = Path(artifact_path)

    return artifact_path.with_suffix(MODEL_INFO_SUFFIX)


def read_model_info(path) -> Dict:
    """
    Function to read the (optional) description of a model.

    The sidecar is a JSON file next to the artifact, holding what the model does
    and what its labels mean:

        {
          "model": "semantic",
          "summary": "Classifier of the semantic role of a paragraph ...",
          "task": "text classification (fastText, supervised)",
          "applies-to": ["text", "table", "document"],
          "dependencies": ["numval", "link", "name", "sentence"],
          "label-description": "semantic role of the paragraph",
          "labels": {"text": "Running text of the document.", ...}
        }

    A label may map to its description directly, or to an object holding a
    `description`. Missing or unreadable sidecars yield an empty dictionary, the
    file is optional by design.
    """

    path = Path(path)
    if not path.exists():
        return {}

    try:
        with path.open(encoding="utf-8") as fr:
            sidecar = json.load(fr)
    except Exception:
        return {}

    if not isinstance(sidecar, dict):
        return {}

    descriptions = {}
    for label, description in (sidecar.get("labels") or {}).items():
        if isinstance(description, dict):
            description = description.get("description", "")
        descriptions[label] = str(description)

    sidecar["labels"] = descriptions

    return sidecar


def _describe_labels(
    labels: List[NlpModelLabel], descriptions: Dict[str, str]
) -> List[NlpModelLabel]:
    """Merge the descriptions of the sidecar into the labels read from an artifact."""

    if not descriptions:
        return labels

    described = [
        NlpModelLabel(
            name=label.name,
            description=descriptions.get(label.name, ""),
            count=label.count,
        )
        for label in labels
    ]

    # labels a sidecar documents but the artifact does not hold: keep them, they
    # tell that the artifact and its description have drifted apart
    known = {label.name for label in labels}
    described += [
        NlpModelLabel(name=name, description=description)
        for name, description in descriptions.items()
        if name not in known
    ]

    return described


def _remote_size(repo_id: str, relative_path: str, revision: str | None):
    """Size of an artifact that is not downloaded yet, `None` when unreachable."""

    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        url = hf_hub_url(repo_id=repo_id, filename=relative_path, revision=revision)
        return get_hf_file_metadata(url).size
    except Exception:
        return None


def describe_nlp_model(
    name: str,
    resources_dir=None,
    models_dir=None,
    remote: bool = True,
) -> NlpModelInfo:
    """
    Function to describe a single pretrained NLP model.

    The declaration is read from `models.json` below `resources_dir`, the artifact
    itself is looked up below `models_dir` (the resources directory by default),
    together with the lock file stamping which revision it came from. When the
    artifact is missing and `remote` is set, its size is looked up on HuggingFace.
    """

    if resources_dir is None:
        resources_dir = get_resources_dir()

    resources_dir = Path(resources_dir)
    target_dir = Path(models_dir) if models_dir is not None else resources_dir

    specs = {spec.name: spec for spec in list_pretrained_nlp_models(resources_dir)}
    if name not in specs:
        raise ValueError(
            f"Unknown NLP model: {name}. Available: {', '.join(sorted(specs))}"
        )

    spec: NlpModelSpec = specs[name]
    huggingface = load_models_config(resources_dir).get("huggingface", {})
    repo_id = huggingface.get("repo-id", "")

    path = spec.target(target_dir)
    downloaded = path.exists()

    # the revision the artifact on disk came from, as stamped by the downloader
    local_revision = locked_model_revision(
        read_models_lock(target_dir), spec.name, repo_id
    )

    labels: List[NlpModelLabel] = []
    details: Dict[str, str] = {}
    size_bytes = None
    remote_size_bytes = None

    if downloaded:
        size_bytes = path.stat().st_size
        labels, details = read_model_labels(spec.kind, path)
    elif remote:
        remote_size_bytes = _remote_size(
            repo_id,
            spec.relative_path,
            huggingface.get("revision"),
        )

    # the description of the model is optional: it is looked up next to the artifact
    # and, failing that, next to the packaged copy in the resources directory
    sidecar_path: Path | None = model_info_path(path)
    sidecar = read_model_info(sidecar_path)
    if not sidecar and target_dir != resources_dir:
        sidecar_path = model_info_path(spec.target(resources_dir))
        sidecar = read_model_info(sidecar_path)
    if not sidecar:
        sidecar_path = None

    labels = _describe_labels(labels, sidecar.get("labels", {}))

    return NlpModelInfo(
        name=spec.name,
        kind=spec.kind,
        filename=spec.filename,
        relative_path=spec.relative_path,
        path=path,
        downloaded=downloaded,
        repo_id=repo_id,
        revision=huggingface.get("revision"),
        local_revision=local_revision,
        size_bytes=size_bytes,
        remote_size_bytes=remote_size_bytes,
        summary=sidecar.get("summary", ""),
        task=sidecar.get("task", ""),
        applies_to=tuple(sidecar.get("applies-to", [])),
        dependencies=tuple(sidecar.get("dependencies", [])),
        upstream=sidecar.get("upstream"),
        labels=tuple(labels),
        label_description=sidecar.get("label-description", ""),
        label_count_header=_LABEL_COUNT_HEADERS.get(spec.kind, "count"),
        info_path=sidecar_path,
        details=details,
    )
