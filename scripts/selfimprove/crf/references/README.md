# Training the reference CRF

This directory contains a small, reproducible workflow for training Docling's
native reference parser from the
[GROBID citation corpus](https://github.com/grobidOrg/grobid/tree/master/grobid-trainer/resources/dataset/citation/corpus).

The workflow has two stages:

1. Download the citation XML files and convert their TEI markup into labelled
   training and validation records.
2. Tokenize those records, train Docling's native `CUSTOM_CRF` model, and report
   metrics on the held-out validation split.

## Quick start

Run the commands from the repository root. Install the project dependencies
first:

```bash
uv sync
```

Prepare the dataset:

```bash
uv run python scripts/selfimprove/crf/references/prepare_trainig_data_from_grobid.py
```

Train and validate the model:

```bash
uv run python scripts/selfimprove/crf/references/train_crf_references.py \
  --config scripts/selfimprove/crf/references/crf_references.yaml
```

Both commands use `artifacts/grobid-crf-references` by default, so the example
configuration works directly with the default preparation command.

> [!NOTE]
> `prepare_trainig_data_from_grobid.py` retains the originally requested
> `trainig` spelling in its filename.

## Preparing the data

`prepare_trainig_data_from_grobid.py` resolves the requested GROBID branch, tag,
or commit to a commit SHA. It then uses GitHub's API to enumerate the corpus and
downloads only regular files whose names end in `.xml`. It does not clone the
repository, download Git history, or copy neighboring GROBID resources.

The parser supports both the legacy `<citations><bibl>` layout and namespaced
TEI documents. Mixed XML content is flattened into the original citation text,
while supported TEI elements become labelled, end-exclusive character spans.

The split is deterministic for a given `--seed`. Splitting happens by source
XML file rather than by individual citation, preventing references from the
same source document from appearing in both partitions.

Useful options:

```text
--output-dir PATH       Destination for XML and derived data
--ref REF               GROBID branch, tag, or commit; default: master
--validation-ratio N    Fraction of source files held out; default: 0.1
--seed N                Deterministic split seed; default: 42
--force                 Redownload XML files already present
--debug                 Enable detailed logging
```

Use `--help` for the complete CLI reference.

### Prepared artifacts

The default output layout is:

```text
artifacts/grobid-crf-references/
├── raw/                # XML files downloaded from GROBID
├── train.jsonl         # Training references and character spans
├── validation.jsonl    # Held-out references and character spans
└── manifest.json       # Source revision, split, checksums, and counts
```

Each JSONL record contains the flattened citation, its annotations, and source
provenance:

```json
{
  "source_file": "example.training.references.tei.xml",
  "source_index": 7,
  "text": "A. Writer. An Article. A Journal 4:10-12 (2024).",
  "annotation": [
    {"label": "authors", "start": 0, "end": 9},
    {"label": "title", "start": 11, "end": 21},
    {"label": "journal", "start": 23, "end": 32}
  ]
}
```

Offsets use Python string character positions, and `end` is exclusive. The
training script validates every range and converts it to the UTF-8 byte offsets
expected by the native CRF implementation.

### TEI label mapping

| TEI content | CRF label |
| --- | --- |
| `author`, `editor` | `authors` |
| Article or monograph `title` | `title` |
| Journal `title` (`level="j"`) | `journal` |
| Series `title` (`level="s"`) | `conference` |
| `biblScope` volume | `volume` |
| `biblScope` issue | `issue` |
| `biblScope` page, pages, or pp | `pages` |
| `date` | `date` |
| `publisher`, `orgName`, `institution` | `publisher` |
| `pubPlace` | `location` |
| `ptr`, `ref` | `url` |
| DOI, ISBN, or ISSN `idno` | `doi`, `isbn`, or `issn` |
| Other `idno` | `identifier` |
| `note` | `note` |

Text outside supported elements is retained and receives the native `null`
label during tokenization.

## Configuring training

`train_crf_references.py` takes one required `--config` argument. The included
[`crf_references.yaml`](crf_references.yaml) is a complete example:

```yaml
data:
  train_file: artifacts/grobid-crf-references/train.jsonl
  validation_file: artifacts/grobid-crf-references/validation.jsonl

training:
  model_name: references
  epochs: 20
  gaussian_sigma: 2.0
  loglevel: WARNING
  verbose: false

output:
  directory: artifacts/grobid-crf-references/model
  prepared_file: references.annot.jsonl
  model_file: crf_reference.bin
  metrics_file: validation_metrics.txt
```

Configuration paths are resolved relative to the current working directory,
not relative to the YAML file. Running from the repository root is therefore
recommended.

The training fields are:

| Field | Meaning |
| --- | --- |
| `model_name` | Descriptive name passed to the native configuration |
| `epochs` | Positive number of CRF training epochs |
| `gaussian_sigma` | Positive Gaussian regularization parameter |
| `loglevel` | Native model logging level |
| `verbose` | Include verbose native evaluation output |

## Model and validation outputs

With the included configuration, training creates:

```text
artifacts/grobid-crf-references/model/
├── references.annot.jsonl
├── crf_reference.bin
└── validation_metrics.txt
```

- `references.annot.jsonl` is the complete native training input. Each record
  contains token offsets, token text, a `true-label`, and a
  `training-sample` flag.
- `crf_reference.bin` is the trained native `CUSTOM_CRF` model.
- `validation_metrics.txt` contains per-label precision, recall, F1, label
  counts, sequence accuracy, and the token-level confusion matrix.

Evaluation uses only records with `"training-sample": false`; these are derived
from `validation.jsonl`. Metrics are token-level rather than exact citation-span
metrics.

The metrics report is also printed to standard output after a successful run.

## Reproducibility

The preparation manifest records:

- the resolved upstream GROBID commit;
- SHA-256 checksums for every downloaded XML file;
- the split assigned to every source file;
- the validation ratio and random seed;
- record and label counts.

Keep `manifest.json`, the YAML configuration, and `references.annot.jsonl` with
the trained model when comparing or reproducing runs.

To refresh the corpus from the current upstream branch, run the preparation
step with `--force`. To reproduce an earlier run exactly, pass the commit stored
in its manifest to `--ref`.

## Troubleshooting

### A download was interrupted

Run the preparation command again. Completed XML files are reused, while
missing files are downloaded. Use `--force` if an existing file should be
replaced.

### GitHub rejects or rate-limits requests

Wait and retry. The downloader uses bounded retries for transient errors. A
specific commit passed through `--ref` also protects a resumed run from branch
changes.

### An annotation-offset error is reported

The training script intentionally rejects invalid, overlapping, or unsafe
annotations instead of silently assigning incorrect token labels. Inspect the
reported JSONL record and its source XML before continuing.

### Training completes without expected files

Check that output filenames use the conventional `.bin` model suffix and `.txt`
metrics suffix, that the output directory is writable, and that both JSONL
splits contain records. Rerun with `--debug` and set `training.loglevel` to
`INFO` for more native diagnostics.
