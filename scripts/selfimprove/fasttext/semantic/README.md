# Training the semantic FastText classifier

This workflow downloads the maintained semantic dataset, prepares it with the
native semantic model's `link` and `numval` dependencies, trains FastText, and
evaluates the candidate on the provided validation split.

The classifier intentionally has three labels: `meta-data`, `reference`, and
`text`. Any `header` rows are discarded during conversion. The installed model
and its model-information file are not replaced by this workflow.

## Train and evaluate

Run from the repository root:

```bash
uv run python scripts/selfimprove/semantic/train_semantic.py \
  --config scripts/selfimprove/semantic/semantic.yaml
```

The source CSVs are downloaded from
[`docling-project/docling-nlp-datasets`](https://huggingface.co/datasets/docling-project/docling-nlp-datasets/tree/main/data/fasttext/semantic)
only when they are missing from `artifacts/semantic/data`. Use
`--force-download` to replace the local copies.

By default, training writes:

```text
artifacts/semantic/
├── data/train.csv
├── data/validation.csv
├── semantic.jsonl
├── semantic.jsonl.fasttext.train.txt
├── semantic.jsonl.fasttext.validate.txt
├── fst_semantic.bin
├── fst_semantic.vec
├── validation_metrics.txt
└── manifest.json
```

FastText autotuning runs for one hour by default. Edit `semantic.yaml` to alter
the duration, model-size budget, training parameters, output paths, or dataset
revision.

### Training parameters

The YAML uses `null` to mean that an argument is not passed to FastText. With
autotuning enabled, FastText can then select supported parameters. Without
autotuning, the FastText supervised default applies. Setting a concrete value
fixes that parameter and removes it from the autotune search.

| YAML field | FastText option | YAML default | FastText supervised default / autotune behavior |
| --- | --- | --- | --- |
| `learning_rate` | `-lr` | `null` | `0.1`; autotune searches `0.01`–`5.0` |
| `epochs` | `-epoch` | `null` | `5`; autotune searches `1`–`100` |
| `dimension` | `-dim` | `null` | `100`; autotune searches `1`–`1000` |
| `context_window` | `-ws` | `null` | `5`; not optimized |
| `ngram` | `-wordNgrams` | `null` | `1`; autotune searches `1`–`5` |
| `loss` | `-loss` | `null` | `softmax`; values: `softmax`, `ova`, `one-vs-all`, `hs`, `ns` |
| `min_count` | `-minCount` | `null` | `1`; not optimized |
| `min_char_ngram` | `-minn` | `null` | `0`; autotune chooses `0`, `2`, or `3` |
| `max_char_ngram` | `-maxn` | `null` | `0`; autotune uses the selected minimum plus `3` |
| `bucket` | `-bucket` | `null` | `2,000,000`; autotune searches `10,000`–`10,000,000` |
| `threads` | `-thread` | `null` | `12`; not optimized; use `1` for reproducibility |
| `seed` | `-seed` | `null` | `0`; not optimized |

The autotune controls have explicit workflow defaults:

| YAML field | Workflow default | Native FastText default | Meaning |
| --- | ---: | ---: | --- |
| `autotune` | `true` | disabled | Enable validation-guided parameter search |
| `duration` | `3600` | `300` | Maximum search time in seconds |
| `model_size` | `100M` | unlimited | Final model-size constraint; may trigger quantization |
| `metric` | `f1` | `f1` | Search objective; for example `f1:reference` |
| `predictions` | `1` | `1` | Predictions per validation example used by the objective |

FastText autotune also adjusts its internal `dsub` quantization parameter when
`model_size` is constrained. Parameters marked “not optimized” retain their
FastText defaults unless explicitly set in the YAML. Reproducibility requires
both a fixed `seed` and `threads: 1`.

## Run a candidate

Classify one string:

```bash
uv run python scripts/selfimprove/semantic/run_semantic.py \
  artifacts/semantic/fst_semantic.bin \
  "IBM Research, Zurich, Switzerland"
```

Or classify newline-delimited text from standard input:

```bash
printf '%s\n' "IBM Research" "This paper presents ..." | \
  uv run python scripts/selfimprove/semantic/run_semantic.py
```

Each prediction is emitted as one JSON object containing `label`, `confidence`,
and `text`. Candidate loading uses the same preprocessing and dependencies as
the installed semantic model.

### Run on DCLX or PDF

Apply the candidate model to every `<text>` item in an existing DCLX document:

```bash
uv run python scripts/selfimprove/semantic/run_semantic_document.py \
  document.dclx
```

For PDF input, install the optional Docling dependency and run the same command:

```bash
uv sync --extra docling
uv run python scripts/selfimprove/semantic/run_semantic_document.py \
  document.pdf
```

The PDF is converted to a `DoclingDocument` with the standard PDF layout
pipeline. OCR and table-structure extraction are disabled. The document is then
exported as a native DCLX archive, including page images, to
`artifacts/semantic/documents/document.dclx`. Set a different destination with
`--output-dclx`. Predictions are made only for `text` items returned by the
DCLX iterator, so picture and table contents are skipped. The output table
contains the DCLX XPath, predicted semantic label, confidence, and complete
text of each selected item.

Use another model or emit machine-readable JSON Lines with:

```bash
uv run python scripts/selfimprove/semantic/run_semantic_document.py \
  document.dclx \
  --model path/to/fst_semantic.bin \
  --format jsonl
```

Existing semantic annotations in a DCLX archive are ignored: every item is
classified directly with the selected candidate model.

Once a candidate is accepted, update the installed model, its
`fst_semantic.info.json`, and the overall semantic-model description in the C++
source as a separate review step.
