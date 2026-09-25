# Training and applying CRF models

This guide covers the public CRF workflow implemented by base_crf_model and exposed as REFERENCE and CUSTOM_CRF. For a new model, use CUSTOM_CRF for training and custom_crf(name:path) for application.

## Outputs

A successful run produces:

- a prepared annotation JSONL file;
- a model binary, normally ending in .bin; and
- an evaluation report, normally ending in .metrics.txt.

Training does not write metrics itself. Run evaluation after training to create or refresh the metrics report.

## Data layout

### Raw annotation input for the Python preparation helper

docling_nlp/nlp_train_crf.py accepts JSON Lines. Each line is one text example with an annotation list:

    {"text":"FeSe has Tc of 30 K.","annotation":[{"label":"material","start":0,"end":4}]}

start and end are Python string-character offsets into text, with end exclusive. The helper runs the native tokenizer, converts the annotations to the byte-oriented offsets used by the CRF record, adds true-label to every word token, and writes a prepared JSONL file.

Every raw record should have:

| Field | Required | Meaning |
| --- | --- | --- |
| text | Yes | UTF-8 input text |
| annotation | Yes | List of labelled spans; use an empty list for a labelled negative example |
| annotation[].label | Yes | Training label; the helper lowercases, trims, and replaces spaces with underscores |
| annotation[].start, annotation[].end | Yes | End-exclusive character offsets into text |

Only word tokens wholly contained in a labelled span receive that label; all other word tokens receive null. Therefore, inspect boundary behavior for partially overlapping tokens before training.

### Prepared JSONL accepted by the C++ CRF trainer

The trainer reads one JSON object per example. Required fields are:

    {
      "training-sample": true,
      "word-tokens": {
        "headers": ["char_i", "char_j", "word", "true-label"],
        "data": [
          [0, 4, "FeSe", "material"],
          [5, 8, "has", "null"]
        ]
      }
    }

The headers may contain additional columns and may be in any order. The required names are exactly char_i, char_j, word, and true-label. Each data row must have one value for every header. char_i and char_j are unsigned byte offsets in the stored UTF-8 text. training-sample is mandatory: true puts the complete token sequence in the training partition and false puts it in validation/evaluation.

Keep sequences grouped by source text: do not split tokens from one text record across partitions. Use a fixed held-out file for final evaluation instead of evaluating only on the same prepared training file.

## Python: prepare, train, evaluate

The most direct end-to-end entry point is:

    uv run python docling_nlp/nlp_train_crf.py       --mode all       --input-file data/materials.jsonl       --output-dir artifacts/material-crf       --max-items 10000

Modes are prepare, train, evaluate, and all. The command creates:

    artifacts/material-crf/materials.annot.jsonl
    artifacts/material-crf/materials.crf_model.bin
    artifacts/material-crf/materials.crf_model.bin.metrics.txt

The equivalent Python call is:

    from docling_nlp.nlp_train_crf import create_crf_model

    prepared, model_path, metrics_path = create_crf_model(
        mode="all",
        ifile="data/materials.jsonl",
        odir="artifacts/material-crf",
        max_items=10_000,
    )

For separate steps, use the public helpers:

    from docling_nlp.nlp_utils import train_crf, eval_crf

    train_crf(
        model_name="custom_crf",
        train_file="artifacts/material-crf/materials.annot.jsonl",
        model_file="artifacts/material-crf/materials.crf_model.bin",
        metrics_file="artifacts/material-crf/materials.crf_model.bin.metrics.txt",
    )
    eval_crf(
        model_name="custom_crf",
        train_file="artifacts/material-crf/materials.annot.jsonl",
        model_file="artifacts/material-crf/materials.crf_model.bin",
        metrics_file="artifacts/material-crf/materials.crf_model.bin.metrics.txt",
    )

model_name is currently not used to select a distinct custom trainer; CUSTOM_CRF is the public target. The config parameters exposed by the native trainer are epoch (default 20), gaussian-sigma (default 2.0), train-file, validate-file, test-file, model-file, and metrics-file. For JSONL, the trainer uses training-sample to split the single train-file; a separate validate-file is not read in that branch.

## Python: apply a custom CRF

Load the model with the custom expression, then apply it to text:

    from docling_nlp.nlp_utils import init_nlp_model

    model = init_nlp_model(
        "language;custom_crf(material:artifacts/material-crf/materials.crf_model.bin)",
        filters=["properties", "instances"],
    )
    result = model.apply_on_text("FeSe has Tc of 30 K.")

The instances table in result holds extracted spans. The custom CRF currently has no dependencies itself; language is optional for CUSTOM_CRF, but is often useful when using the same wider pipeline as other models.

## Native C++ CLI

The native executable consumes the same JSON configuration that pybind passes to the dispatcher. A CRF training configuration is:

    {
      "mode": "train",
      "model": "custom_crf",
      "verbose": false,
      "args": {
        "epoch": 20,
        "gaussian-sigma": 2.0
      },
      "files": {
        "train-file": "artifacts/material-crf/materials.annot.jsonl",
        "validate-file": "null",
        "test-file": "artifacts/material-crf/materials.annot.jsonl",
        "model-file": "artifacts/material-crf/materials.crf_model.bin",
        "metrics-file": "artifacts/material-crf/materials.crf_model.bin.metrics.txt"
      }
    }

Run it with:

    ./nlp.exe --mode train --config train-crf.json

The executable creates template configurations with:

    ./nlp.exe --mode create-configs

For native application, use a predict configuration with models set to custom_crf(material:path), plus the desired producer configuration. Generate a predict template first, then replace models and input/output producer fields; producer layouts are shared by all native NLP models.

## Direct C++ integration

The stable integration boundary is the dispatcher used by the CLI:

    #include "andromeda.h"

    nlohmann::json config = ...; // same structure as train-crf.json
    andromeda::nlp_train(config);

For application, construct a model list through andromeda::to_models with the custom_crf expression, prepare an andromeda::subject<TEXT> using the normalizers used by the pipeline, and call apply on each model. These C++ classes are header-level implementation APIs, not a separately versioned public SDK; prefer the Python façade or CLI for external integrations.

## Recall, precision, and F1

Evaluation loads model-file, reads test-file when it exists (otherwise train-file), and writes metrics-file. The report includes:

- percentage of perfectly predicted token sequences;
- one row per true or predicted label;
- true count and predicted count;
- from-model, which identifies labels known by the trained model;
- per-label recall, precision, and F1; and
- a token-level confusion matrix.

Run evaluation explicitly after any training or model-file change. The metrics are token-level, not exact entity-span metrics. If your acceptance criterion is span extraction, compute a separate exact/partial-span evaluation from the instances returned by apply_on_text.

## Practical checks

- Ensure every label expected at inference is represented in the training partition.
- Reserve false training-sample records before the run; do not let the same source appear in both partitions.
- Test Unicode spans, punctuation, and multi-token entities because offsets and whole-token labelling matter.
- Keep the prepared JSONL alongside the model binary: it is the reproducible training input and evaluation source.

