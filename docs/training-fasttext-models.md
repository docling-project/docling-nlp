# Training and applying FastText models

This guide covers the supervised FastText implementation, fasttext_supervised_model. The public training targets are SEMANTIC and CUSTOM_FST. For a new classifier, train CUSTOM_FST and apply it through custom_fst(name:path).

## Outputs

A training workflow creates:

- a source JSONL dataset;
- generated FastText train and validation files beside that dataset;
- a .bin classifier model;
- a .vec vector export; and
- a .metrics.txt evaluation report.

Training does not evaluate automatically. Call evaluation after training to write recall, precision, and F1.

## Data layout

The source dataset is JSON Lines, one labelled text per line:

    {"label":"person-name","text":"Marie Curie","training-sample":true}
    {"label":"expr","text":"IBM Research","training-sample":false}

| Field | Required | Meaning |
| --- | --- | --- |
| label | Yes | Class label without the FastText prefix |
| text | Yes | UTF-8 source text |
| training-sample | Recommended | true for training; false for validation/evaluation |

During preparation, the implementation tokenizes and preprocesses text. It then writes:

    __label__person-name Marie Curie
    __label__expr IBM Research

to data.jsonl.fasttext.train.txt or data.jsonl.fasttext.validate.txt.

If training-sample is absent, the current implementation assigns a random 90/10 split. Always set it explicitly for repeatable training and stable metrics. Keep examples from the same source/document in one partition.

For classifiers whose preprocessing relies on dependencies, data preparation applies dependency models before generating the FastText text. SEMANTIC depends on LINK and NUMVAL. CUSTOM_FST has no dependencies by default.

## Python: prepare, train, evaluate

Use the helpers in docling_nlp.nlp_utils:

    from docling_nlp.nlp_utils import (
        prepare_data_for_fst_training,
        train_fst,
        eval_fst,
    )

    data_file = "data/name-classifier.jsonl"
    model_file = "artifacts/name-classifier.bin"
    metrics_file = "artifacts/name-classifier.metrics.txt"

    prepare_data_for_fst_training(data_file=data_file)
    train_fst(
        data_file=data_file,
        model_file=model_file,
        metrics_file=metrics_file,
        autotune=True,
        duration=600,
        modelsize="10M",
        ngram=1,
    )
    eval_fst(
        data_file=data_file,
        model_file=model_file,
        metrics_file=metrics_file,
    )

prepare_data_for_fst_training creates the generated FastText files. train_fst trains CUSTOM_FST. eval_fst reloads the model, reads records marked training-sample false, and writes metrics_file.

The template configuration also supports these fields:

| Config group | Fields |
| --- | --- |
| hpo | autotune, modelsize, duration, metric, predictions |
| args | mode (supervised), learning-rate, epoch, dim, ws, n-gram, loss, min-count, min-char-ngram, max-char-ngram, bucket, thread, seed |
| files | data-file, model-file, metrics-file |

Pass only parameters you intend to override. A missing or null training parameter is not forwarded to FastText, allowing its native default or autotune search to apply.

## Python: apply a custom classifier

    from docling_nlp.nlp_utils import init_nlp_model
    import pandas as pd

    model = init_nlp_model(
        "custom_fst(name:artifacts/name-classifier.bin)",
        filters=["properties"],
    )
    result = model.apply_on_text("Marie Curie")

    properties = pd.DataFrame(
        result["properties"]["data"],
        columns=result["properties"]["headers"],
    )
    print(properties[["label", "confidence"]])

CUSTOM_FST writes a property containing the predicted label and confidence. Use the returned property table rather than assuming a particular row order when applying several models.

## Native C++ CLI

The C++ training dispatcher accepts this configuration:

    {
      "mode": "train",
      "model": "custom_fst",
      "hpo": {
        "autotune": true,
        "modelsize": "10M",
        "duration": 600
      },
      "args": {
        "mode": "supervised",
        "n-gram": 1
      },
      "files": {
        "data-file": "data/name-classifier.jsonl",
        "model-file": "artifacts/name-classifier.bin",
        "metrics-file": "artifacts/name-classifier.metrics.txt"
      }
    }

Preparing data, training, and evaluation are separate dispatcher calls. Use the same config with the respective pybind methods, or call the native executable for training:

    ./nlp.exe --mode train --config train-fasttext.json

The native executable does not have a separate evaluate mode. Its train mode calls nlp_train only. To evaluate through the supported public interface, use Python nlp_model.evaluate or eval_fst. Native predict configuration supports models such as custom_fst(name:artifacts/name-classifier.bin); generate a template with ./nlp.exe --mode create-configs and set its models/producers fields.

## Direct C++ integration

The CLI path is the supported C++ integration boundary:

    #include "andromeda.h"

    nlohmann::json config = ...; // same structure as train-fasttext.json
    andromeda::nlp_train(config);

For application, build models with andromeda::to_models using custom_fst(name:path), construct a normalized subject<TEXT>, and apply each selected model. The direct fasttext_supervised_model class is an internal header implementation; use the dispatcher unless you own the embedding application and can track source-level API changes.

## Recall, precision, and F1

eval_fst calls the FastText evaluator. It reads the source JSONL, selects only training-sample false records, preprocesses them with the same dependency models, predicts labels, and writes a text report to metrics-file.

The report contains:

- a per-label true count and predicted count;
- per-label recall, precision, and F1;
- a confusion matrix; and
- incorrectly predicted examples in the application log.

The report is not JSON. For automated model selection, parse it carefully or add machine-readable metrics as proposed in the self-improving-loop plan. Do not compare runs whose validation partitions differ.

## Practical checks

- Ensure every label has enough training and validation examples; inspect per-label counts, not just aggregate F1.
- Use explicit training-sample values and retain the source JSONL, generated FastText files, model, vectors, config, and metrics together.
- Keep model-file as the intended base path. Saving normalizes a .bin suffix and writes both .bin and .vec.
- Test application with representative short, long, multilingual, and dependency-sensitive text before deployment.
