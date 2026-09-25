# Downloading the NLP model artifacts

The pretrained NLP models are not shipped inside the wheel. They live in the HuggingFace
repository declared in `docling_nlp/resources/models.json` and are fetched on demand, the
first time `load_pretrained_nlp_models()` runs. The `docling-nlp-tools` CLI fetches them
ahead of time, which is what you want when preparing an offline machine, a container
image, or a CI cache.

## The model artifacts

| Name | Kind | Artifact |
| --- | --- | --- |
| part-of-speech | crf | models/crf/part-of-speech/crf_pos_model_en.bin |
| reference | crf | models/crf/reference/crf_reference.bin |
| material | crf | models/crf/ucmi/crf_material.bin |
| language | fasttext | models/fasttext/language/fst_language.bin |
| name | fasttext | models/fasttext/person-name/fst_person_name.bin |
| semantic | fasttext | models/fasttext/semantic/fst_semantic.bin |
| metadata | fasttext | models/fasttext/metadata/fst_author.bin |
| geoloc | rgx | models/rgx/geoloc/rgx_geoloc.json |

The table is the current content of `models.json`; the CLI always reads that file, so it
stays correct when models are added or the repository revision is bumped. Together the
artifacts are roughly 350 MB. The `kind` column is derived from the artifact path and is
what `--kind` selects on.

## Listing what is available

    uv run docling-nlp-tools models list

The command prints the HuggingFace repository, every declared model with its kind and
artifact path, and whether the artifact is already present in the resources directory:

                    NLP models of docling-project/docling-nlp-models
    ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
    ┃ name           ┃ kind     ┃ path                                ┃ downloaded ┃
    ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
    │ part-of-speech │ crf      │ models/crf/part-of-speech/crf_pos_m │ yes        │
    │                │          │ odel_en.bin                         │            │
    │ material       │ crf      │ models/crf/ucmi/crf_material.bin    │ stale      │
    │ geoloc         │ rgx      │ models/rgx/geoloc/rgx_geoloc.json   │ no         │
    └────────────────┴──────────┴─────────────────────────────────────┴────────────┘
    Models directory: /path/to/docling_nlp/resources
    Revision: 0fc3b3e19e76259818ee360d85edf26f4131be8a

    1 model(s) are not from revision 0fc3b3e1, or were downloaded before the
    revision was recorded. Run `docling-nlp-tools models download material` to
    update them.

`stale` means the artifact is there but did not come from the revision that
`models.json` pins — see [The lock file](#the-lock-file). Pass `--output-dir` to inspect
a directory other than the packaged resources directory.

## Inspecting a model

    uv run docling-nlp-tools models info reference

The command describes a single model: what it does, which task it solves, what it
depends on, how large the artifact is, and — read back from the artifact itself — the
labels it produces and its hyper-parameters. What the model and its labels mean comes
from the sidecar described in the next section:

    ╭───────────────────────────────── reference ──────────────────────────────────╮
    │ Parser of bibliographic references: it cuts a reference paragraph into its   │
    │ fields (authors, title, journal, ...).                                       │
    ╰──────────────────────────────────────────────────────────────────────────────╯

      name                 reference
      kind                 crf
      task                 sequence labelling (CRF)
      applies to           text, document
      depends on           link, numval, semantic
      size                 8.1 MB
      format               CRF (tab-separated label/feature/weight)
      feature weights      301,089
      repository           docling-project/docling-nlp-models
      revision             0fc3b3e19e76259818ee360d85edf26f4131be8a
      local revision       0fc3b3e19e76259818ee360d85edf26f4131be8a
      artifact             models/crf/reference/crf_reference.bin
      path                 /path/to/resources/models/crf/reference/crf_reference.bin
      downloaded           yes
      info file            /path/to/resources/models/crf/reference/crf_reference.i…

                   labels (reference fields, one per word-token): 14

      label              description                               feature weights
     ──────────────────────────────────────────────────────────────────────────────
      authors            Author (or editor) names of the cited              53,456
                         work.
      conference         Conference or proceedings the cited                19,276
                         work appeared in.
      date               Publication date, most often the year.             15,814
      ...

`revision` is what `models.json` pins, `local revision` is what the artifact on disk was
stamped with. When they differ a `status` row flags the artifact as stale — see [The lock
file](#the-lock-file).

The labels and the hyper-parameters are read from the artifact, so they describe the
model that is actually on disk:

| Kind | Labels | Count column | Extra details |
| --- | --- | --- | --- |
| `crf` | the distinct labels of the tagger (the `!BOS!` sentinel is dropped) | feature weights of the label | number of feature weights |
| `fasttext` | the classes the classifier predicts | training examples of the label | training mode, loss, dimension, n-grams, buckets, vocabulary, training tokens, epochs, min-count |
| `rgx` | the type/subtype pairs of the regex table | expressions of the label | columns and total number of expressions |

When the artifact is not downloaded yet, the description and the size are still shown
(the size is looked up on HuggingFace) and so are the labels the sidecar declares, but
without the counts only the artifact can give:

| Option | Meaning |
| --- | --- |
| `-o`, `--output-dir` | Where the artifact is looked up. Defaults to the resources directory. |
| `--json` | Print everything as JSON, for scripting. |
| `--offline` | Do not contact HuggingFace for the size of a missing artifact. |

## Where the descriptions come from

What a model does, and what its labels *mean*, cannot be read from an artifact, so it
comes from an optional sidecar next to it, named after the artifact with an
`.info.json` suffix:

    models/crf/reference/crf_reference.bin        # the artifact
    models/crf/reference/crf_reference.info.json  # what it is and what its labels mean

    {
      "model": "reference",
      "summary": "Parser of bibliographic references: it cuts a reference ...",
      "task": "sequence labelling (CRF)",
      "applies-to": ["text", "document"],
      "dependencies": ["link", "numval", "semantic"],
      "label-description": "reference fields, one per word-token",
      "labels": {
        "authors": "Author (or editor) names of the cited work.",
        "title": "Title of the cited work.",
        "...": "..."
      }
    }

Every field is optional, `labels` included. A label maps to its description directly,
or to an object holding a `description`, so the file can grow more per-label fields
later without breaking the reader. A model without a sidecar is still described by
what its artifact holds, and a broken or unreadable sidecar is ignored rather than
fatal.

The sidecar is looked up next to the artifact first and, failing that, next to the
packaged copy in the resources directory — so the descriptions also show for artifacts
downloaded into a directory of their own, and for models that are not downloaded at
all. In that last case the labels shown are the ones the sidecar declares, which the
command says explicitly; only a present artifact can confirm them and count them.

The sidecars belong in the HuggingFace repository, next to the artifacts they describe;
they ship in the package meanwhile. `models download` fetches the sidecar of every
model it downloads and moves on quietly when the repository does not hold one, and
`--standalone` copies the packaged sidecars into the output directory like the other
support resources. Labels that a sidecar describes but a downloaded artifact does not
hold are listed without a count, which is how a description that drifted away from its
model shows up.

## Downloading

    # all model artifacts
    uv run docling-nlp-tools models download

    # a selection, by name
    uv run docling-nlp-tools models download language geoloc

    # a selection, by kind
    uv run docling-nlp-tools models download --kind crf --kind rgx

    # into a specific directory
    uv run docling-nlp-tools models download --all --output-dir ./artifacts/models

Without arguments, all models are downloaded; `--all` states the same explicitly. Naming
models together with `--all` or `--kind` is rejected, as are unknown names and kinds. The
error message lists the valid values.

| Option | Meaning |
| --- | --- |
| `-o`, `--output-dir` | Where to write the artifacts. Defaults to the resources directory. |
| `-k`, `--kind` | Download every model of a kind (`crf`, `fasttext`, `rgx`). Repeatable. |
| `--all` | Download every model. Mutually exclusive with named models and `--kind`. |
| `--force` | Re-download artifacts that are already present. |
| `--standalone` / `--no-standalone` | Copy the packaged support resources next to the models. Default: `--standalone`. Only relevant with `--output-dir`. |
| `-q`, `--quiet` | Print only the target directory. Intended for scripting. |

Artifacts already present are skipped, unless `--force` is given or they did not come
from the pinned revision — see [The lock file](#the-lock-file). Every artifact takes its
label descriptions (`*.info.json`) along, when the repository holds them — also for an
artifact that is skipped, so a description added to the repository after the artifact was
downloaded is picked up. The HuggingFace cache is used underneath, so a re-download after
`--force` is cheap when the revision is unchanged.

## The lock file

`models.json` pins the revision of the HuggingFace repository the artifacts come from:

    "huggingface": {
      "repo-id": "docling-project/docling-nlp-models",
      "revision": "0fc3b3e19e76259818ee360d85edf26f4131be8a"
    }

An artifact on disk is just a file — nothing about it says which revision it was built
from. So the downloader stamps that, in a `models.lock.json` at the root of the directory
it writes to:

    {
        "version": 1,
        "models": {
            "semantic": {
                "repo-id": "docling-project/docling-nlp-models",
                "revision": "0fc3b3e19e76259818ee360d85edf26f4131be8a",
                "relative-path": "models/fasttext/semantic/fst_semantic.bin",
                "size-bytes": 112595120
            }
        }
    }

The download then compares the stamp against the pin, per model, instead of only checking
that the file exists:

| Stamp | What happens |
| --- | --- |
| matches the pinned revision | the artifact is reused |
| another revision | the artifact is downloaded again, no `--force` needed |
| missing | the revision is unknown, so the artifact is downloaded again, once, to stamp it |

The last row is the one-time cost of adopting the lock file: artifacts downloaded before
it existed carry no stamp, and an unstamped artifact cannot be shown to be the pinned one.
They are refetched on the next download and stamped from then on. The HuggingFace cache
makes that cheap when the blob is still cached.

The comparison is offline — no request is made to decide whether an artifact is current.
An artifact stamped as coming from a different `repo-id` counts as unstamped: its revision
says nothing about the repository that is configured now.

The lock file describes the directory it sits in, so `--standalone` does not copy it along
to another one; the standalone directory gets its own when the models are downloaded into
it.

## The resources directory

By default the artifacts are written into the packaged resources directory, at the paths
the native models look them up at. That directory is resolved as:

1. `$DOCLING_NLP_RESOURCES_DIR`, when set; otherwise
2. the resources path reported by the compiled `nlp_model`, that is
   `docling_nlp/resources` inside the installed package.

Model artifacts are not the only thing the models read from that directory. The native
code also loads the confusables tables (`confusables/`), the unit regexes
(`models/rgx/vau/units.jsonl`) and the tokenizer (`models/tok/`), and the Python loader
reads `models.json` and `data.json`. A directory holding only downloaded artifacts is
therefore not a usable resources directory.

That is what `--standalone` handles: when `--output-dir` points somewhere other than the
resources directory, the packaged support files are copied along, so the result is a
complete resources directory. The training data under `data/` and `data_nlp/` is not
copied. Point the environment variable at it and the models load from there:

    uv run docling-nlp-tools models download --all --output-dir ./artifacts/models
    export DOCLING_NLP_RESOURCES_DIR=./artifacts/models

Pass `--no-standalone` to write only the model artifacts, for instance when the target
directory is layered onto a complete resources directory later on.

For scripting, `--quiet` suppresses everything but the target directory:

    export DOCLING_NLP_RESOURCES_DIR=$(uv run docling-nlp-tools models download -q -o ./artifacts/models)

## From Python

The CLI is a thin layer over `docling_nlp.utils.load_pretrained_models`:

    from docling_nlp.utils.load_pretrained_models import (
        copy_support_resources,
        download_pretrained_nlp_models,
        list_pretrained_nlp_models,
    )

    # -> [NlpModelSpec(name='part-of-speech', kind='crf', filename=..., relative_path=...), ...]
    specs = list_pretrained_nlp_models()

    download_pretrained_nlp_models(
        names=["language", "geoloc"],  # None downloads all of them
        output_dir="./artifacts/models",  # None writes to the resources directory
        force=False,
        verbose=True,
    )
    copy_support_resources("./artifacts/models")

The same description the `info` command prints is available from
`docling_nlp.utils.model_info`:

    from docling_nlp.utils.model_info import describe_nlp_model

    info = describe_nlp_model("reference")  # -> NlpModelInfo
    info.labels  # -> (NlpModelLabel(name='authors', description=..., count=...), ...)
    info.info_path  # -> the sidecar that was read, None when there is none
    info.to_dict()  # -> plain JSON-serialisable data

Pass `models_dir` to look the artifact up outside the resources directory, and
`remote=False` to keep the call offline.

`download_pretrained_nlp_models` returns the names it downloaded or found in place, raises
`ValueError` on unknown names, and raises `RuntimeError` when a download fails.
`load_pretrained_nlp_models(force=..., verbose=...)` remains available and downloads every
model, as before.

## Docker and CI

Fetch the artifacts during the image build so that the runtime never downloads:

    RUN uv run docling-nlp-tools models download --all

Writing into the packaged resources directory needs no environment variable at runtime.
When the artifacts belong on a mounted volume instead, download with `--output-dir` and
set `DOCLING_NLP_RESOURCES_DIR` in the image.
