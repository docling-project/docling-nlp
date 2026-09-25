# Graph Language Models

[![PyPI version](https://img.shields.io/pypi/v/docling-nlp)](https://pypi.org/project/docling-nlp/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-nlp)](https://pypi.org/project/docling-nlp/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Pybind11](https://img.shields.io/badge/build-pybind11-blue)](https://github.com/pybind/pybind11/)
[![Platforms](https://img.shields.io/badge/platform-macos%20|%20linux%20|%20windows-blue)](https://github.com/docling-project/docling-nlp/)
[![License MIT](https://img.shields.io/github/license/docling-project/docling-nlp)](https://opensource.org/licenses/MIT)

## Getting Started

### Finding entities and relations via NLP on text and documents

To get easily started, simply install the `docling-nlp` package from PyPi. This can be
done using the traditional `pip install docling-nlp` or via uv `uv add docling-nlp`.

### Downloading the NLP model artifacts

The pretrained NLP models live in a HuggingFace repository and are fetched on demand.
The `docling-nlp-tools` CLI downloads them ahead of time, which is handy to prepare an
offline or containerised environment,

```sh
# list the available models and see which ones are downloaded and up to date
docling-nlp-tools models list

# describe one model: what it does, how big it is and which labels it produces
docling-nlp-tools models info reference

# download all model artifacts
docling-nlp-tools models download

# download a selection, by name or by kind
docling-nlp-tools models download language geoloc
docling-nlp-tools models download --kind crf

# download into a specific directory (instead of the packaged resources directory)
docling-nlp-tools models download --all --output-dir ./artifacts/models
```

When downloading into a custom directory, the packaged support resources the NLP
models also read (confusables tables, regex data, tokenizer and JSON configurations)
are copied along, so the directory is a complete resources directory. Point
`DOCLING_NLP_RESOURCES_DIR` to it and `docling-nlp` picks the models up,

```sh
export DOCLING_NLP_RESOURCES_DIR=./artifacts/models
```

Pass `--no-standalone` to download only the model artifacts themselves.

`models.json` pins the revision of the HuggingFace repository, and the downloader stamps
which revision every artifact came from in a `models.lock.json` next to them. When the
pin moves, `models list` marks the artifacts left behind as `stale` and the next
`models download` refetches exactly those, without needing `--force`. See
[the download guide](docs/downloading-nlp-models.md) for the details.

The same can be done from Python,

```python
from docling_nlp.utils.load_pretrained_models import (
    download_pretrained_nlp_models,
    list_pretrained_nlp_models,
)

list_pretrained_nlp_models()  # -> [NlpModelSpec(name='part-of-speech', kind='crf', ...), ...]
download_pretrained_nlp_models(names=["language", "geoloc"], force=False, verbose=True)
```

See [`docs/downloading-nlp-models.md`](docs/downloading-nlp-models.md) for the full CLI
reference, the list of model artifacts, and the Docker/CI recipes.

Below, you can find the code-snippet to process pieces of text,

```python
from docling_nlp.utils.load_pretrained_models import load_pretrained_nlp_models
from docling_nlp.nlp_utils import init_nlp_model, print_on_shell

load_pretrained_nlp_models(force=False, verbose=False)
mdl = init_nlp_model()

# from Wikipedia (https://en.wikipedia.org/wiki/France)
text = """
France (French: [fʁɑ̃s] Listen), officially the French Republic
(French: République française [ʁepyblik fʁɑ̃sɛz]),[14] is a country
located primarily in Western Europe. It also includes overseas regions
and territories in the Americas and the Atlantic, Pacific and Indian
Oceans,[XII] giving it one of the largest discontiguous exclusive
economic zones in the world.
"""

res = mdl.apply_on_text(text)
print_on_shell(text, res)
```

The last command will print the pandas dataframes on the shell and provides the
following output,

```sh
text:

   #France (French: [fʁɑ̃s] Listen), officially the French Republic
(French: République française [ʁepyblik fʁɑ̃sɛz]),[14] is a country
located primarily in Western Europe. It also includes overseas regions
and territories in the Americas and the Atlantic, Pacific and Indian
Oceans, giving it one of the largest discontiguous exclusive economic
zones in the world.

properties:

         type label  confidence
0  language    en    0.897559

instances:

  type         subtype               subj_path      char_i    char_j  original
-----------  --------------------  -----------  --------  --------  ---------------------------------------------------------------------
sentence                           #                   1       180  France (French: [fʁɑ̃s] Listen), officially the French Republic
                                                                    (French: République française [ʁepyblik fʁɑ̃sɛz]),[14] is a country
                                                                    located primarily in Western Europe.
term         single-term           #                   1         8  #France
expression   wtoken-concatenation  #                   1         8  #France
parenthesis  round brackets        #                   9        36  (French: [fʁɑ̃s] Listen)
expression   wtoken-concatenation  #                  18        28  [fʁɑ̃s]
term         single-term           #                  29        35  Listen
term         single-term           #                  53        68  French Republic
parenthesis  round brackets        #                  69       125  (French: République française [ʁepyblik fʁɑ̃sɛz])
term         single-term           #                  78       100  République française
term         single-term           #                 112       124  fʁɑ̃sɛz]
parenthesis  reference             #                 126       130  [14]
numval       ival                  #                 127       129  14
term         single-term           #                 136       143  country
term         single-term           #                 165       179  Western Europe
sentence                           #                 181       373  It also includes overseas regions and territories in the Americas and
                                                                    the Atlantic, Pacific and Indian Oceans, giving it one of the largest
                                                                    discontiguous exclusive economic zones in the world.
term         single-term           #                 198       214  overseas regions
term         enum-term-mark-3      #                 207       230  regions and territories
term         single-term           #                 219       230  territories
term         single-term           #                 238       246  Americas
term         enum-term-mark-4      #                 255       290  Atlantic, Pacific and Indian Oceans
term         single-term           #                 255       263  Atlantic
term         single-term           #                 265       272  Pacific
term         single-term           #                 277       290  Indian Oceans
term         single-term           #                 313       359  largest discontiguous exclusive economic zones
term         single-term           #                 367       372  world
```

For complete documents, use DocLang archives (`.dclx`). The C++ document path
reads `document.xml`, applies the selected NLP models to DocLang text surfaces,
and writes annotation CSV files back into the archive.

### Creating Graphs from NLP entities and relations in document collections

To create graphs, you need two ingredients, namely,

1. a collection of text or documents
2. a set of NLP models that provide entities and relations

The document-level workflow is based on `.dclx` archives with annotation CSVs
under `annotations/`. Legacy Deep Search document JSON scripts are no longer
part of the supported document path.

## Install for development

### Python installation

To use the python interface, first make sure all dependencies are installed. We use [uv](https://docs.astral.sh/uv/)
for that. To install all the dependent python packages and get the python bindings, simply execute,

```sh
uv sync --all-extras
```

### CXX compilation

To compile from scratch, simply run the following command in the `docling-nlp` root folder to
create the `build` directory,

```sh
cmake -B ./build; 
```

Next, compile the code from scratch,

```sh
cmake --build ./build -j
```

## Run using the Python Interface

### NLP and GLM examples

The Python interface supports text, structured subject APIs, and direct DocLang
archive access:

```python
from docling_nlp.andromeda_doclang import DocLangXDocument

doc = DocLangXDocument()
doc.read("document.dclx")
doc.apply_nlp("language;term")
doc.write("document.nlp.dclx")

properties = doc.properties()
entities = doc.entities()
instances = doc.instances()
relations = doc.relations()

terms = doc.query_entities(type="term")
term_mentions = doc.query_instances(type="term")
contains_relations = doc.query_relations(name="contains")

entity_hash = DocLangXDocument.hash("Western Europe")
mentions = doc.query_instances(entity_hash=entity_hash)
xpaths = mentions["subj_path"].unique().tolist()
```

The legacy Deep Search document JSON workflow has been removed from the
supported examples.

## Run using CXX executables

If you like to be bare-bones, you can also use the executables for NLP and GLM's directly. In general, we
follow a simple scheme of the form

```sh
./nlp-on-dclx.exe --input <document.dclx> --models 'language;term' --output <document.nlp.dclx>
./nlp.exe -m <mode> -c <JSON-config file>
./glm.exe -m <mode> -c <JSON-config file>
```

In both cases, the modes can be queried directly via the `-h` or `--help`

```sh
./nlp.exe -h
./glm.exe -h
```

and the configuration files can be generated,

```sh
./nlp.exe -m create-configs
./glm.exe -m create-configs
```

### Natural Language Processing (NLP)

After you have generated the configuration files (see above), you can

1. train simple NLP models
```sh
./nlp.exe -m train -c nlp_train_config.json
```
2. leverage pre-trained models
```sh
./nlp.exe -m predict -c nlp_predict.example.json
```

### Graph Language Models (GLM)

1. create a GLM
```sh
./glm.exe -m create -c glm_config_create.json
```
2. explore interactively the GLM
```sh
./glm.exe -m explore -c glm_config_explore.json
```

## Testing

To run the tests, simply execute (after installation),

```sh
uv run pytest ./tests -vvv -s
```
