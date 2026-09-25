# NLP model capabilities and interfaces

This inventory covers models exposed by the production factory and reusable bases in src/andromeda/tooling/models. “Train” means trainable through the current public dispatcher, not merely inheriting training code.

## Capability summary

| Family | Models | Apply | Train through public API | Output |
| --- | --- | --- | --- | --- |
| SentencePiece tokenizer | SPM, CUSTOM_SPM | Yes | Yes | IDs and subword strings on word tokens |
| FastText supervised family | SEMANTIC, CUSTOM_FST | Yes | Yes | Label/confidence property |
| CRF sequence-tagging family | REFERENCE, CUSTOM_CRF | Yes | Yes | Token labels and entity instances |
| FastText instantiations | LANGUAGE, NAME | Yes | No* | Classification/property or name instances |
| CRF instantiation | LAPOS | Yes | No | POS tags |
| Rule/regex extractor | NUMVAL, GEOLOC, LINK, CITE, QUOTE, PARENTHESIS, EXPRESSION, SENTENCE | Yes | No | Entity instances |
| POS-pattern extractor | CONN, TERM, VERB | Yes | No | Entity instances |
| Relation extractor | ABBREVIATION, VAU | Yes | No | Relation/instance annotations |
| Record extractor | METADATA | Yes | No | Document metadata instances |

* NAME inherits FastText training machinery, but to_trainable_model does not return it. Thus neither the C++ dispatcher nor the pybind public API can train it today.

The FastText and CRF rows describe reusable training/inference implementations.
LANGUAGE and NAME are concrete, pretrained FastText-backed model instantiations;
LAPOS is a concrete, pretrained CRF-backed POS-tagging instantiation. Their
current public training availability is determined by factory registration, not
by their parent implementation.

## Apply-only models

| Model(s) | Capability | Scope and dependencies |
| --- | --- | --- |
| NUMVAL | Rule-based numeric value recognition. | Text |
| GEOLOC | Regex/resource-backed geolocation recognition; loads models/rgx/geoloc/rgx_geoloc.json. | Text |
| LINK, CITE, PARENTHESIS | Link, citation, and bracketed-span recognition. | Text |
| QUOTE | Quote recognition. | Text; depends on NUMVAL |
| EXPRESSION | Heuristic expression grouping. | Text; depends on name, link, cite, number, quote, parenthesis |
| SENTENCE | Heuristic sentence segmentation. | Text; depends on preceding entity annotations |
| LANGUAGE | Pretrained FastText language classifier; masks numeric values in tables. | Text, table, document; depends on NUMVAL |
| LAPOS | Pretrained CRF part-of-speech tagger. | Text uses LANGUAGE and SENTENCE; table uses LANGUAGE |
| CONN, TERM, VERB | PCRE2 patterns over POS-tag encodings. | Text; require SENTENCE and LAPOS |
| NAME | Pretrained FastText-assisted name extraction. | Text |
| ABBREVIATION | Abbreviation/name-term relation extraction. | Text; depends on NAME, TERM |
| VAU | Value-and-unit extraction. | Text/table; depends on NUMVAL |
| METADATA | Title, abstract, and related record extraction. | Document; depends on NAME, GEOLOC, SEMANTIC |

These rules are not interchangeable instances of base_rgx_model; many embed their own matching logic. GEOLOC is the prominent external-regex-resource model.

## Apply-and-train models

### SentencePiece: SPM and CUSTOM_SPM

base_tok_model wraps SentencePiece. Training accepts one sentence per line and configures unigram/BPE/word/character type, vocabulary size, character coverage, thread count, maximum sentence/piece lengths, number splitting, and control/user symbols. SPM loads the standard resource tokenizer; CUSTOM_SPM(name:path) loads a supplied model.

SentencePiece writes trainer outputs itself. base_tok_model::save currently returns false, so persistence is not a working reusable model-object operation.

### FastText supervised: SEMANTIC and CUSTOM_FST

fasttext_supervised_model supports supervised FastText training, optional autotuning, loading/saving .bin and .vec artifacts, classification with confidence, and confusion-matrix precision/recall/F1. Input is JSONL records with label and text; optional training-sample controls the split, otherwise the implementation makes a random 90/10 split.

SEMANTIC classifies text/table/document content and depends on LINK and NUMVAL. CUSTOM_FST(name:path) is the general custom classifier. LANGUAGE and NAME use the same base but are not public training targets.

### CRF: REFERENCE and CUSTOM_CRF

base_crf_model trains, loads, saves, predicts, and evaluates the in-tree CRF. It accepts prepared token-annotation JSONL, supports epoch and gaussian-sigma, writes a .bin model, and writes a text precision/recall/F1 report. Long predictions are chunked at the CRF maximum length.

REFERENCE depends on LINK, NUMVAL, and SEMANTIC. CUSTOM_CRF(name:path) maps token predictions back to entity instances and post-processes contiguous/BIO-like labels.

## Declared or reusable, but not supported end-to-end

| Item | Current state |
| --- | --- |
| TOPIC, DATE | Enum entries with no factory implementation. |
| base_rgx_model | Loads/saves a JSON PCRE2-expression table and applies it to text, but no concrete factory model derives from it. |
| base_dct_model | Intended dictionary/regex base, but its operations are absent or return false; it is not factory-exposed. |
| base_mxr_model | Placeholder NLP-mixer base whose trainer returns false; not factory-exposed. |
| base_fst_model | Unused low-level FastText base with no public training flow; unlike the supervised implementation it does not initialise model before load. |

## C++ interface

All implementations derive from base_nlp_model. The base provides model identity, dependencies, and apply overloads for text, tables, figures, and documents. Default table/document logic recursively applies a model and records its key. to_models resolves dependencies before appending the requested model.

The generic training contract is is_trainable, create_train_config, prepare_data_for_train, train, and evaluate_model. Only SentencePiece, FastText supervised, and CRF provide a complete useful training path. Configurations are untyped JSON. The C++ CLI and pybind binding both use to_trainable_model, whose switch defines the actual training surface.

## Python interface via pybind11

andromeda_nlp.nlp_model is a stateful façade, not a per-model Python hierarchy. A caller selects models with initialise or initialise_models, then calls apply_on_text, apply_on_table, or apply_on_doc. The string text overload returns annotation JSON; subject-wrapper overloads mutate the wrapper and return bool.

prepare_data_for_train, train, and evaluate receive JSON configs and return a JSON status envelope. get_apply_configs and get_train_configs provide templates. Python helpers in docling_nlp/nlp_utils.py wrap CRF, tokenizer, and FastText flows. The binding exposes no per-model typed configuration, load/save/predict API, metrics API, or capability discovery.

## Recommended improvements

1. Register capabilities as data: name, task, scopes, dependencies, trainability, schemas, and artifact formats. Use the registry for factory construction and Python discovery.
2. Replace JSON-only public APIs with typed C++ config/result structures and pybind dataclasses; retain JSON conversion for CLI compatibility.
3. Return structured results with diagnostics, artifact paths, model version, warnings, and machine-readable metrics instead of bools and log-only failures.
4. Add typed Python Tokenizer, Classifier, SequenceTagger, and RuleExtractor APIs with explicit load/predict/train/evaluate/save semantics.
5. Require deterministic split assignments or a seed, and persist resolved configuration and data hashes.
6. Emit JSON aggregate and per-label metrics while preserving text reports.
7. Either complete/register dictionary and generic regex models or clearly keep them internal. Complete tokenizer persistence and repair or retire base_fst_model before exposing it.
