# Native DocLang Module

This module owns native C++ DocLang support.

## DCLX Archive Layout

A DCLX file is a ZIP archive. The current archive layout convention is:

```text
document.dclx
├── [Content_Types].xml
├── _rels/
│   └── .rels
├── document.xml
└── annotations/
    ├── properties.csv
    ├── instances.csv
    ├── entities.csv
    ├── relations.csv
    ├── edges.csv
    ├── document_reference.bib
    ├── references.bib
    ├── summary.dclg
    ├── toc.dclg
    └── concepts.dclg
```

`document.xml` is the only required archive entry. It must contain
well-formed DocLang XML with a `<doclang>` root element.

`[Content_Types].xml` and `_rels/.rels` are created for a newly written
archive. They are conventions of the writer, rather than requirements imposed
by the reader. An archive can additionally contain arbitrary artifact entries,
such as embedded images; when a DCLX archive is read and then written, those
entries are retained.

The `annotations/` entries are optional. If present, each CSV must use the
exact filename and ordered header defined below. DCLX files without annotation
CSVs remain valid documents.

## Canonical Representation

The native model is split into a sidecar-free DCLG base and a DCLX-specific
derived type:

```text
dclg_document
  raw DCLG/XML bytes, parsed pugixml DOM, XML traversal, XPath/text access

dclx_document : dclg_document
  archive, source path, CSV annotations, NLP state, BibTeX, DCLG sidecars
```

`andromeda::doclang::dclg_document` owns exactly one raw DCLG string and
exactly one `pugi::xml_document` parsed from that string. A successful `read()`
updates both; a failed `read()` leaves the object empty and exposes the reason
through `get_last_error()`. `dclx_document` inherits that state and never
holds a second DOM for `document.xml`.

DocLang XML is the source of truth for document content. NLP annotations are
stored on the DCLX document as shared vectors of `base_property`,
`base_instance`, and `base_relation`, and are serialized in `.dclx` archives as
CSV files under `annotations/`.

The reader mirrors the split:

```cpp
reader::read_dclg_buffer(std::string_view, dclg_document&);
reader::read_dclx_buffer(std::span<const std::byte>, dclx_document&);
reader::read(const std::filesystem::path&, dclx_document&);
```

## Boundary Rules

Core DocLang XML/archive headers should stay independent of legacy subject and
element structures where possible:

- `archive.h`
- `content.h`
- `reader.h`
- `view.h`
- `writer.h`

These files may depend on:

- the C++ standard library
- `pugixml`
- `miniz`

They should not include:

- `andromeda/tooling/structs/subjects.h`
- `andromeda/tooling/structs/elements.h`
- individual legacy element headers

`dclg_document.h` is the complete XML layer. Beyond the list above it adds only
`andromeda/utils/hash/utils.h` and `andromeda/tooling/base_types.h`, for the
reproducible hash and the scalar typedefs.
`dclx_document.h` is the explicit annotation-storage exception and may include
legacy item types. `adapters.h` is the subject compatibility exception because
its job is to create execution views for existing NLP code.

## Annotation Files

DocLang archives use these stable annotation paths:

- `annotations/properties.csv`
- `annotations/instances.csv`
- `annotations/entities.csv`
- `annotations/relations.csv`
- `annotations/edges.csv`

When present, these files are read into the document annotation vectors. When a
document is written as `.dclx`, the writer packages the current annotation
vectors into these CSV files.

Instances are occurrence-level text snippets with document paths and ranges.
Entities are collapsed from instances into exact and suffix-derived entity rows;
if `annotations/entities.csv` is missing, the document computes entities from
the loaded instances.

## Annotation CSV Schema

CSV values use RFC-style quoting: fields containing commas, quotes, or line
breaks are quoted, and embedded quotes are doubled. Each generated file ends
with a newline. Empty `coor_i` and `coor_j` cells in `instances.csv` represent
null coordinate values.

### `annotations/properties.csv`

```csv
type,subj_hash,subj_name,subj_path,label,confidence
```

### `annotations/instances.csv`

```csv
type,subtype,subj_hash,subj_name,subj_path,conf,hash,ihash,coor_i,coor_j,char_i,char_j,ctok_i,ctok_j,wtok_i,wtok_j,wtok-match,name,original
```

### `annotations/entities.csv`

```csv
type,subtype,name,entity_kind,hash,count,parent,parent_hash
```

### `annotations/relations.csv`

```csv
flvr,name,conf,hash_i,hash_j,name_i,name_j
```

For backward compatibility, the reader also accepts a leading
`doclang_xpath` column in `relations.csv`. The column is validated and then
ignored; writers emit the seven-column header above.

### `annotations/edges.csv`

```csv
hash,flvr,name,hash_i,hash_j,count,probability
```

The reader rejects unexpected headers and row widths. If
`entities.csv` is absent while `instances.csv` is present, entities are
computed from the loaded instances.

## Document-Level Annotations

The following optional entries extend the annotation layout with document-level
metadata and derived content. Native C++ and Python read/write support is
provided by this module.

The three `.dclg` sidecars are held as parsed `dclg_document` objects, never as
unvalidated strings: `dclx_document::get_summary()`, `get_toc()` and
`get_concepts()` return `std::optional<std::shared_ptr<dclg_document> >`. The
reader parses every DCLG archive entry before exposing it, and the writer
serialises each sidecar's own `raw()` string. A DCLX file carrying none of
these sidecars remains valid.

### `annotations/document_reference.bib`

This file contains the document's own bibliographic reference in BibTeX
format. It contains zero or one BibTeX entry: the preferred citation for the
document represented by `document.xml`.

### `annotations/references.bib`

This file contains the bibliographic references cited by the document, in
BibTeX format. It may contain any number of entries. Citation keys are the
canonical identifiers used by consumers that resolve citations in the document.

### `annotations/summary.dclg`

This file contains a standalone DocLang XML document (`.dclg`) representing a
summary of the document. Its root is `<doclang>` and it uses the same DocLang
version and XML conventions as `document.xml`.

The summary is independent of the source document tree: it is derived content,
not a replacement for `document.xml`. It may contain headings, paragraphs,
lists, tables, and other normal DocLang elements needed to express the
summary.

### `annotations/toc.dclg`

This file contains a standalone DocLang XML document (`.dclg`) representing
the table of contents. Its root is `<doclang>`. Each `<entry>` identifies a
subsection in `document.xml` through its absolute DocLang path and contains a
short description:

```xml
<doclang version="0.7">
  <toc>
    <entry xpath="/doclang[1]/section[2]/section[1]">
      <description>Methods</description>
    </entry>
  </toc>
</doclang>
```

`xpath` is required and targets the subsection element in `document.xml`.
Each entry has one `<description>` child.

### `annotations/concepts.dclg`

This file contains a standalone DocLang XML document (`.dclg`) that defines
concepts extracted from or associated with the document. Its root is
`<doclang>`, with concepts grouped beneath a `<concepts>` element:

```xml
<doclang version="0.7">
  <concepts>
    <concept>
      <header>Concept name</header>
      <abbreviation>Optional abbreviation</abbreviation>
      <description>Optional definition or explanation.</description>
      <table>
        <!-- Optional properties table, using DocLang table conventions. -->
      </table>
    </concept>
  </concepts>
</doclang>
```

Every `<concept>` has exactly one `<header>`. The `<abbreviation>`,
`<description>`, and `<table>` children are optional. The properties table
uses ordinary DocLang table encoding and may define any concept-specific
properties.

## Python: `DocLangXDocument`

The `docling_nlp.andromeda_doclang` extension exposes the same DCLX document
path through `DocLangXDocument`. It reads `.dclx` archives and plain DocLang
XML, returns NLP annotations as pandas DataFrames, and writes DCLX archives.
Pandas is imported when a table or query method is called.

```python
from docling_nlp.andromeda_doclang import DocLangXDocument

doc = DocLangXDocument()
if not doc.read("input.dclx"):
    raise RuntimeError(doc.last_error())

doc.apply_nlp("language;term")
doc.materialize_edges()

terms = doc.query_entities(type="term")
mentions = doc.query_instances(
    entity_hash=DocLangXDocument.hash("Western Europe")
)

if not doc.write("output.dclx"):
    raise RuntimeError(doc.last_error())
```

### Python API Summary

| API | Purpose |
| --- | --- |
| `read(path)` | Read a `.dclx` archive or `.dclg` XML file. |
| `read_xml(xml)` | Parse a DocLang XML string without an archive. |
| `write(path)` | Write the current document as DCLX, including annotations. |
| `valid()` / `last_error()` | Inspect parse success and a failure message. |
| `has_archive()` / `archive_paths()` | Inspect the retained DCLX archive and its entries. |
| `has_annotations()` / `annotation_paths()` | Inspect loaded annotations and their stable archive paths. |
| `document_reference()` / `references()` | Return optional BibTeX content for the document and its citations. |
| `document_summary()` / `toc()` / `concepts()` | Return the DCLG document-level annotations as optional `DoclangDocument` objects. |
| `set_*()` / `clear_*()` | Set or remove document-level annotations; DCLG setters validate their XML contract. |
| `xml()` | Return the raw DocLang XML the document was parsed from. |
| `at(xpath=..., mode="auto")` | Read a DocLang path as text or XML. |
| `__iter__` | Yield direct child elements of `<doclang>` one at a time. |
| `iterate_items(xpath=None)` | Yield direct children as `(xpath, item, page_no, bbox)`. |
| `iterate_items_on_page(page_no)` | Yield root items on a 1-based page. |
| `summary()` | Return document and annotation counts as a dictionary. |
| `properties()`, `entities()`, `instances()`, `relations()`, `edges()` | Return the corresponding annotation table as a pandas DataFrame. |
| `query_*()` | Return a filtered pandas DataFrame for the corresponding annotation type. |
| `apply_nlp(models, progress_every=25)` | Initialise and apply models for one document. |
| `materialize_edges(derived_entity_mode="terms")` | Rebuild graph edges from current annotations. |

`DoclangDocument` is the XML-only base of `DocLangXDocument`, mirroring the
native `dclg_document` / `dclx_document` hierarchy: every `DocLangXDocument`
*is* a `DoclangDocument`, and the two wrappers share one allocation. It parses
a DCLG string and iterates the direct child elements of its `<doclang>` root:

```python
from docling_nlp.andromeda_doclang import DoclangDocument

doc = DoclangDocument('<doclang version="0.7"><text>Body</text></doclang>')
for element in doc:
    print(element["name"], element["text"])
```

Each element is a dictionary with `name`, `xml`, and `text` fields. Use `xml()`
to get the full document for Python-side parsing or filtering. `iterate_items()`
returns `None` for unavailable page numbers or boxes; otherwise its box has
four rounded integers in the 0–1000 coordinate space. A supplied `xpath`
restricts iteration to that node's direct children. Lists are expanded into
one logical `list_item` per `<ldiv>` delimiter, including flat delimiter forms;
each item has an XPath ending in `/ldiv[n]` and a well-formed `<list_item>` XML
fragment. Plain `for element in doc` still yields direct root children.

`read_xml()`, `valid()`, `xml()`, `last_error()`, `at()`, and iteration all come
from `DoclangDocument`; `DocLangXDocument` adds the archive,
annotation and NLP surface on top. The DCLG sidecars are returned as
`DoclangDocument` objects sharing the parsed sidecar allocation:

```python
summary = doc.document_summary()
if summary is not None:
    print(summary.at(xpath="/doclang[1]/text[1]"))
```

Python `summary()` remains the count/status dictionary; `document_summary()`
is the summary sidecar.

`at()` requires keyword arguments. Its `mode` is `"auto"` by default; use
`"text"` for text content or `"doclang"` for serialised DocLang XML. Paths
use the DocLang form, for example `/doclang[1]/text[1]`.

`DocLangXNlp` is available when the same initialised model expression should
be reused for multiple documents:

```python
from docling_nlp.andromeda_doclang import DocLangXNlp

nlp = DocLangXNlp("language;term")
for path in paths:
    doc = DocLangXDocument()
    if doc.read(path) and nlp.apply(doc):
        doc.write(f"{path}.nlp.dclx")
```

## Traversal

Prefer callback-based traversal on `doclang::dclg_document` when possible:

```cpp
doc.iterate_elements([](pugi::xml_node node)
{
  // inspect node without allocating a vector of views
});
```

Use `document_view` when callers benefit from convenience collections such as
`text_like_elements()` or `table_elements()`.

DocLang tables encode cell content as XML text nodes between marker elements
such as `<fcel/>`, `<ched/>`, and `<nl/>`. Use `iterate_table_text()` when code
needs those table text fragments directly:

```cpp
doc.iterate_table_text([](pugi::xml_node table, pugi::xml_node text_node)
{
  // text_node.value() is the table text fragment.
});
```

DocLang pictures and figures can contain text-bearing children such as
`<caption>`, `<list>`, and `<text>`, plus artifact children such as `<src/>`.
Use `iterate_picture_text()` to process the text-bearing descendants without
flattening the whole picture:

```cpp
doc.iterate_picture_text([](pugi::xml_node picture, pugi::xml_node text_node)
{
  // text_node is a text-bearing descendant of picture.
});
```

## Archive Handling

`.dclx` archives should be read from bytes and extracted in memory. Normal
processing should not require temporary files or extraction directories.
