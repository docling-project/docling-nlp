# The DocLang Document Model: `dclg_document` and `dclx_document`

This page describes the native C++ document model used by `docling-nlp` for
DocLang content, and the Python types that mirror it.

The model is split into two layers. Understanding which layer you need is
usually the only decision you have to make:

| You have | You want | Use |
| --- | --- | --- |
| A DocLang XML string or `.dclg` file | Parse, traverse, read text at a path | `dclg_document` |
| A `.dclx` archive | The above, plus annotations, NLP, artifacts (eg page-images), sidecars | `dclx_document` |

---

## 1. The two layers

```text
dclg_document
  raw DCLG/XML bytes, parsed pugixml DOM, XML traversal, XPath/text access

dclx_document : dclg_document
  archive, source path, CSV annotations, NLP state, BibTeX, DCLG sidecars
```

`dclx_document` **publicly inherits** `dclg_document`. Every DCLX document *is*
a DCLG document, so anything that only needs to look at XML can accept the base
type and work equally well on a bare `.dclg` fragment, a summary sidecar, or a
full `.dclx` archive.

### Why the split

DCLG is the content format; DCLX is a container that happens to hold DCLG. Prior
to the split a single type owned both, which meant every consumer of "some
DocLang XML" dragged in the archive, annotation, and NLP machinery — and there
was no type at all for a standalone DCLG fragment. Splitting the model gives:

- **A cheap XML type.** `dclg_document.h` depends on pugixml, the standard
  library, the reproducible hash helper, and the scalar typedefs. No archive, no
  annotation, no legacy item headers.
- **One owner for the DOM.** `dclx_document` holds no second
  `pugi::xml_document`. The inherited DCLG state is the sole owner of the
  primary document XML, so a DOM edit is visible everywhere and cannot drift.
- **Structurally valid sidecars.** Summary, ToC, and concepts are themselves
  parsed `dclg_document` objects rather than unvalidated strings.

### File extensions

| Extension | Contents |
| --- | --- |
| `.dclg` | A standalone DocLang XML document with a `<doclang>` root |
| `.dclx` | A ZIP archive containing `document.xml`, annotations, and artifacts |

`reader::read()` dispatches on the extension (case-insensitively) and rejects
anything else.

---

## 2. `dclg_document` — the XML layer

Header: `src/andromeda/tooling/doclang/dclg_document.h`

### State

A `dclg_document` owns exactly two pieces of primary state:

1. one raw DCLG string, and
2. one `pugi::xml_document` parsed from that string,

plus a last-error message. There is no third source of truth and no lazily
reparsed copy.

### Parse semantics

```cpp
andromeda::doclang::dclg_document doc;

if(not doc.read(xml))
  {
    std::cerr << doc.get_last_error() << "\n";
  }
```

`read()` is all-or-nothing:

- It clears the object first, so a document is never a mix of old and new state.
- On a **successful** parse it stores both the DOM and the raw string.
- On a **failed** parse — malformed XML, or well-formed XML whose root is not
  `<doclang>` — it resets the DOM, leaves the raw string empty, and sets a
  message retrievable through `get_last_error()`.

There is therefore no state in which `raw()` and the DOM disagree.

`valid()` is true when the DOM has a `<doclang>` root element.

### API reference

| Member | Purpose |
| --- | --- |
| `static hash_type hash(std::string_view)` | Reproducible hash, stable across runs and processes |
| `void clear()` | Drop the raw string, DOM, and error message |
| `bool read(std::string_view dclg)` | Parse a DCLG string; see the semantics above |
| `bool valid() const` | Whether a `<doclang>` root is present |
| `const std::string& raw() const` | The exact bytes the document was parsed from |
| `pugi::xml_document& xml()` / `const` | The parsed DOM |
| `pugi::xml_node root()` / `const` | The `<doclang>` element |
| `void iterate_elements(cb)` | Visit every direct child of `<doclang>` |
| `void iterate_elements(name, cb)` | Visit the direct children with a given name |
| `void iterate_table_text(cb)` | Visit table text fragments across the document |
| `void iterate_table_text(table, cb)` | Visit text fragments of one table node |
| `void iterate_picture_text(cb)` | Visit text-bearing picture descendants |
| `void iterate_picture_text(picture, cb)` | Visit text-bearing descendants of one picture |
| `std::string at(xpath, mode="auto")` | Read one DocLang path as text or XML |
| `const std::string& get_last_error() const` | Last failure message |
| `void set_last_error(std::string)` | Record a failure from a surrounding layer |

`hash()` is exposed here, rather than on the archive layer, because entity and
instance identity is derived from document text and must be computable without
an archive.

### Traversal

Prefer callback traversal — it visits nodes in place and allocates no
intermediate vector:

```cpp
doc.iterate_elements([](pugi::xml_node node)
{
  // every direct child of <doclang>
});

doc.iterate_elements("text", [](pugi::xml_node node)
{
  // only <text> children
});
```

DocLang tables encode cell content as XML text nodes sitting between marker
elements such as `<fcel/>`, `<ched/>`, and `<nl/>`. Reaching that content
through ordinary child iteration is awkward, so tables have their own traversal:

```cpp
doc.iterate_table_text([](pugi::xml_node table, pugi::xml_node text_node)
{
  // text_node.value() is one table text fragment
});
```

Pictures and figures mix text-bearing children (`<caption>`, `<list>`,
`<text>`) with artifact children (`<src/>`). `iterate_picture_text()` walks the
text-bearing descendants and skips the rest, so callers get caption and label
text without flattening the whole subtree:

```cpp
doc.iterate_picture_text([](pugi::xml_node picture, pugi::xml_node text_node)
{
  // a text-bearing descendant of picture
});
```

Both table and picture traversals have a single-node overload for when you have
already located the element.

### Path access with `at()`

DocLang paths are indexed element steps, for example
`/doclang[1]/section[2]/text[1]`. `at()` resolves one and returns its content:

| `mode` | Result |
| --- | --- |
| `"text"` | Text content. Tables and pictures use their specialised extraction. |
| `"doclang"` | The node serialised back to DocLang XML |
| `"auto"` (default) | Text when the node is text-like or yields non-empty text; otherwise the serialised XML |

`"auto"` is the useful default for inspection: leaf text comes back as text, and
a structural node you point at comes back as markup rather than an empty string.

A path that does not resolve returns an empty string and sets the last-error
message, so check `get_last_error()` to distinguish "not found" from "found but
empty".

---

## 3. `dclx_document` — the archive layer

Header: `src/andromeda/tooling/doclang/dclx_document.h`

`dclx_document` inherits everything above and adds the container concerns. Its
`clear()` calls `dclg_document::clear()` first, then resets only DCLX state.

### What it adds

**Source path** — `set_source_path()` / `get_source_path()`, the path the
document was read from.

**Archive** — `has_archive()`, `artifacts()`, `set_archive()`,
`clear_archive()`. The retained archive holds every entry that was read,
including artifacts the model does not interpret (embedded images and similar),
so a read/write round trip does not drop them.

**Annotation vectors** — properties, instances, entities, relations, and edges,
each held as a `shared_ptr<std::vector<...>>` so NLP code can operate on them
without copying. Three accessor families are provided:

```cpp
doc.shared_properties();    // shared_ptr, for handing to NLP code
doc.mutable_instances();    // vector&, for in-place modification
doc.get_entities();         // const vector&, for reading
```

**Derived annotations**

```cpp
doc.compute_entities();      // collapse instances into exact/derived entities
doc.materialize_edges();     // rebuild graph edges from current annotations
```

`materialize_edges()` takes a `derived_entity_mode` of `"terms"` (default),
`"all"`, or `"none"`, controlling which derived entities become graph nodes.

**Document-level annotations** — see the next section.

`has_annotations()` reports true when any annotation vector is non-empty *or*
any document-level annotation is present.

### DCLX archive layout

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

`document.xml` is the only required entry; it must be well-formed DocLang XML
with a `<doclang>` root. Everything under `annotations/` is optional, and an
archive carrying none of it is a valid document. `[Content_Types].xml` and
`_rels/.rels` are written for new archives as a convention of the writer, not a
requirement of the reader.

The CSV schemas are documented in
[`src/andromeda/tooling/doclang/README.md`](../src/andromeda/tooling/doclang/README.md).

---

## 4. Document-level annotations

Five optional entries carry metadata and derived content about the document as a
whole.

### BibTeX entries

| Entry | Contents |
| --- | --- |
| `annotations/document_reference.bib` | Zero or one entry: the preferred citation *for* this document |
| `annotations/references.bib` | Any number of entries: the works this document cites |

Both are held as `std::optional<std::string>` and round-trip verbatim.

### DCLG sidecars

Summary, ToC, and concepts are standalone DocLang documents, and the model
stores them as such:

```cpp
using sidecar_type = std::optional<std::shared_ptr<dclg_document> >;

const sidecar_type& get_summary()  const;
const sidecar_type& get_toc()      const;
const sidecar_type& get_concepts() const;
```

Holding them as parsed `dclg_document` objects rather than strings means a
sidecar can be traversed and queried with the full DCLG API — `at()`,
`iterate_elements()`, and the rest — and it means an invalid sidecar is rejected
at read time rather than surfacing much later.

Each sidecar goes through a common parse helper that parses the DCLG, forwards
any parse error to the owning DCLX document, and then applies a
sidecar-specific structural check:

**`summary.dclg`** — derived prose about the document. Any valid DCLG with a
`<doclang>` root is accepted; it may use headings, paragraphs, lists, and
tables like any other DocLang content. It is derived content, not a replacement
for `document.xml`.

**`toc.dclg`** — exactly one `<toc>`; every `<entry>` needs a non-empty `xpath`
attribute pointing into `document.xml` and exactly one `<description>`.

```xml
<doclang version="0.7">
  <toc>
    <entry xpath="/doclang[1]/section[2]/section[1]">
      <description>Methods</description>
    </entry>
  </toc>
</doclang>
```

**`concepts.dclg`** — exactly one `<concepts>`; every `<concept>` needs exactly
one `<header>` and at most one each of `<abbreviation>`, `<description>`, and
`<table>`.

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

On write, the writer serialises each sidecar's own `raw()` string, so a sidecar
is never re-emitted from an unvalidated source.

---

## 5. Reading and writing

The reader mirrors the type split:

```cpp
reader::read_dclg_buffer(std::string_view, dclg_document&);
reader::read_dclx_buffer(std::span<const std::byte>, dclx_document&);
reader::read(const std::filesystem::path&, dclx_document&);
```

`read_dclg_buffer()` takes the **base** type, so it can fill a bare DCLG
document, a sidecar, or the DCLG portion of a DCLX document.

`read_dclx_buffer()` loads the ZIP from memory, parses `document.xml` through
the inherited DCLG parser — preserving the original bytes as the raw DCLG
string — attaches the archive, then loads annotations. `.dclx` archives are read
from bytes and extracted in memory; normal processing requires no temporary
files or extraction directories.

`read()` reads the file, dispatches on extension, and sets the source path
afterwards.

Writing goes through:

```cpp
writer::write_dclx(const std::filesystem::path&, dclx_document&, options);
writer::write_dclx_buffer(dclx_document&, std::vector<std::byte>&, options);
```

`writer_options::include_annotations` (default `true`) controls whether the
annotation entries are emitted. `document.xml` is serialised from the DOM, so
programmatic DOM edits round-trip.

```cpp
using namespace andromeda::doclang;

dclx_document doc;
if(not reader::read("input.dclx", doc))
  {
    std::cerr << doc.get_last_error() << "\n";
    return 1;
  }

doc.materialize_edges();

if(not writer::write_dclx("output.dclx", doc))
  {
    std::cerr << doc.get_last_error() << "\n";
    return 1;
  }
```

---

## 6. Views and adapters

`document_view` (`view.h`) wraps a `shared_ptr<const dclg_document>` and offers
convenience collections when a vector of elements is more ergonomic than a
callback:

```cpp
document_view view(doc_ptr);

view.body_elements();
view.elements_by_name("text");
view.text_like_elements();
view.table_elements();
```

Each `element_view` exposes `name()`, `text_content()`, `heading_level()`, and
`location()`. Because the view is built on the base type, it works on sidecars
and bare DCLG as well as on full DCLX documents.

`subject_adapter` (`adapters.h`) converts a document into the `subject<DOCUMENT>`
structures the existing NLP code consumes. It takes `dclx_document` rather than
the base type, because it uses the source path when naming the subject.

---

## 7. The Python mirror

`docling_nlp.andromeda_doclang` exposes the same hierarchy. `DocLangXDocument`
inherits `DoclangDocument` in both C++ and pybind, and the two wrappers share a
single underlying allocation — so `isinstance(dclx_doc, DoclangDocument)` is
true, and there is no copy to keep in sync.

### `DoclangDocument` — the XML layer

```python
from docling_nlp.andromeda_doclang import DoclangDocument

doc = DoclangDocument('<doclang version="0.7"><text>Body</text></doclang>')

for element in doc:
    print(element["name"], element["text"])

print(doc.at(xpath="/doclang[1]/text[1]"))
```

| Method | Purpose |
| --- | --- |
| `read_xml(xml)` | Parse a DocLang XML string |
| `valid()` / `last_error()` | Parse status and failure message |
| `xml()` | The raw DocLang XML the document was parsed from |
| `at(xpath=..., mode="auto")` | Read a DocLang path as text or XML |
| `bounding_box(xpath)` | Return `((x0, y0), (x1, y1))` as floats, or `None` |
| `page_number(xpath)` | Return the 1-based page containing a node, or `None` |
| `__iter__` | Yield direct children of `<doclang>` one at a time |
| `iterate_items(xpath=None)` | Yield direct children as `(xpath, item, page_no, bbox)` |
| `iterate_items_on_page(page_no)` | Yield root items on a 1-based page |

`DoclangDocument` accepts a DCLG string in its constructor or through
`read_xml()`. It does not read a file path; use `DocLangXDocument.read(path)`
for `.dclg` and `.dclx` files. An unsuccessful parse returns `False`; inspect
`last_error()` for the reason.

```python
doc = DoclangDocument(
    '<doclang version="0.7">'
    '<text><location value="10"/><location value="20"/>'
    '<location value="30"/><location value="40"/>First</text>'
    '<page_break/><text>Second</text>'
    '</doclang>'
)

for xpath, item, page_no, bbox in doc.iterate_items():
    print(xpath, item["text"], page_no, bbox)

assert doc.bounding_box("/doclang[1]/text[1]") == (
    (10.0, 20.0), (30.0, 40.0)
)
assert doc.page_number("/doclang[1]/text[2]") == 2
```

Iteration yields dictionaries with `name`, `xml`, and `text` keys. Use `xml()`
to get the full document for Python-side parsing. `at()` requires keyword
arguments; its modes are `"auto"`, `"text"`, and `"doclang"`.
`iterate_items()` is a lazy iterator. Its `page_no` is 1-based and is `None`
for a page-break item. Its `bbox` is `None` when the item has no valid location,
or a list of four rounded integers in DocLang's 0–1000 coordinate space.
`bounding_box()` instead returns two pairs of unrounded floats. An unresolved
path makes `bounding_box()` or `page_number()` return `None` and sets
`last_error()`. `iterate_items(xpath=...)` raises `ValueError` for an unresolved
path; `iterate_items_on_page()` requires a page number of at least 1.

A supplied `xpath` limits iteration to that node's direct children. For
example, `doc.iterate_items(xpath="/doclang[1]/table[1]")` visits that table's
children. `iterate_items_on_page(2)` visits root items on page 2 and excludes
page-break items.

Lists are expanded into one logical `list_item` per `<ldiv>` delimiter; each
item has an XPath ending in `/ldiv[n]` and a well-formed `<list_item>` XML
fragment. This works for nested and flat delimiter forms. Plain
`for element in doc` still yields direct root children, including the original
`list` element, as dictionaries rather than four-part tuples.

### `DocLangXDocument` — the archive layer

Everything above is inherited; the derived type adds archives, annotations,
and NLP.

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

| Method | Purpose |
| --- | --- |
| `read(path)` | Read a `.dclx` archive or `.dclg` XML file |
| `read_xml(xml)` | Replace the document with bare DCLG XML, clearing prior archive and annotation state |
| `write(path)` | Write the document as DCLX, including annotations |
| `source_path()` | Path passed to `read()`, or an empty string for an XML-only document |
| `hash(text)` | Reproducible text hash, useful in entity and edge queries |
| `has_archive()` / `archive_paths()` | Inspect the retained archive |
| `has_annotations()` | Whether any annotation data or document-level annotation is present |
| `annotation_paths()` | List the supported annotation entry names, whether or not they are present |
| `document_reference()` / `references()` | Optional BibTeX content |
| `summary()` / `toc()` / `concepts()` | Sidecars, as optional `DoclangDocument` |
| `set_*()` / `clear_*()` | Set or remove document-level annotations |
| `overview()` | Document and annotation **counts and status flags**, as a dictionary |
| `properties()`, `entities()`, `instances()`, `relations()`, `edges()` | Annotation tables as pandas DataFrames |
| `query_*()` | Filtered pandas DataFrames |
| `apply_nlp(models, progress_every=25)` | Initialise and apply models to one document |
| `materialize_edges(derived_entity_mode="terms")` | Rebuild graph edges |

Pandas is imported lazily, when a table or query method is first called.

The query methods accept optional filters and return DataFrames with the same
columns as their unfiltered counterparts:

| Method | Filters |
| --- | --- |
| `query_properties()` | `type`, `label`, `subj_path`, `min_conf` |
| `query_entities()` | `type`, `subtype`, `name`, `name_contains`, `entity_kind`, `min_count` |
| `query_instances()` | `type`, `subtype`, `name`, `name_contains`, `subj_path`, `min_conf`, `entity_hash` |
| `query_relations()` | `name`, `name_i`, `name_j`, `name_contains`, `min_conf` |
| `query_edges()` | `name`, `hash_i`, `hash_j`, `min_count` |

String filters default to the empty string and numeric filters default to zero,
which leaves that filter inactive. Filters can be combined:

```python
mentions = doc.query_instances(
    type="term",
    subj_path="/doclang[1]/text[1]",
    min_conf=0.8,
    entity_hash=DocLangXDocument.hash("Western Europe"),
)
```

`overview()` returns counts and status flags; `summary()` returns the optional
summary sidecar as a `DoclangDocument`.

Sidecar getters return an optional `DoclangDocument` sharing the parsed
allocation, so the full DCLG surface is available on them:

```python
summary = doc.summary()
if summary is not None:
    print(summary.at(xpath="/doclang[1]/text[1]"))
```

The DCLG setters validate against the structural contracts in section 4 and
return `False` with a message in `last_error()` when the XML does not satisfy
them.

The document-level annotation methods are:

| Content | Get | Set | Remove |
| --- | --- | --- | --- |
| Citation for this document | `document_reference()` | `set_document_reference(bibtex)` | `clear_document_reference()` |
| Cited references | `references()` | `set_references(bibtex)` | `clear_references()` |
| Summary DCLG | `summary()` | `set_document_summary(dclg)` | `clear_document_summary()` |
| Table of contents DCLG | `toc()` | `set_toc(dclg)` | `clear_toc()` |
| Concepts DCLG | `concepts()` | `set_concepts(dclg)` | `clear_concepts()` |

The BibTeX setters accept strings. The DCLG setters accept XML strings and
validate their structure. A missing getter returns `None`; a present DCLG
sidecar can be read and traversed like any other `DoclangDocument`. The
corresponding `clear_*()` method removes it.

### Reusing models across documents

`DocLangXNlp` initialises a model expression once and applies it repeatedly:

```python
from docling_nlp.andromeda_doclang import DocLangXNlp, DocLangXDocument

nlp = DocLangXNlp()
if not nlp.initialise("language;term"):
    raise RuntimeError(nlp.last_error())

print(nlp.initialised(), nlp.model_expr(), nlp.models())
for path in paths:
    doc = DocLangXDocument()
    if not doc.read(path):
        raise RuntimeError(doc.last_error())
    if not nlp.apply(doc, progress_every=25):
        raise RuntimeError(doc.last_error())
    if not doc.write(f"{path}.nlp.dclx"):
        raise RuntimeError(doc.last_error())
```

The constructor also accepts a model expression. `DocLangXDocument.apply_nlp()`
is the one-document shortcut: it initialises models for that call. Use
`DocLangXNlp` when processing multiple documents with the same models.

### Two things that differ from the C++ surface

- **`at()` lives on the base wrapper.** The DCLG layer owns XPath access and the
  derived wrapper carries no duplicate XML methods, so `at()` is defined on
  `DoclangDocument`. `DocLangXDocument.at()` resolves through inheritance.
- **`DoclangDocument.xml()` returns the raw string.** The wrapper keeps no
  separate copy of the DCLG text, so "raw XML" is the base's retained buffer.
  The DCLX writer still serialises the DOM for `document.xml`, so DOM edits
  continue to round-trip.

---

## 8. Design rules

These hold across the model, and changes to it are expected to preserve them:

1. `dclg_document` owns no archive, path, CSV vector, NLP model result, BibTeX
   content, or sidecar.
2. `dclg_document` owns exactly one raw DCLG string and exactly one
   `pugi::xml_document` parsed from that string.
3. A successful parse updates both raw and parsed state. A failed parse leaves
   the object empty and exposes a useful error message.
4. `dclx_document` has no second DOM for `document.xml`; the inherited DCLG
   state is the sole owner of the primary-document XML.
5. Summary, ToC, and concepts are optional parsed `dclg_document` objects, not
   unvalidated strings.
6. The writer serialises the DCLG string owned by each parsed sidecar; the
   reader parses every DCLG archive entry before exposing it.
7. Existing `.dclx` files with none of the newer sidecars remain valid.

Header dependency rules for the module — which files may include what — are
documented under "Boundary Rules" in
[`src/andromeda/tooling/doclang/README.md`](../src/andromeda/tooling/doclang/README.md).

---

## 9. Migrating from `andromeda::doclang::document`

The former single `document` type, in the now-removed
`tooling/doclang/document.h`, has been replaced:

| Old | New |
| --- | --- |
| `#include <andromeda/tooling/doclang/document.h>` | `dclg_document.h` or `dclx_document.h` |
| `doclang::document` (XML use only) | `doclang::dclg_document` |
| `doclang::document` (archive/annotations) | `doclang::dclx_document` |
| `reader::read_dclg_buffer(view, document&)` | takes `dclg_document&` |
| `reader::read_dclx_buffer(bytes, document&)` | takes `dclx_document&` |

The XML surface — `read()`, `valid()`, `raw()`, `xml()`, `root()`, `at()`, the
iteration helpers, `hash()`, and the error accessors — keeps its previous names
and signatures; it simply lives on the base class now. Code that used the XML
surface should compile after switching the include and the type name.

On the Python side, `DocLangXDocument` inherits the new `DoclangDocument`
base type. Code using `elements()` must switch to direct iteration or
`iterate_items()`. Code unpacking `(xpath, item)` from `iterate_items()` must
now unpack `(xpath, item, page_no, bbox)`. The iterator also expands lists into
individual logical items. Document-level annotation accessors are additive.
