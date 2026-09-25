import zipfile
from xml.etree import ElementTree

import pandas as pd
import pytest
from docling_nlp.andromeda_doclang import DoclangDocument, DocLangXDocument, DocLangXNlp


def create_dclx(path):
    xml = '<doclang version="0.7"><text>FeSe is a material.</text></doclang>'
    fese_hash = DocLangXDocument.hash("FeSe")

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("_rels/.rels", "<Relationships></Relationships>")
        archive.writestr("document.xml", xml)
        archive.writestr(
            "annotations/properties.csv",
            "type,subj_hash,subj_name,subj_path,label,confidence\n"
            "language,123,text,/doclang[1]/text[1],en,0.99\n",
        )
        archive.writestr(
            "annotations/instances.csv",
            "type,subtype,subj_hash,subj_name,subj_path,conf,hash,ihash,coor_i,coor_j,char_i,char_j,ctok_i,ctok_j,wtok_i,wtok_j,wtok-match,name,original\n"
            f"term,,123,text,/doclang[1]/text[1],1,{fese_hash},789,,,0,4,0,1,0,1,true,FeSe,FeSe\n",
        )
        archive.writestr(
            "annotations/relations.csv",
            "flvr,name,conf,hash_i,hash_j,name_i,name_j\n"
            "42,contains,0.75,456,789,FeSe,material\n",
        )


def create_entity_tree_dclx(path):
    xml = (
        '<doclang version="0.7"><text>very tall man tall man small man</text></doclang>'
    )
    rows = [
        "term,,123,text,/doclang[1]/text[1],1,1001,2001,,,0,13,0,3,0,3,true,very tall man,very tall man",
        "term,,123,text,/doclang[1]/text[1],1,1001,2002,,,14,27,4,7,4,7,true,very tall man,very tall man",
        "term,,123,text,/doclang[1]/text[1],1,1001,2003,,,28,41,8,11,8,11,true,very tall man,very tall man",
        "term,,123,text,/doclang[1]/text[1],1,1001,2004,,,42,55,12,15,12,15,true,very tall man,very tall man",
        "term,,123,text,/doclang[1]/text[1],1,1002,2011,,,56,64,16,18,16,18,true,tall man,tall man",
        "term,,123,text,/doclang[1]/text[1],1,1002,2012,,,65,73,19,21,19,21,true,tall man,tall man",
        "term,,123,text,/doclang[1]/text[1],1,1002,2013,,,74,82,22,24,22,24,true,tall man,tall man",
        "term,,123,text,/doclang[1]/text[1],1,1003,2021,,,83,92,25,27,25,27,true,small man,small man",
        "term,,123,text,/doclang[1]/text[1],1,1003,2022,,,93,102,28,30,28,30,true,small man,small man",
    ]

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("_rels/.rels", "<Relationships></Relationships>")
        archive.writestr("document.xml", xml)
        archive.writestr(
            "annotations/instances.csv",
            "type,subtype,subj_hash,subj_name,subj_path,conf,hash,ihash,coor_i,coor_j,char_i,char_j,ctok_i,ctok_j,wtok_i,wtok_j,wtok-match,name,original\n"
            + "\n".join(rows)
            + "\n",
        )


def test_doclangx_document_read_query_write(tmp_path):
    input_path = tmp_path / "input.dclx"
    output_path = tmp_path / "output.dclx"
    create_dclx(input_path)

    doc = DocLangXDocument()
    assert doc.read(str(input_path))
    assert doc.valid()
    assert doc.has_archive()
    assert doc.has_annotations()

    summary = doc.overview()
    assert summary["properties"] == 1
    assert summary["instances"] == 1
    assert summary["relations"] == 1

    assert doc.annotation_paths() == [
        "annotations/properties.csv",
        "annotations/instances.csv",
        "annotations/entities.csv",
        "annotations/relations.csv",
        "annotations/edges.csv",
        "annotations/document_reference.bib",
        "annotations/references.bib",
        "annotations/summary.dclg",
        "annotations/toc.dclg",
        "annotations/concepts.dclg",
    ]
    assert "document.xml" in doc.archive_paths()

    properties = doc.properties()
    entities = doc.entities()
    instances = doc.instances()
    relations = doc.relations()

    assert isinstance(properties, pd.DataFrame)
    assert isinstance(entities, pd.DataFrame)
    assert isinstance(instances, pd.DataFrame)
    assert isinstance(relations, pd.DataFrame)
    assert properties.loc[0, "label"] == "en"
    assert entities.loc[0, "name"] == "FeSe"
    assert entities.loc[0, "entity_kind"] == "exact"
    assert entities.loc[0, "count"] == 1
    assert DocLangXDocument.hash("FeSe") == entities.loc[0, "hash"]
    assert instances.loc[0, "name"] == "FeSe"
    assert instances.loc[0, "hash"] == entities.loc[0, "hash"]
    assert relations.loc[0, "name"] == "contains"
    assert str(properties.dtypes["type"]) == "string"
    assert str(properties.dtypes["subj_hash"]) == "UInt64"
    assert str(properties.dtypes["confidence"]) == "Float32"
    assert str(entities.dtypes["hash"]) == "UInt64"
    assert str(entities.dtypes["count"]) == "UInt64"
    assert str(entities.dtypes["entity_kind"]) == "string"
    assert str(instances.dtypes["coor_i"]) == "UInt64"
    assert str(instances.dtypes["wtok-match"]) == "boolean"
    assert pd.isna(instances.loc[0, "coor_i"])
    assert str(relations.dtypes["flvr"]) == "UInt64"
    assert str(relations.dtypes["conf"]) == "Float32"

    assert len(doc.query_properties(type="language")) == 1
    assert len(doc.query_entities(type="term", name_contains="Fe")) == 1
    assert len(doc.query_instances(type="term", name_contains="Fe")) == 1
    mentions = doc.query_instances(entity_hash=DocLangXDocument.hash("FeSe"))
    assert len(mentions) == 1
    assert mentions["subj_path"].unique().tolist() == ["/doclang[1]/text[1]"]
    assert len(doc.query_instances(type="term", name="missing")) == 0
    assert len(doc.query_relations(name="contains", name_contains="material")) == 1
    assert len(doc.query_relations(name="contains", min_conf=0.9)) == 0
    assert doc.overview()["edges"] == 0

    doc.materialize_edges()
    edges = doc.edges()
    assert len(edges) == 3
    assert sorted(edges["name"].tolist()) == [
        "contains",
        "to-entities",
        "to-instances",
    ]
    assert (
        len(doc.query_edges(name="to-instances", hash_i=DocLangXDocument.hash("FeSe")))
        == 1
    )
    assert doc.overview()["edges"] == 3

    assert doc.write(str(output_path))

    restored = DocLangXDocument()
    assert restored.read(str(output_path))
    assert restored.overview()["instances"] == 1
    assert restored.overview()["entities"] == 1
    assert len(restored.query_instances(name="FeSe")) == 1
    assert restored.overview()["edges"] == 3
    assert len(restored.query_edges(name="to-entities")) == 1


def test_doclangx_document_entities_are_collapsed_suffix_tree(tmp_path):
    input_path = tmp_path / "entity-tree.dclx"
    create_entity_tree_dclx(input_path)

    doc = DocLangXDocument()
    assert doc.read(str(input_path))

    instances = doc.instances()
    entities = doc.entities()
    assert len(instances) == 9
    assert len(entities) == 4

    by_name = entities.set_index("name")
    assert by_name.loc["very tall man", "entity_kind"] == "exact"
    assert by_name.loc["very tall man", "count"] == 4
    assert by_name.loc["very tall man", "parent"] == "tall man"
    assert by_name.loc["tall man", "entity_kind"] == "exact"
    assert by_name.loc["tall man", "count"] == 7
    assert by_name.loc["tall man", "parent"] == "man"
    assert by_name.loc["small man", "entity_kind"] == "exact"
    assert by_name.loc["small man", "count"] == 2
    assert by_name.loc["small man", "parent"] == "man"
    assert by_name.loc["man", "entity_kind"] == "derived"
    assert by_name.loc["man", "count"] == 9
    assert by_name.loc["man", "parent"] == ""

    derived = doc.query_entities(entity_kind="derived", min_count=5)
    assert len(derived) == 1
    assert derived.loc[0, "name"] == "man"


def test_doclangx_document_apply_nlp_empty_model_expr(tmp_path):
    output_path = tmp_path / "empty-models.dclx"

    doc = DocLangXDocument()
    assert doc.read_xml('<doclang version="0.7"><text>Body text</text></doclang>')
    assert doc.apply_nlp("", progress_every=0)
    assert doc.overview()["instances"] == 0
    assert doc.write(str(output_path))

    restored = DocLangXDocument()
    assert restored.read(str(output_path))
    assert restored.valid()


def test_doclangx_document_document_level_annotations_round_trip(tmp_path):
    output_path = tmp_path / "annotations.dclx"

    doc = DocLangXDocument()
    assert doc.read_xml('<doclang version="0.7"><text>Body text</text></doclang>')
    doc.set_document_reference("@article{document, title={Document}}\n")
    doc.set_references("@article{reference, title={Reference}}\n")
    assert doc.set_document_summary(
        '<doclang version="0.7"><text>Summary</text></doclang>'
    )
    assert doc.set_toc(
        '<doclang version="0.7"><toc><entry xpath="/doclang[1]/section[1]">'
        "<description>Introduction</description></entry></toc></doclang>"
    )
    assert doc.set_concepts(
        '<doclang version="0.7"><concepts><concept><header>FeSe</header>'
        "<abbreviation>FeSe</abbreviation><description>Material</description>"
        "</concept></concepts></doclang>"
    )
    assert doc.overview()["has_document_reference"]
    assert doc.overview()["has_references"]
    assert doc.overview()["has_summary"]
    assert doc.overview()["has_toc"]
    assert doc.overview()["has_concepts"]
    assert doc.write(str(output_path))

    restored = DocLangXDocument()
    assert restored.read(str(output_path))
    assert restored.document_reference() == "@article{document, title={Document}}\n"
    assert restored.references() == "@article{reference, title={Reference}}\n"
    # the sidecars come back as parsed DoclangDocument objects
    assert restored.summary().valid()
    assert "<text>Summary</text>" in restored.summary().xml()
    assert restored.summary().at(xpath="/doclang[1]/text[1]") == "Summary"
    assert 'xpath="/doclang[1]/section[1]"' in restored.toc().xml()
    assert "<header>FeSe</header>" in restored.concepts().xml()

    restored.clear_document_reference()
    restored.clear_references()
    restored.clear_document_summary()
    restored.clear_toc()
    restored.clear_concepts()
    assert restored.document_reference() is None
    assert restored.references() is None
    assert restored.summary() is None
    assert restored.toc() is None
    assert restored.concepts() is None


def test_doclangx_document_rejects_invalid_document_level_dclg():
    doc = DocLangXDocument()
    assert doc.read_xml('<doclang version="0.7"><text>Body text</text></doclang>')
    assert not doc.set_toc('<doclang version="0.7"><toc><entry /></toc></doclang>')
    assert "entries require xpath" in doc.last_error()
    assert not doc.set_concepts(
        '<doclang version="0.7"><concepts><concept /></concepts></doclang>'
    )
    assert "require one <header>" in doc.last_error()


def test_doclang_document_iterates_dclg_elements():
    doc = DoclangDocument(
        '<doclang version="0.7"><heading>Title</heading><text>Body</text></doclang>'
    )

    assert doc.valid()
    assert doc.xml().startswith("<doclang")
    iterator = iter(doc)
    assert iter(iterator) is iterator
    assert next(iterator)["text"] == "Title"
    assert "<text>Body</text>" in next(iterator)["xml"]
    with pytest.raises(StopIteration):
        next(iterator)
    assert not hasattr(doc, "elements")

    invalid = DoclangDocument()
    assert not invalid.read_xml("<text>missing root</text>")
    assert "root <doclang>" in invalid.last_error()


@pytest.mark.parametrize("document_type", [DoclangDocument, DocLangXDocument])
def test_doclang_document_bounding_box_and_page_number(document_type):
    doc = document_type()
    assert doc.read_xml(
        '<doclang version="0.7">'
        '<text><location value="10"/><location value="20.5"/>'
        '<location value="30"/><location value="40"/>First</text>'
        "<page_break/>"
        '<table><cell><location value="0"/><location value="100"/>'
        '<location value="500"/><location value="1000"/>Second</cell></table>'
        "<text>Unlocated</text>"
        "</doclang>"
    )

    first = "/doclang[1]/text[1]"
    second = "/doclang[1]/table[1]/cell[1]"
    assert doc.bounding_box(first) == ((10.0, 20.5), (30.0, 40.0))
    assert doc.page_number(first) == 1
    assert doc.bounding_box(second) == ((0.0, 100.0), (500.0, 1000.0))
    assert doc.page_number(second) == 2
    assert doc.page_number("/doclang[1]/text[2]") == 2
    assert doc.bounding_box("/doclang[1]/text[2]") is None
    assert doc.bounding_box("/doclang[1]/missing[1]") is None
    assert "not found" in doc.last_error()
    assert doc.page_number("/doclang[1]/missing[1]") is None

    items = list(doc.iterate_items())
    assert [xpath for xpath, _, _, _ in items] == [
        "/doclang[1]/text[1]",
        "/doclang[1]/page_break[1]",
        "/doclang[1]/table[1]",
        "/doclang[1]/text[2]",
    ]
    assert items[0][2:] == (1, [10, 21, 30, 40])
    assert items[1][2:] == (None, None)
    assert items[2][2:] == (2, None)
    assert items[3][2:] == (2, None)

    children = list(doc.iterate_items(xpath="/doclang[1]/table[1]"))
    assert len(children) == 1
    assert children[0][0] == "/doclang[1]/table[1]/cell[1]"
    assert children[0][2:] == (2, [0, 100, 500, 1000])
    assert list(doc.iterate_items_on_page(1)) == items[:1]
    assert list(doc.iterate_items_on_page(2)) == items[2:]
    assert list(doc.iterate_items_on_page(3)) == []
    with pytest.raises(ValueError, match="not found"):
        doc.iterate_items(xpath="/doclang[1]/missing[1]")
    with pytest.raises(ValueError, match="at least 1"):
        doc.iterate_items_on_page(0)


def test_doclangx_document_is_a_doclang_document():
    doc = DocLangXDocument()
    assert isinstance(doc, DoclangDocument)

    assert doc.read_xml(
        '<doclang version="0.7"><heading>Title</heading><text>Body</text></doclang>'
    )
    assert doc.valid()
    assert doc.xml().startswith("<doclang")
    assert [element["name"] for element in doc] == ["heading", "text"]
    items = doc.iterate_items()
    assert iter(items) is items
    assert next(items)[0] == "/doclang[1]/heading[1]"
    xpath, item, page_no, bbox = next(items)
    assert xpath == "/doclang[1]/text[1]"
    assert item["text"] == "Body"
    assert page_no == 1
    assert bbox is None
    with pytest.raises(StopIteration):
        next(items)


def test_doclang_iterator_keeps_document_alive():
    doc = DoclangDocument("<doclang><text>One</text><text>Two</text></doclang>")
    items = doc.iterate_items()
    del doc

    assert [(xpath, item["text"]) for xpath, item, _, _ in items] == [
        ("/doclang[1]/text[1]", "One"),
        ("/doclang[1]/text[2]", "Two"),
    ]


@pytest.mark.parametrize("document_type", [DoclangDocument, DocLangXDocument])
def test_doclang_iterate_items_expands_flat_and_nested_lists(document_type):
    doc = document_type()
    assert doc.read_xml(
        '<doclang version="0.7">'
        "<list><ldiv><marker>•</marker></ldiv>"
        '<location value="10"/><location value="20"/>'
        '<location value="30"/><location value="40"/>First &amp; more'
        "<ldiv><marker>•</marker></ldiv>"
        '<location value="50"/><location value="60"/>'
        '<location value="70"/><location value="80"/>Second</list>'
        "<page_break/>"
        "<list><ldiv><marker>a.</marker>"
        '<location value="90"/><location value="100"/>'
        '<location value="110"/><location value="120"/>Nested one</ldiv>'
        "<ldiv><marker>b.</marker>Nested two</ldiv></list>"
        "</doclang>"
    )

    items = list(doc.iterate_items())
    assert [xpath for xpath, _, _, _ in items] == [
        "/doclang[1]/list[1]/ldiv[1]",
        "/doclang[1]/list[1]/ldiv[2]",
        "/doclang[1]/page_break[1]",
        "/doclang[1]/list[2]/ldiv[1]",
        "/doclang[1]/list[2]/ldiv[2]",
    ]
    assert [item["text"] for _, item, _, _ in items if item["name"] == "list_item"] == [
        "First & more",
        "Second",
        "Nested one",
        "Nested two",
    ]
    assert [entry[2:] for entry in items] == [
        (1, [10, 20, 30, 40]),
        (1, [50, 60, 70, 80]),
        (None, None),
        (2, [90, 100, 110, 120]),
        (2, None),
    ]
    for _, item, _, _ in items:
        if item["name"] == "list_item":
            assert ElementTree.fromstring(item["xml"]).tag == "list_item"

    assert list(doc.iterate_items(xpath="/doclang[1]/list[1]")) == items[:2]
    assert list(doc.iterate_items(xpath="/doclang[1]/list[2]")) == items[3:]
    assert list(doc.iterate_items_on_page(1)) == items[:2]
    assert list(doc.iterate_items_on_page(2)) == items[3:]
    assert [item["name"] for item in doc] == ["list", "page_break", "list"]


def test_doclangx_document_read_xml_drops_previous_dclx_state(tmp_path):
    input_path = tmp_path / "input.dclx"
    create_dclx(input_path)

    doc = DocLangXDocument()
    assert doc.read(str(input_path))
    assert doc.has_archive()
    assert doc.has_annotations()

    # a bare DCLG buffer has no archive and no annotations to carry over
    assert doc.read_xml('<doclang version="0.7"><text>Body</text></doclang>')
    assert doc.valid()
    assert not doc.has_archive()
    assert not doc.has_annotations()
    assert doc.overview()["properties"] == 0
    assert doc.overview()["instances"] == 0


def test_doclangx_document_at_resolves_doclang_paths():
    doc = DocLangXDocument()
    xml = (
        '<doclang version="0.7">'
        "<text>Body text</text>"
        "<table><fcel/>cell-a<lcel/><nl/><ched/>head</table>"
        "<picture>"
        "<caption>Figure caption</caption>"
        '<src uri="assets/image.png"/>'
        "<list><ldiv><marker>a.</marker></ldiv>List body</list>"
        "<text>Picture text</text>"
        "</picture>"
        "</doclang>"
    )

    assert doc.read_xml(xml)
    assert doc.at(xpath="/doclang[1]/text[1]") == "Body text"
    assert doc.at(xpath="/doclang[1]/text[1]", mode="text") == "Body text"
    assert "<text>Body text</text>" in doc.at(
        xpath="/doclang[1]/text[1]", mode="doclang"
    )
    assert doc.at(xpath="/doclang[1]/table[1]/text()[1]") == "cell-a"
    assert doc.at(xpath="/doclang[1]/table[1]", mode="text") == "cell-a\nhead"
    assert (
        doc.at(xpath="/doclang[1]/picture[1]", mode="text")
        == "Figure caption\nList body\nPicture text"
    )
    assert doc.at(xpath="/doclang[1]/missing[1]") == ""
    assert "not found" in doc.last_error()

    with pytest.raises(TypeError):
        doc.at("/doclang[1]/text[1]")


def test_doclangx_nlp_reuses_initialised_models_across_documents():
    nlp = DocLangXNlp("")
    assert nlp.initialised()
    assert nlp.model_expr() == ""
    assert nlp.models() == []

    first = DocLangXDocument()
    second = DocLangXDocument()
    assert first.read_xml('<doclang version="0.7"><text>First text</text></doclang>')
    assert second.read_xml('<doclang version="0.7"><text>Second text</text></doclang>')

    assert nlp.apply(first, progress_every=0)
    assert nlp.apply(second, progress_every=0)
    assert first.overview()["instances"] == 0
    assert second.overview()["instances"] == 0


def test_doclangx_nlp_reports_uninitialised_apply():
    nlp = DocLangXNlp()
    doc = DocLangXDocument()
    assert doc.read_xml('<doclang version="0.7"><text>Body text</text></doclang>')

    assert not nlp.apply(doc, progress_every=0)
    assert nlp.last_error() == "models have not been initialised"
    assert doc.last_error() == "models have not been initialised"
