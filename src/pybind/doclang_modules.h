//-*-C++-*-

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind/utils/pybind11_json.h>

#include <pybind/doclang/doclang_document.h>
#include <pybind/doclang/doclang_x_document.h>
#include <pybind/doclang/doclang_x_nlp.h>

PYBIND11_MODULE(andromeda_doclang, m) {

  pybind11::class_<andromeda_py::DoclangDocument::Iterator>(m, "_DoclangIterator")
    .def("__iter__", [](andromeda_py::DoclangDocument::Iterator& iterator)
         -> andromeda_py::DoclangDocument::Iterator&
         {
           return iterator;
         }, pybind11::return_value_policy::reference_internal)
    .def("__next__", &andromeda_py::DoclangDocument::Iterator::next);

  pybind11::class_<andromeda_py::DoclangDocument>(m, "DoclangDocument")
    .def(pybind11::init())
    .def(pybind11::init<const std::string&>(), pybind11::arg("dclg"))
    .def("read_xml", &andromeda_py::DoclangDocument::read_xml,
         pybind11::arg("dclg"))
    .def("valid", &andromeda_py::DoclangDocument::valid)
    .def("xml", &andromeda_py::DoclangDocument::xml)
    .def("last_error", &andromeda_py::DoclangDocument::last_error)
    .def("at", &andromeda_py::DoclangDocument::at,
         pybind11::kw_only(),
         pybind11::arg("xpath"),
         pybind11::arg("mode") = "auto")
    .def("bounding_box", &andromeda_py::DoclangDocument::bounding_box,
         pybind11::arg("xpath"))
    .def("page_number", &andromeda_py::DoclangDocument::page_number,
         pybind11::arg("xpath"))
    .def("iterate_items", &andromeda_py::DoclangDocument::iterate_items,
         pybind11::arg("xpath") = pybind11::none())
    .def("iterate_items_on_page",
         &andromeda_py::DoclangDocument::iterate_items_on_page,
         pybind11::arg("page_no"))
    .def("__iter__", &andromeda_py::DoclangDocument::iter);

  pybind11::class_<andromeda_py::DocLangXDocument,
                   andromeda_py::DoclangDocument>(m, "DocLangXDocument")
    .def(pybind11::init())
    .def_static("hash", &andromeda_py::DocLangXDocument::hash,
         pybind11::arg("text"))

    .def("read", &andromeda_py::DocLangXDocument::read,
         pybind11::arg("path"))
    .def("read_xml", &andromeda_py::DocLangXDocument::read_xml,
         pybind11::arg("xml"))
    .def("write", &andromeda_py::DocLangXDocument::write,
         pybind11::arg("path"))
    .def("apply_nlp", &andromeda_py::DocLangXDocument::apply_nlp,
         pybind11::arg("models"),
         pybind11::arg("progress_every") = 25)
    .def("materialize_edges", &andromeda_py::DocLangXDocument::materialize_edges,
         pybind11::arg("derived_entity_mode") = "terms")

    .def("has_archive", &andromeda_py::DocLangXDocument::has_archive)
    .def("has_annotations", &andromeda_py::DocLangXDocument::has_annotations)

    .def("source_path", &andromeda_py::DocLangXDocument::source_path)

    .def("archive_paths", &andromeda_py::DocLangXDocument::archive_paths)
    .def("annotation_paths", &andromeda_py::DocLangXDocument::annotation_paths)
    .def("document_reference", &andromeda_py::DocLangXDocument::document_reference)
    .def("references", &andromeda_py::DocLangXDocument::references)
    .def("summary", &andromeda_py::DocLangXDocument::summary)
    .def("toc", &andromeda_py::DocLangXDocument::toc)
    .def("concepts", &andromeda_py::DocLangXDocument::concepts)
    .def("set_document_reference", &andromeda_py::DocLangXDocument::set_document_reference,
         pybind11::arg("bibtex"))
    .def("set_references", &andromeda_py::DocLangXDocument::set_references,
         pybind11::arg("bibtex"))
    .def("set_document_summary", &andromeda_py::DocLangXDocument::set_document_summary,
         pybind11::arg("dclg"))
    .def("set_toc", &andromeda_py::DocLangXDocument::set_toc,
         pybind11::arg("dclg"))
    .def("set_concepts", &andromeda_py::DocLangXDocument::set_concepts,
         pybind11::arg("dclg"))
    .def("clear_document_reference", &andromeda_py::DocLangXDocument::clear_document_reference)
    .def("clear_references", &andromeda_py::DocLangXDocument::clear_references)
    .def("clear_document_summary", &andromeda_py::DocLangXDocument::clear_document_summary)
    .def("clear_toc", &andromeda_py::DocLangXDocument::clear_toc)
    .def("clear_concepts", &andromeda_py::DocLangXDocument::clear_concepts)

    .def("overview", &andromeda_py::DocLangXDocument::overview)
    .def("properties", &andromeda_py::DocLangXDocument::properties)
    .def("entities", &andromeda_py::DocLangXDocument::entities)
    .def("instances", &andromeda_py::DocLangXDocument::instances)
    .def("relations", &andromeda_py::DocLangXDocument::relations)
    .def("edges", &andromeda_py::DocLangXDocument::edges)

    .def("query_properties", &andromeda_py::DocLangXDocument::query_properties,
         pybind11::arg("type") = "",
         pybind11::arg("label") = "",
         pybind11::arg("subj_path") = "",
         pybind11::arg("min_conf") = 0.0)
    .def("query_entities", &andromeda_py::DocLangXDocument::query_entities,
         pybind11::arg("type") = "",
         pybind11::arg("subtype") = "",
         pybind11::arg("name") = "",
         pybind11::arg("name_contains") = "",
         pybind11::arg("entity_kind") = "",
         pybind11::arg("min_count") = 0)
    .def("query_instances", &andromeda_py::DocLangXDocument::query_instances,
         pybind11::arg("type") = "",
         pybind11::arg("subtype") = "",
         pybind11::arg("name") = "",
         pybind11::arg("name_contains") = "",
         pybind11::arg("subj_path") = "",
         pybind11::arg("min_conf") = 0.0,
         pybind11::arg("entity_hash") = 0)
    .def("query_relations", &andromeda_py::DocLangXDocument::query_relations,
         pybind11::arg("name") = "",
         pybind11::arg("name_i") = "",
         pybind11::arg("name_j") = "",
         pybind11::arg("name_contains") = "",
         pybind11::arg("min_conf") = 0.0)
    .def("query_edges", &andromeda_py::DocLangXDocument::query_edges,
         pybind11::arg("name") = "",
         pybind11::arg("hash_i") = 0,
         pybind11::arg("hash_j") = 0,
         pybind11::arg("min_count") = 0);

  pybind11::class_<andromeda_py::DocLangXNlp>(m, "DocLangXNlp")
    .def(pybind11::init())
    .def(pybind11::init<const std::string&>(),
         pybind11::arg("models"))
    .def("initialise", &andromeda_py::DocLangXNlp::initialise,
         pybind11::arg("models"))
    .def("apply", &andromeda_py::DocLangXNlp::apply,
         pybind11::arg("doc"),
         pybind11::arg("progress_every") = 25)
    .def("initialised", &andromeda_py::DocLangXNlp::initialised)
    .def("model_expr", &andromeda_py::DocLangXNlp::model_expr)
    .def("models", &andromeda_py::DocLangXNlp::models)
    .def("last_error", &andromeda_py::DocLangXNlp::last_error);
}
