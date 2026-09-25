//-*-C++-*-

#ifndef PYBIND_ANDROMEDA_DOCLANG_X_DOCUMENT_H_
#define PYBIND_ANDROMEDA_DOCLANG_X_DOCUMENT_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <andromeda.h>
#include <pybind/utils/pybind11_json.h>
#include <pybind/doclang/doclang_document.h>
#include <pybind/doclang/doclang_x_nlp.h>

namespace andromeda_py
{

  class DocLangXDocument: public DoclangDocument
  {
  public:

    DocLangXDocument();
    ~DocLangXDocument() override;

    static std::uint64_t hash(const std::string& text);

    bool read(const std::string& path);
    bool read_xml(const std::string& xml) override;
    bool write(const std::string& path);
    bool apply_nlp(const std::string& models, std::size_t progress_every=25);
    void materialize_edges(const std::string& derived_entity_mode="terms");
    andromeda::doclang::dclx_document& mutable_document();

    bool has_archive() const;
    bool has_annotations() const;

    std::string source_path() const;

    std::vector<std::string> archive_paths() const;
    std::vector<std::string> annotation_paths() const;

    std::optional<std::string> document_reference() const;
    std::optional<std::string> references() const;
    std::optional<DoclangDocument> summary() const;
    std::optional<DoclangDocument> toc() const;
    std::optional<DoclangDocument> concepts() const;

    void set_document_reference(const std::string& bibtex);
    void set_references(const std::string& bibtex);
    bool set_document_summary(const std::string& dclg);
    bool set_toc(const std::string& dclg);
    bool set_concepts(const std::string& dclg);

    void clear_document_reference();
    void clear_references();
    void clear_document_summary();
    void clear_toc();
    void clear_concepts();

    nlohmann::json overview() const;
    pybind11::object properties() const;
    pybind11::object entities() const;
    pybind11::object instances() const;
    pybind11::object relations() const;
    pybind11::object edges() const;

    pybind11::object query_properties(const std::string& type="",
                                      const std::string& label="",
                                      const std::string& subj_path="",
                                      float min_conf=0.0) const;

    pybind11::object query_entities(const std::string& type="",
                                    const std::string& subtype="",
                                    const std::string& name="",
                                    const std::string& name_contains="",
                                    const std::string& entity_kind="",
                                    std::size_t min_count=0) const;

    pybind11::object query_instances(const std::string& type="",
                                     const std::string& subtype="",
                                     const std::string& name="",
                                     const std::string& name_contains="",
                                     const std::string& subj_path="",
                                     float min_conf=0.0,
                                     std::uint64_t entity_hash=0) const;

    pybind11::object query_relations(const std::string& name="",
                                     const std::string& name_i="",
                                     const std::string& name_j="",
                                     const std::string& name_contains="",
                                     float min_conf=0.0) const;

    pybind11::object query_edges(const std::string& name="",
                                 std::uint64_t hash_i=0,
                                 std::uint64_t hash_j=0,
                                 std::size_t min_count=0) const;

  private:

    static pybind11::object dataframe_from_table(const nlohmann::json& table);
    static std::string pandas_dtype_for_column(const std::string& column);

    nlohmann::json properties_table() const;
    nlohmann::json entities_table() const;
    nlohmann::json instances_table() const;
    nlohmann::json relations_table() const;
    nlohmann::json edges_table() const;

  private:

    explicit DocLangXDocument(std::shared_ptr<andromeda::doclang::dclx_document> value);

    std::shared_ptr<andromeda::doclang::dclx_document> doc;
  };

  namespace detail
  {
    inline andromeda::doclang::dclx_document& mutable_doclang_x_document(
      DocLangXDocument& document)
    {
      return document.mutable_document();
    }
  }

  inline DocLangXDocument::DocLangXDocument():
    DocLangXDocument(std::make_shared<andromeda::doclang::dclx_document>())
  {}

  inline DocLangXDocument::DocLangXDocument(
    std::shared_ptr<andromeda::doclang::dclx_document> value):
    DoclangDocument(value),
    doc(std::move(value))
  {}

  inline DocLangXDocument::~DocLangXDocument()
  {}

  inline std::uint64_t DocLangXDocument::hash(const std::string& text)
  {
    return andromeda::doclang::dclx_document::hash(text);
  }

  inline bool DocLangXDocument::read(const std::string& path)
  {
    return andromeda::doclang::reader::read(path, *doc);
  }

  inline bool DocLangXDocument::read_xml(const std::string& xml)
  {
    // a plain DCLG buffer carries no archive or annotations: drop the DCLX
    // state the previous document left behind before reparsing
    doc->clear();
    return andromeda::doclang::reader::read_dclg_buffer(xml, *doc);
  }

  inline bool DocLangXDocument::write(const std::string& path)
  {
    return andromeda::doclang::writer::write_dclx(path, *doc);
  }

  inline bool DocLangXDocument::apply_nlp(const std::string& models,
                                          std::size_t progress_every)
  {
    DocLangXNlp nlp;
    if(not nlp.initialise(models))
      {
        doc->set_last_error(nlp.last_error());
        return false;
      }

    return nlp.apply(*this, progress_every);
  }

  inline void DocLangXDocument::materialize_edges(const std::string& derived_entity_mode)
  {
    doc->materialize_edges(derived_entity_mode);
  }

  inline andromeda::doclang::dclx_document& DocLangXDocument::mutable_document()
  {
    return *doc;
  }

  inline bool DocLangXDocument::has_archive() const
  {
    return doc->has_archive();
  }

  inline bool DocLangXDocument::has_annotations() const
  {
    return doc->has_annotations();
  }

  inline std::string DocLangXDocument::source_path() const
  {
    return doc->get_source_path().string();
  }

  inline std::vector<std::string> DocLangXDocument::archive_paths() const
  {
    if(not doc->has_archive())
      {
        return {};
      }

    return doc->artifacts().paths();
  }

  inline std::vector<std::string> DocLangXDocument::annotation_paths() const
  {
    return {
      andromeda::doclang::PROPERTIES_CSV,
      andromeda::doclang::INSTANCES_CSV,
      andromeda::doclang::ENTITIES_CSV,
      andromeda::doclang::RELATIONS_CSV,
      andromeda::doclang::EDGES_CSV,
      andromeda::doclang::DOCUMENT_REFERENCE_BIB,
      andromeda::doclang::REFERENCES_BIB,
      andromeda::doclang::SUMMARY_DCLG,
      andromeda::doclang::TOC_DCLG,
      andromeda::doclang::CONCEPTS_DCLG
    };
  }

  inline std::optional<std::string> DocLangXDocument::document_reference() const
  {
    return doc->get_document_reference();
  }

  inline std::optional<std::string> DocLangXDocument::references() const
  {
    return doc->get_references();
  }

  inline std::optional<DoclangDocument> to_sidecar_wrapper(
    const andromeda::doclang::dclx_document::sidecar_type& sidecar)
  {
    if(not sidecar.has_value())
      {
        return std::nullopt;
      }

    // the wrapper shares the parsed sidecar allocation, it does not copy it
    return DoclangDocument(sidecar.value());
  }

  inline std::optional<DoclangDocument> DocLangXDocument::summary() const
  {
    return to_sidecar_wrapper(doc->get_summary());
  }

  inline std::optional<DoclangDocument> DocLangXDocument::toc() const
  {
    return to_sidecar_wrapper(doc->get_toc());
  }

  inline std::optional<DoclangDocument> DocLangXDocument::concepts() const
  {
    return to_sidecar_wrapper(doc->get_concepts());
  }

  inline void DocLangXDocument::set_document_reference(const std::string& bibtex)
  {
    doc->set_document_reference(bibtex);
  }

  inline void DocLangXDocument::set_references(const std::string& bibtex)
  {
    doc->set_references(bibtex);
  }

  inline bool DocLangXDocument::set_document_summary(const std::string& dclg)
  {
    return andromeda::doclang::set_summary_dclg(*doc, dclg);
  }

  inline bool DocLangXDocument::set_toc(const std::string& dclg)
  {
    return andromeda::doclang::set_toc_dclg(*doc, dclg);
  }

  inline bool DocLangXDocument::set_concepts(const std::string& dclg)
  {
    return andromeda::doclang::set_concepts_dclg(*doc, dclg);
  }

  inline void DocLangXDocument::clear_document_reference()
  {
    doc->clear_document_reference();
  }

  inline void DocLangXDocument::clear_references()
  {
    doc->clear_references();
  }

  inline void DocLangXDocument::clear_document_summary()
  {
    doc->clear_summary();
  }

  inline void DocLangXDocument::clear_toc()
  {
    doc->clear_toc();
  }

  inline void DocLangXDocument::clear_concepts()
  {
    doc->clear_concepts();
  }

  inline nlohmann::json DocLangXDocument::overview() const
  {
    return nlohmann::json::object({
        {"valid", valid()},
        {"source_path", source_path()},
        {"has_archive", has_archive()},
        {"has_annotations", has_annotations()},
        {"properties", doc->get_properties().size()},
        {"instances", doc->get_instances().size()},
        {"entities", doc->get_entities().size()},
        {"relations", doc->get_relations().size()},
        {"edges", doc->get_edges().size()},
        {"has_document_reference", doc->has_document_reference()},
        {"has_references", doc->has_references()},
        {"has_summary", doc->has_summary()},
        {"has_toc", doc->has_toc()},
        {"has_concepts", doc->has_concepts()}
      });
  }

  inline pybind11::object DocLangXDocument::dataframe_from_table(
    const nlohmann::json& table)
  {
    pybind11::module_ pandas = pybind11::module_::import("pandas");
    pybind11::object dataframe = pandas.attr("DataFrame")(
      pyjson::from_json(table.at("data")),
      pybind11::arg("columns") = pyjson::from_json(table.at("headers")));

    pybind11::dict dtypes;
    for(const auto& header:table.at("headers"))
      {
        const std::string column = header.get<std::string>();
        const std::string dtype = pandas_dtype_for_column(column);
        if(not dtype.empty())
          {
            dtypes[pybind11::str(column)] = pybind11::str(dtype);
          }
      }

    return dataframe.attr("astype")(dtypes);
  }

  inline std::string DocLangXDocument::pandas_dtype_for_column(
    const std::string& column)
  {
    if(column=="type" or
       column=="subtype" or
       column=="subj_name" or
       column=="subj_path" or
       column=="label" or
       column=="entity_kind" or
       column=="name" or
       column=="original" or
       column=="parent" or
       column=="name_i" or
       column=="name_j")
      {
        return "string";
      }

    if(column=="confidence" or column=="conf" or column=="probability")
      {
        return "Float32";
      }

    if(column=="wtok-match")
      {
        return "boolean";
      }

    if(column=="subj_hash" or
       column=="hash" or
       column=="ihash" or
       column=="hash_i" or
       column=="hash_j" or
       column=="parent_hash" or
       column=="count" or
       column=="coor_i" or
       column=="coor_j" or
       column=="char_i" or
       column=="char_j" or
       column=="ctok_i" or
       column=="ctok_j" or
       column=="wtok_i" or
       column=="wtok_j" or
       column=="flvr" or
       column=="total-count")
      {
        return "UInt64";
      }

    return "";
  }

  inline nlohmann::json DocLangXDocument::properties_table() const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_property::HEADERS;
    result["data"] = nlohmann::json::array();

    for(auto prop:doc->get_properties())
      {
        result["data"].push_back(prop.to_json_row());
      }

    return result;
  }

  inline nlohmann::json DocLangXDocument::entities_table() const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_entity::HEADERS;
    result["data"] = nlohmann::json::array();

    for(const auto& ent:doc->get_entities())
      {
        result["data"].push_back(ent.to_json_row());
      }

    return result;
  }

  inline nlohmann::json DocLangXDocument::instances_table() const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_instance::HEADERS;
    result["data"] = nlohmann::json::array();

    for(const auto& inst:doc->get_instances())
      {
        result["data"].push_back(inst.to_json_row());
      }

    return result;
  }

  inline nlohmann::json DocLangXDocument::relations_table() const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_relation::headers();
    result["data"] = nlohmann::json::array();

    for(auto rel:doc->get_relations())
      {
        result["data"].push_back(rel.to_json_row());
      }

    return result;
  }

  inline nlohmann::json DocLangXDocument::edges_table() const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_graph_edge::HEADERS;
    result["data"] = nlohmann::json::array();

    for(auto edge:doc->get_edges())
      {
        result["data"].push_back(edge.to_json_row());
      }

    return result;
  }

  inline pybind11::object DocLangXDocument::properties() const
  {
    return dataframe_from_table(properties_table());
  }

  inline pybind11::object DocLangXDocument::entities() const
  {
    return dataframe_from_table(entities_table());
  }

  inline pybind11::object DocLangXDocument::instances() const
  {
    return dataframe_from_table(instances_table());
  }

  inline pybind11::object DocLangXDocument::relations() const
  {
    return dataframe_from_table(relations_table());
  }

  inline pybind11::object DocLangXDocument::edges() const
  {
    return dataframe_from_table(edges_table());
  }

  inline pybind11::object DocLangXDocument::query_properties(
    const std::string& type,
    const std::string& label,
    const std::string& subj_path,
    float min_conf) const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_property::HEADERS;
    result["data"] = nlohmann::json::array();

    for(auto prop:doc->get_properties())
      {
        if((not type.empty()) and prop.get_type()!=type)
          {
            continue;
          }
        if((not label.empty()) and prop.get_label()!=label)
          {
            continue;
          }
        if((not subj_path.empty()) and prop.get_subj_path()!=subj_path)
          {
            continue;
          }
        if(prop.get_conf()<min_conf)
          {
            continue;
          }

        result["data"].push_back(prop.to_json_row());
      }

    return dataframe_from_table(result);
  }

  inline pybind11::object DocLangXDocument::query_entities(
    const std::string& type,
    const std::string& subtype,
    const std::string& name,
    const std::string& name_contains,
    const std::string& entity_kind,
    std::size_t min_count) const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_entity::HEADERS;
    result["data"] = nlohmann::json::array();

    for(const auto& ent:doc->get_entities())
      {
        if((not type.empty()) and ent.get_type()!=type)
          {
            continue;
          }
        if((not subtype.empty()) and ent.get_subtype()!=subtype)
          {
            continue;
          }
        if((not name.empty()) and ent.get_name()!=name)
          {
            continue;
          }
        if((not name_contains.empty()) and
           ent.get_name().find(name_contains)==std::string::npos)
          {
            continue;
          }
        if((not entity_kind.empty()) and ent.get_entity_kind()!=entity_kind)
          {
            continue;
          }
        if(ent.get_count()<min_count)
          {
            continue;
          }

        result["data"].push_back(ent.to_json_row());
      }

    return dataframe_from_table(result);
  }

  inline pybind11::object DocLangXDocument::query_instances(
    const std::string& type,
    const std::string& subtype,
    const std::string& name,
    const std::string& name_contains,
    const std::string& subj_path,
    float min_conf,
    std::uint64_t entity_hash) const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_instance::HEADERS;
    result["data"] = nlohmann::json::array();

    for(const auto& inst:doc->get_instances())
      {
        if((not type.empty()) and inst.get_type()!=type)
          {
            continue;
          }
        if((not subtype.empty()) and inst.get_subtype()!=subtype)
          {
            continue;
          }
        if((not name.empty()) and inst.get_name()!=name)
          {
            continue;
          }
        if((not name_contains.empty()) and
           inst.get_name().find(name_contains)==std::string::npos)
          {
            continue;
          }
        if((not subj_path.empty()) and inst.get_subj_path()!=subj_path)
          {
            continue;
          }
        if(entity_hash!=0 and inst.get_ehash()!=entity_hash)
          {
            continue;
          }
        if(inst.get_conf()<min_conf)
          {
            continue;
          }

        result["data"].push_back(inst.to_json_row());
      }

    return dataframe_from_table(result);
  }

  inline pybind11::object DocLangXDocument::query_relations(
    const std::string& name,
    const std::string& name_i,
    const std::string& name_j,
    const std::string& name_contains,
    float min_conf) const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_relation::headers();
    result["data"] = nlohmann::json::array();

    for(auto rel:doc->get_relations())
      {
        if((not name.empty()) and rel.get_name()!=name)
          {
            continue;
          }
        if((not name_i.empty()) and rel.get_name_i()!=name_i)
          {
            continue;
          }
        if((not name_j.empty()) and rel.get_name_j()!=name_j)
          {
            continue;
          }
        if((not name_contains.empty()) and
           rel.get_name_i().find(name_contains)==std::string::npos and
           rel.get_name_j().find(name_contains)==std::string::npos)
          {
            continue;
          }
        if(rel.get_conf()<min_conf)
          {
            continue;
          }

        result["data"].push_back(rel.to_json_row());
      }

    return dataframe_from_table(result);
  }

  inline pybind11::object DocLangXDocument::query_edges(
    const std::string& name,
    std::uint64_t hash_i,
    std::uint64_t hash_j,
    std::size_t min_count) const
  {
    nlohmann::json result = nlohmann::json::object();
    result["headers"] = andromeda::base_graph_edge::HEADERS;
    result["data"] = nlohmann::json::array();

    for(auto edge:doc->get_edges())
      {
        if((not name.empty()) and edge.get_name()!=name)
          {
            continue;
          }
        if(hash_i!=0 and edge.get_hash_i()!=hash_i)
          {
            continue;
          }
        if(hash_j!=0 and edge.get_hash_j()!=hash_j)
          {
            continue;
          }
        if(edge.get_count()<min_count)
          {
            continue;
          }

        result["data"].push_back(edge.to_json_row());
      }

    return dataframe_from_table(result);
  }

}

#endif
