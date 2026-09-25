//-*-C++-*-

#ifndef PYBIND_ANDROMEDA_DOCLANG_X_NLP_H_
#define PYBIND_ANDROMEDA_DOCLANG_X_NLP_H_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <andromeda.h>

namespace andromeda_py
{

  class DocLangXDocument;

  namespace detail
  {
    andromeda::doclang::dclx_document& mutable_doclang_x_document(
      DocLangXDocument& document);
  }

  class DocLangXNlp
  {
  public:

    DocLangXNlp();
    explicit DocLangXNlp(const std::string& models);
    ~DocLangXNlp();

    bool initialise(const std::string& models);
    bool apply(DocLangXDocument& doc, std::size_t progress_every=25);

    bool initialised() const;
    std::string model_expr() const;
    std::string last_error() const;
    std::vector<std::string> models() const;

  private:

    bool is_initialised;
    std::string model_expr_value;
    std::string last_error_value;
    std::vector<std::shared_ptr<andromeda::base_nlp_model> > nlp_models;
  };

  inline DocLangXNlp::DocLangXNlp():
    is_initialised(false),
    model_expr_value(""),
    last_error_value(""),
    nlp_models({})
  {}

  inline DocLangXNlp::DocLangXNlp(const std::string& models):
    DocLangXNlp()
  {
    initialise(models);
  }

  inline DocLangXNlp::~DocLangXNlp()
  {}

  inline bool DocLangXNlp::initialise(const std::string& models)
  {
    nlp_models.clear();
    model_expr_value = "";
    last_error_value = "";
    is_initialised = false;

    if(not andromeda::to_models(models, nlp_models, true))
      {
        last_error_value = "could not initialise models: " + models;
        return false;
      }

    model_expr_value = models;
    is_initialised = true;
    return true;
  }

  inline bool DocLangXNlp::apply(DocLangXDocument& document,
                                 std::size_t progress_every)
  {
    if(not is_initialised)
      {
        last_error_value = "models have not been initialised";
        detail::mutable_doclang_x_document(document).set_last_error(last_error_value);
        return false;
      }

    andromeda::doclang::nlp_apply_options options;
    options.document_name = detail::mutable_doclang_x_document(document).get_source_path().string();
    options.progress_every = progress_every;

    andromeda::doclang::nlp_apply_result result;
    return andromeda::doclang::apply_models(detail::mutable_doclang_x_document(document),
                                            nlp_models,
                                            options,
                                            result);
  }

  inline bool DocLangXNlp::initialised() const
  {
    return is_initialised;
  }

  inline std::string DocLangXNlp::model_expr() const
  {
    return model_expr_value;
  }

  inline std::string DocLangXNlp::last_error() const
  {
    return last_error_value;
  }

  inline std::vector<std::string> DocLangXNlp::models() const
  {
    std::vector<std::string> names;
    for(const auto& model:nlp_models)
      {
        names.push_back(model->get_key());
      }

    return names;
  }

}

#include <pybind/doclang/doclang_x_document.h>

#endif
