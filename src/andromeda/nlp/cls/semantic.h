//-*-C++-*-

#ifndef ANDROMEDA_MODELS_CLASSIFIERS_SEMANTIC_H_
#define ANDROMEDA_MODELS_CLASSIFIERS_SEMANTIC_H_

namespace andromeda
{
  /*
    The goal of this classifier is to find the semantic meaning of the
    text in the document. To that end, we have several classes:

    1. header
    2. meta-data: mostly authors, affiliations, etc
    3. reference
    4. text

    The goal is to use the semantic labels downstream to extract meta-data
    items and parse the references.
  */
  template<>
  class nlp_model<CLS, SEMANTIC>: public fasttext_supervised_model
  {
    const static model_type type = CLS;
    const static model_name name = SEMANTIC;

    const static inline std::set<std::string> known_headers
    = {"abstract", "introduction", "references", "conclusion"};

  public:

    nlp_model();
    explicit nlp_model(std::filesystem::path model_file);

    ~nlp_model();

    virtual std::set<model_name> get_dependencies() { return dependencies; }

    virtual model_type get_type() { return CLS; }
    virtual model_name get_name() { return SEMANTIC; }

    template<typename subject_type>
    bool get(subject_type& subj, base_property& prop);

    virtual bool preprocess(const subject<TEXT>& subj, std::string& text);
    virtual bool preprocess(const subject<TABLE>& subj, std::string& text);

    virtual bool apply(subject<TEXT>& subj);
    virtual bool apply(subject<TABLE>& subj);
    virtual bool apply(subject<DOCUMENT>& subj);

  private:

    void initialise();

    void initialise_regex();
    void initialise_model();

    void get_semantic_mapping();

  private:

    const static std::set<model_name> dependencies;

    //std::filesystem::path resources_dir;
    std::filesystem::path model_file;

    std::vector<pcre2_expr> author_list, authors;
    //std::vector<pcre2_expr> table_refs, figure_refs;
    std::vector<pcre2_expr> caption_refs;
  };

  const std::set<model_name> nlp_model<CLS, SEMANTIC>::dependencies = {LINK, NUMVAL};

  nlp_model<CLS, SEMANTIC>::nlp_model():
    fasttext_supervised_model(),
    model_file(glm_variables::get_fst_dir() / "semantic/fst_semantic.bin")
  {
    initialise();
  }

  nlp_model<CLS, SEMANTIC>::nlp_model(std::filesystem::path model_file_):
    fasttext_supervised_model(),
    model_file(std::move(model_file_))
  {
    initialise();
  }

  nlp_model<CLS, SEMANTIC>::~nlp_model()
  {}

  void nlp_model<CLS, SEMANTIC>::initialise()
  {
    initialise_regex();

    initialise_model();
  }

  void nlp_model<CLS, SEMANTIC>::initialise_regex()
  {
    // Yinhan Liu , Myle Ott , Naman Goyal , J . S . - A . Du , Mandar Joshi , Danqi Chen , Omer Levy , Mike Lewis , Luke Zettlemoyer , and Veselin Stoyanov
    {
      std::string authors_str = R"(((?P<author>(([A-Z][a-z]+\s)(([A-Z][a-z]+|[A-Z]\s\.|\-|\')\s)*([A-Z][a-z]+)))\s((\,|and|\&)\s)+)+(?P<lauthor>(([A-Z][a-z]+\s)(([A-Z][a-z]+|[A-Z]\s\.|\-|\')\s)*([A-Z][a-z]+))))";

      pcre2_expr expr(this->get_key(), "__author_list__", authors_str);
      author_list.push_back(expr);
    }

    // Y . Liu , M . Ott , N . Goyal , J . S . - A . Du , M . Joshi , D . Chen , O . Levy , M . Lewis , L . Zettlemoyer
    {
      std::string authors_str = R"(((?P<author>((([A-Z]\s\.|\-)\s)+([A-Z][a-z]+)))\s((\,|\&|and)\s)+)+(?P<lauthor>((([A-Z]\s\.|\-)\s)+([A-Z][a-z]+))))";

      pcre2_expr expr(this->get_key(), "__author_list__", authors_str);
      author_list.push_back(expr);
    }

    // __ival__ . Srivastava , R . - K . , Greff , K . & Schmidhuber , J . Highway networks . CoRR e - prints ( __year__ ) . arXiv : __fval__ .
    {
      std::string authors_str = R"((((?P<author>(([A-Z][a-z]+\s)(\,\s)(([A-Z]\s\.|\-)\s)+)))((\,|\&|and)\s))+(?P<lauthor>(([A-Z][a-z]+\s)(\,\s)(([A-Z]\s\.|\-)\s)+)))";

      pcre2_expr expr(this->get_key(), "__author_list__", authors_str);
      author_list.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__author__",
                      R"((?P<author>([A-Z][a-z]+)\s\,(\s[A-Z\-]\s\.)+)\s(\,|and|\&))");
      authors.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__author__",
                      R"((and|\&)\s(?P<author>([A-Z][a-z]+)\s\,\s([A-Z\-]\s\.)+)\s)");
      authors.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__author__",
                      R"((?P<author>((\s[A-Z\-]\s\.)+\s([A-Z][a-z]+)))\s(\,|and|\&)+)");
      authors.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__table__",
                      R"(^(?P<table>Table|TABLE|Tab|TAB)(\s*\.)?(\s*)(?P<index>(__(i|f)val__|[A-Z])))");
      caption_refs.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__table__",
                      R"(^(?P<table>Table|TABLE|Tab|TAB))");
      caption_refs.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__figure__",
                      R"(^(?P<figure>(Figure|FIGURE|Fig|FIG))(\s*\.)?(\s*)(?P<index>(__(i|f)val__|[A-Z])))");
      caption_refs.push_back(expr);
    }

    {
      pcre2_expr expr(this->get_key(), "__figure__",
                      R"(^(?P<figure>(Figure|FIGURE|Fig|FIG))(\s*\.)?)");
      caption_refs.push_back(expr);
    }
  }

  void nlp_model<CLS, SEMANTIC>::initialise_model()
  {
    //std::filesystem::path path = resources_dir / model_file;

    if(not fasttext_supervised_model::load(model_file))
      {
        LOG_S(FATAL) << "could not load semantic model ...";
      }
  }

  template<typename subject_type>
  bool nlp_model<CLS, SEMANTIC>::get(subject_type& subj, base_property& property)
  {
    for(auto& prop:subj.properties)
      {
        if(prop.get_type()==get_key())
          {
            property = prop;
            return true;
          }
      }

    return false;
  }

  bool nlp_model<CLS, SEMANTIC>::preprocess(const subject<TEXT>& subj, std::string& text)
  {
    //auto& wtokens = subj.get_word_tokens();

    //if(wtokens.size()==0)
    if(subj.get_num_wtokens()==0)
      {
        text.clear();
        return false;
      }

    {
      text = subj.get_text();

      std::vector<base_instance> insts={};
      for(auto inst:subj.instances)
        {
          if(inst.is_model(NUMVAL) or inst.is_model(LINK))
            {
              insts.push_back(inst);
            }
        }

      if(insts.size()>0)
        {
          std::sort(insts.begin(), insts.end());

          std::size_t l=0;
          std::stringstream ss;

          for(std::size_t i=0; i<insts.size(); i++)
            {
              auto crng = insts.at(i).get_char_range();

              if(l<crng.at(0))
                {
                  ss << text.substr(l, crng.at(0)-l);
                  ss << " __" << insts.at(i).get_subtype() << "__ ";

                  l = crng.at(1);
                }
            }

          if(l<text.size())
            {
              ss << text.substr(l, text.size()-l);
            }

          text = ss.str();
        }
      else
        {
          text = subj.get_text();

          if(text.size()>=256)
            {
              text = text.substr(0, 256);
            }
        }
    }

    //text = utils::to_lower(text);

    return true;
  }

  bool nlp_model<CLS, SEMANTIC>::preprocess(const subject<TABLE>& subj, std::string& text)
  {
    std::stringstream ss;
    //for(std::size_t i=0; i<subj.data.size(); i++)
    for(std::size_t i=0; i<subj.num_rows(); i++)
      {
        //auto& row = subj.data.at(i);
        for(std::size_t j=0; j<subj.num_cols(); j++)
          {
            ss << subj.at(i,j).get_text() << "; ";
          }
      }

    text = ss.str();

    return true;
  }

  bool nlp_model<CLS, SEMANTIC>::apply(subject<TEXT>& subj)
  {
    std::string text="", label="null";
    double conf=0.0;

    if(not preprocess(subj, text))
      {
        return false; //continue; // skip continue; // skip
      }

    if(not classify(text, label, conf))
      {
        return false; //continue; // skip
      }

    //LOG_S(INFO) << label << ", " << conf << ": " << text.substr(0, 64);
    
    subj.properties.emplace_back(subj.get_hash(), TEXT, subj.get_self_ref(), 
                                 get_name(), label, conf);
    subj.applied_models.insert(get_key());

    return true;
  }

  bool nlp_model<CLS, SEMANTIC>::apply(subject<TABLE>& subj)
  {
    return fasttext_supervised_model::classify(subj);
  }

  bool nlp_model<CLS, SEMANTIC>::apply(subject<DOCUMENT>& subj)
  {
    if(not satisfies_dependencies(subj))
      {
        return false;
      }

    for(uint64_t ind=0; ind<subj.texts.size(); ind++)
      {
        auto& para = subj.texts.at(ind);

        this->apply(*para);
      }

    /*
      if(abs_ind!=-1 and ind<abs_ind and label=="reference")
      {
      label = "meta-data";
      }
      else if(ref_ind!=-1 and ind<ref_ind and label=="reference")
      {
      label = "text";
      }
    */

    return update_applied_models(subj);
  }

}

#endif
