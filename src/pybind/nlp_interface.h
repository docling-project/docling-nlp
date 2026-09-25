//-*-C++-*-

#ifndef PYBIND_ANDROMEDA_NLP_INTERFACE_H
#define PYBIND_ANDROMEDA_NLP_INTERFACE_H

namespace andromeda_py
{
  class nlp_model: public base_log,
		   public base_resources
  {
  public:
    
    nlp_model();
    nlp_model(std::string loglevel, bool text_ordering, bool normalise_chars, bool normalise_text);
    
    ~nlp_model();

    bool initialise(const nlohmann::json config_);
    bool initialise_models(const std::string model_names);
    
    nlohmann::json get_apply_configs();
    nlohmann::json get_train_configs();

    nlohmann::json prepare_data_for_train(nlohmann::json& config);
    nlohmann::json apply(nlohmann::json& config);
    nlohmann::json train(nlohmann::json& config);
    nlohmann::json evaluate(nlohmann::json& config);
    
    nlohmann::json apply_on_text(std::string& text);

    bool apply_on_text(ds_text& subj);
    bool apply_on_table(ds_table& subj);
    bool apply_on_doc(ds_document& subj);
    
  private:

    void apply_paragraphs(std::shared_ptr<andromeda::producer<andromeda::TEXT> > producer,
			  nlohmann::json& results);

    void apply_docs(std::shared_ptr<andromeda::producer<andromeda::DOCUMENT> > producer,
		    nlohmann::json& results);

    template<typename subj_type>
    bool apply_on_subj(std::shared_ptr<subj_type> ptr);
    
  private:
    
    nlohmann::json config;

    bool order_text;
    
    std::vector<std::shared_ptr<andromeda::base_nlp_model> > models;

    std::shared_ptr<andromeda::utils::char_normaliser> char_normaliser;
    std::shared_ptr<andromeda::utils::text_normaliser> text_normaliser;    
  };
  
  nlp_model::nlp_model():
    base_log::base_log(),
    base_resources::base_resources(),

    config(nlohmann::json::object({})),
    
    order_text(false),
    models({}),

    char_normaliser(andromeda::text_element::create_char_normaliser(false)),
    text_normaliser(andromeda::text_element::create_text_normaliser(false))
  {}

  nlp_model::nlp_model(std::string loglevel, bool text_ordering, bool normalise_chars, bool normalise_text):
    base_log::base_log(loglevel),
    base_resources::base_resources(),
    
    config(nlohmann::json::object({})),
    
    order_text(text_ordering),
    models({}),

    char_normaliser(andromeda::text_element::create_char_normaliser(normalise_chars)),
    text_normaliser(andromeda::text_element::create_text_normaliser(normalise_text))
  {
    config["order-text"] = order_text;
  }
  
  nlp_model::~nlp_model()
  {}
  
  bool nlp_model::initialise(const nlohmann::json config_)
  {
    std::string mode = config_["mode"].get<std::string>();
    
    if(mode=="apply")
      {
	config = config_;
	
	order_text = true;
	order_text = config.value("order-text", order_text);
	
	std::string models_expr = "semantic;term"; // default models
	models_expr = config.value("models", models_expr);

	return andromeda::to_models(models_expr, this->models, false);
      }
    else
      {
	LOG_S(WARNING) << "could not initialise nlp_model";	
	return false;
      }
  }

  bool nlp_model::initialise_models(const std::string model_names)
  {
    config.clear();
    order_text = true;
    
    return andromeda::to_models(model_names, this->models, true);    
  }
  
  nlohmann::json nlp_model::get_apply_configs()
  {
    nlohmann::json configs = nlohmann::json::array({});

    {
      nlohmann::json config;
      {
	config["mode"] = "apply";

	config["order-text"] = true;
	
	config["models"] = "reference;conn;term;verb";
	config["interactive"] = true;
	
	config["subject"] = "<prompt;paragraph;paragraph:jsonl:text;pdfdoc>";
	config["subject-filters"] = std::set<std::string>({});
	
	config["input"] = "";
	config["output"] = "";
      }

      configs.push_back(config);
    }

    return configs;
  }

  nlohmann::json nlp_model::get_train_configs()
  {
    nlohmann::json configs = nlohmann::json::array({});
    
    std::shared_ptr<andromeda::base_nlp_model> model;
    for(auto name:andromeda::MODEL_NAMES)
      {
	model = andromeda::to_trainable_model(name);
	//LOG_S(INFO) << "model: " << model;
	
	if(model!=NULL and model->is_trainable())
	  {
	    //LOG_S(INFO) << andromeda::to_string(name) << ": trainable";
	    
	    nlohmann::json config = model->create_train_config();
	    configs.push_back(config);
	  }
	else
	  {
	    //LOG_S(WARNING) << andromeda::to_string(name) << ": not trainable";
	  }
      }

    return configs;    
  }
  
  nlohmann::json nlp_model::apply(nlohmann::json& config)
  {
    nlohmann::json result = nlohmann::json::array({});

    std::vector<std::shared_ptr<andromeda::base_nlp_model> > nlp_models={};
    andromeda::to_models(config, nlp_models, false);
    
    std::vector<std::shared_ptr<andromeda::base_producer> > base_producers={};
    andromeda::to_producers(config, nlp_models, base_producers);

    nlohmann::json results = nlohmann::json::array({});

    for(auto& base_producer:base_producers)
      {
	switch(base_producer->get_subject_name())
	  {
	  case andromeda::TEXT:
	    {
	      typedef andromeda::producer<andromeda::TEXT> producer_type;
	      auto producer = std::dynamic_pointer_cast<producer_type>(base_producer);

	      apply_paragraphs(producer, results);
	    }
	    break;

	  case andromeda::DOCUMENT:
	    {
	      typedef andromeda::producer<andromeda::DOCUMENT> producer_type;
	      auto producer = std::dynamic_pointer_cast<producer_type>(base_producer);

	      apply_docs(producer, results);
	    }
	    break;	    

	  default:
	    {
	      LOG_S(WARNING) << "nlp-model (" << __FILE__ << ":" << __LINE__ << ") "
			     << "does not support producer-type: "
			     << to_string(base_producer->get_subject_name());
	    }

	  }
      }
    
    return results;
  }

  void nlp_model::apply_paragraphs(std::shared_ptr<andromeda::producer<andromeda::TEXT> > producer,
				   nlohmann::json& results)
  {
    andromeda::subject<andromeda::TEXT> subj;

    std::size_t count=0;
    
    bool read=true;
    while(read)      
      {
	{
	  //std::scoped_lock lock(read_mtx);
	  read = producer->read(subj, count);
	}

	if(not read)
	  {
	    continue;
	  }
	
	producer->apply(subj);
	
	//nlohmann::json out = subject.to_json();
	subj.show(true, true,
		  false, true,
		  true, true, true);
	
	std::string line="y";
	std::getline(std::cin, line);
	
	if(line=="quit" or line=="q")
	  {
	    break;
	  }	    	
      }
  }
  
  void nlp_model::apply_docs(std::shared_ptr<andromeda::producer<andromeda::DOCUMENT> > producer,
			     nlohmann::json& results)
  {
    andromeda::subject<andromeda::DOCUMENT> subj;

    std::size_t count=0;
    
    bool read=true;
    while(read)
      {
	{
	  //std::scoped_lock lock(read_mtx);
	  read = producer->read(subj, count);
	}

	if(not read)
	  {
	    continue;
	  }
	
	producer->apply(subj);
      }
  }  

  nlohmann::json nlp_model::prepare_data_for_train(nlohmann::json& config)
  {
    std::string model_name = "null";
    model_name = config.value("model", model_name);

    std::vector<std::shared_ptr<andromeda::base_nlp_model> > dep_models={};
    // The generic CUSTOM_FST training config has no model file yet. It also
    // has no dependencies, so attempting to initialise custom_fst(:) is both
    // unnecessary and produces a misleading warning.
    if(model_name!="custom_fst(:)")
      {
	andromeda::to_models(model_name, dep_models, true);

	if(dep_models.size()>0)
	  {
	    dep_models.pop_back();
	  }
      }
    
    andromeda::model_name name = andromeda::to_modelname(model_name);
    std::shared_ptr<andromeda::base_nlp_model> model = andromeda::to_trainable_model(name);

    bool success=false;
    std::stringstream ss;    

    if(model==NULL or (not (model->is_trainable())))
      {
	ss << "model '" << model_name << "' can not be trained";
      }
    else
      {
	success = model->prepare_data_for_train(config, dep_models);
	
	if(success)
	  {
	    ss << "data for model '" << model->get_key() << "' is prepared!";
	  }
	else
	  {
	    ss << "data for model '" << model->get_key() << "' is not prepared!";
	  }	
      }

    nlohmann::json result = config;
    {
      nlohmann::json& training = result["model-training"];
      {
	training["success"] = success;
	training["message"] = ss.str();
      }
    }
    
    return result;    
  }
  
  nlohmann::json nlp_model::train(nlohmann::json& config)
  {
    std::string model_name = "null";
    model_name = config.value("model", model_name);
    
    andromeda::model_name name = andromeda::to_modelname(model_name);
    std::shared_ptr<andromeda::base_nlp_model> model = andromeda::to_trainable_model(name);

    bool success=false;
    std::stringstream ss;    

    if(model==NULL or (not (model->is_trainable())))
      {
	ss << "model '" << model_name << "' can not be trained";
      }
    else
      {
	success = model->train(config);
	
	if(success)
	  {
	    ss << "model '" << model->get_key() << "' is trained!";
	  }
	else
	  {
	    ss << "model '" << model->get_key() << "' is not trained!";
	  }	
      }

    nlohmann::json result = config;
    {
      nlohmann::json& training = result["model-training"];
      {
	training["success"] = success;
	training["message"] = ss.str();
      }
    }
    
    return result;
  }
  
  nlohmann::json nlp_model::evaluate(nlohmann::json& config)
  {
    LOG_S(INFO) << __FUNCTION__;
    
    std::string model_name = "null";
    model_name = config.value("model", model_name);
    
    std::vector<std::shared_ptr<andromeda::base_nlp_model> > dep_models={};
    
    andromeda::model_name name = andromeda::to_modelname(model_name);
    std::shared_ptr<andromeda::base_nlp_model> model = andromeda::to_trainable_model(name);

    if(model!=NULL)
      {
	for(auto dep_name:model->get_dependencies())
	  {
	    andromeda::to_models(dep_name, "", dep_models, true);
	  }
      }

    bool success=false;
    std::stringstream ss;    

    if(model==NULL or (not (model->is_trainable())))
      {
	ss << "model '" << model_name << "' can not be trained";
      }
    else
      {
	success = model->evaluate_model(config, dep_models);
	
	if(success)
	  {
	    ss << "model '" << model->get_key() << "' is evaluated";
	  }
	else
	  {
	    ss << "model '" << model->get_key() << "' is not evaluated!";
	  }	
      }

    nlohmann::json result = config;
    {
      nlohmann::json& training = result["model-training"];
      {
	training["success"] = success;
	training["message"] = ss.str();
      }
    }
    
    return result;    
  }
  
  nlohmann::json nlp_model::apply_on_text(std::string& text)
  {
    andromeda::subject<andromeda::TEXT> paragraph;
    bool valid = paragraph.set(text, char_normaliser, text_normaliser);

    bool success=false;
    std::stringstream ss;    
    
    if(valid)
      {
	for(auto& model:models)
	  {
	    model->apply(paragraph);
	  }

	success=true;
	ss << "success";
      }
    else
      {
	success=false;
	ss << "text is not UTF8 compliant";
      }

    //LOG_S(INFO) << "starting to sort";
    
    paragraph.sort();

    std::set<std::string> subj_filters = {};
    if(config.is_object())
      {
	subj_filters = config.value("subject-filters", subj_filters);
      }

    //LOG_S(INFO) << "to json ...";
    
    nlohmann::json result = paragraph.to_json(subj_filters);
    {
      nlohmann::json& application = result["model-application"];
      {
	application["success"] = success;
	application["message"] = ss.str();
      }
    }
    
    return result;
  }

  template<typename subj_type>
  bool nlp_model::apply_on_subj(std::shared_ptr<subj_type> ptr)
  {    
    if(ptr==NULL)
      {
	return false;
      }

    //LOG_S(INFO) << "nlp_model::apply_on_subj(std::shared_ptr<subj_type> ptr) set_tokens ...";
    
    bool valid = ptr->set_tokens(char_normaliser, text_normaliser);

    //LOG_S(INFO) << "valid: " << valid;
    
    if(valid)
      {
	for(auto& model:models)
	  {
	    model->apply(*ptr);
	  }
	ptr->sort();

	return true;
      }
    
    return false;
  }
  
  bool nlp_model::apply_on_text(ds_text& subj)
  {
    return apply_on_subj(subj.get_ptr());
  }
  
  bool nlp_model::apply_on_table(ds_table& subj)
  {
    return apply_on_subj(subj.get_ptr());
  }
  
  bool nlp_model::apply_on_doc(ds_document& subj)
  {
    bool success = apply_on_subj(subj.get_ptr());

    if(success)
      {
	success = subj.get_ptr()->finalise();
      }

    return success;
  }
  

}

#endif
