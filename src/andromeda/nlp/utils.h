//-*-C++-*-

#ifndef ANDROMEDA_MODELS_UTILS_H_
#define ANDROMEDA_MODELS_UTILS_H_

#include <andromeda/utils/string/utils.h>

namespace andromeda
{

  std::string from_models(std::vector<std::shared_ptr<base_nlp_model> >& models)
  {
    std::stringstream ss;
    for(std::size_t l=0; l<models.size(); l++)
      {
	if(l+1==models.size())
	  {
	    ss << models.at(l)->get_key();
	  }
	else
	  {
	    ss << models.at(l)->get_key() << ";";
	  }
      }

    return ss.str();
  }
  
  std::shared_ptr<base_nlp_model> to_trainable_model(model_name name)
  {
    std::shared_ptr<base_nlp_model> model=NULL;
    
    switch(name)
      {
      case SPM:
	{
	  typedef nlp_model<TOK, SPM> model_type;
	  model = std::make_shared<model_type>();
	}
	break;
	
      case SEMANTIC:
	{
	  typedef nlp_model<CLS, SEMANTIC> model_type;
	  model = std::make_shared<model_type>();
	}
	break;

      case REFERENCE:
	{
	  typedef nlp_model<ENT, REFERENCE> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;

      case CUSTOM_CRF:
	{
	  typedef nlp_model<ENT, CUSTOM_CRF> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;

      case CUSTOM_SPM:
	{
	  typedef nlp_model<TOK, CUSTOM_SPM> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;

      case CUSTOM_FST:
	{
	  typedef nlp_model<CLS, CUSTOM_FST> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;
	
      default:
	{
	  model = NULL;
	}
      }
    
    return model;
  }
  
  bool to_models(model_name name, std::string desc,
		 std::vector<std::shared_ptr<base_nlp_model> >& models, bool verbose)
  {
    if(verbose)
      {
	LOG_S(INFO) << "initialising model: " << to_string(name);
      }
    
    std::set<model_name> curr_names={};
    for(auto& cmodel:models)
      {
	curr_names.insert(cmodel->get_name());
      }

    // already satisfied ...
    if(curr_names.count(name)) 
      {
	return true;
      }
    
    std::shared_ptr<base_nlp_model> model=NULL;
    
    switch(name)
      {
	// TOK
      case SPM:
	{
	  typedef nlp_model<TOK, SPM> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;
	
	// POS
      case LAPOS:
	{
	  typedef nlp_model<POS, LAPOS> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;
	
	// CLS
      case LANGUAGE:
	{
	  typedef nlp_model<CLS, LANGUAGE> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;

      case SEMANTIC:
	{
	  typedef nlp_model<CLS, SEMANTIC> model_type;
	  const std::string prefix = "semantic(";
	  if(desc.starts_with(prefix) and desc.ends_with(")"))
	    {
	      model = std::make_shared<model_type>(
	        std::filesystem::path(desc.substr(prefix.size(),
	                                          desc.size()-prefix.size()-1)));
	    }
	  else
	    {
	      model = std::make_shared<model_type>();
	    }
	}
	break;	
	
	// ENT
      case NAME:
	{
	  typedef nlp_model<ENT, NAME> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;

      case LINK:
	{
	  typedef nlp_model<ENT, LINK> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;

      case CITE:
	{
	  typedef nlp_model<ENT, CITE> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;	

      case QUOTE:
	{
	  typedef nlp_model<ENT, QUOTE> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;		
	
      case NUMVAL:
	{
	  typedef nlp_model<ENT, NUMVAL> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;

      case GEOLOC:
	{
	  typedef nlp_model<ENT, GEOLOC> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;
	
      case PARENTHESIS:
	{
	  typedef nlp_model<ENT, PARENTHESIS> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;

      case EXPRESSION:
	{
	  typedef nlp_model<ENT, EXPRESSION> model_type;
	  model = std::make_shared<model_type>();	  	  
	}
	break;	

      case SENTENCE:
	{
	  typedef nlp_model<ENT, SENTENCE> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;

      case REFERENCE:
	{
	  typedef nlp_model<ENT, REFERENCE> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;	

      case CUSTOM_SPM:
	{
	  typedef nlp_model<TOK, CUSTOM_SPM> model_type;
	  model = std::make_shared<model_type>(desc);	  	  	  
	}
	break;
	
      case CUSTOM_CRF:
	{
	  typedef nlp_model<ENT, CUSTOM_CRF> model_type;
	  model = std::make_shared<model_type>(desc);	  	  	  
	}
	break;

      case CUSTOM_FST:
	{
	  typedef nlp_model<CLS, CUSTOM_FST> model_type;
	  model = std::make_shared<model_type>(desc);	  	  	  
	}
	break;

      case MATERIAL: 
	{
	  typedef nlp_model<ENT, MATERIAL> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;
	
      case CONN:
	{
	  typedef nlp_model<ENT, CONN> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;
	
      case TERM:
	{
	  typedef nlp_model<ENT, TERM> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;

      case VERB:
	{
	  typedef nlp_model<ENT, VERB> model_type;
	  model = std::make_shared<model_type>();	  	  	  
	}
	break;	

      case ABBREVIATION:
	{
	  typedef nlp_model<REL, ABBREVIATION> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;

      case VAU:
	{
	  typedef nlp_model<REL, VAU> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;

      case METADATA:
	{
	  typedef nlp_model<REC, METADATA> model_type;
	  model = std::make_shared<model_type>();	  
	}
	break;		
	
      default:
	{
	  LOG_S(ERROR) << "no implementation for model: " << to_string(name);
	  return false;
	}
      }

    std::set<model_name> deps_names = model->get_dependencies();
    for(model_name dep_name:deps_names)
      {
	if(curr_names.count(dep_name)==0)
	  {
	    if(not to_models(dep_name, "", models, verbose))
	      {
		LOG_S(FATAL) << "can not satisfy dependencies for model: "
			     << to_string(name) << " for dependency " << to_string(dep_name);
	      }
	  }
      }

    models.push_back(model);
    return true;
  }
  
  bool to_model(std::string name, std::vector<std::shared_ptr<base_nlp_model> >& models,
		bool verbose)
  {
    model_name name_ = to_modelname(name);
    return to_models(name_, name, models, verbose);
  }
  
  bool to_models(std::string expr,
		 std::vector<std::shared_ptr<base_nlp_model> >& models,
		 bool verbose)
  {
    if(verbose)
      {
	LOG_S(INFO) << "initialising models-expression: " << expr;
      }
    
    std::vector<std::string> model_names;

    if(expr.find(',')!=std::string::npos)
      {
	model_names = utils::split(expr, ',');
      }
    else if(expr.find(';')!=std::string::npos)
      {
	model_names = utils::split(expr, ';');
      }
    else if(expr.size()==0)
      {
	model_names = {};
      }
    else
      {
	model_names = {expr};
      }

    for(std::string name:model_names)
      {
	if(not to_model(name, models, verbose))
	  {
	    return false;
	  }
      }

    if(verbose)
      {    
	std::size_t cnt=0;
	for(auto& model:models)
	  {
	    LOG_S(INFO) << " [" << cnt++ << "] " << to_string(model->get_name());
	  } 
      }

    return true;
  }

  bool to_models(nlohmann::json config,
		 std::vector<std::shared_ptr<base_nlp_model> >& models,
		 bool verbose)
  {
    if(config.count("models"))
      {
	std::string models_expr = "";
	models_expr = config.value("models", models_expr);

	return to_models(models_expr, models, verbose);
      }

    models.clear();
    return false;
  }
  
}

#endif
