import jsonschema
import yaml
from datetime import datetime
from dateutil.relativedelta import relativedelta

def check(file_name): 

    """
    Pass in a config file and check for the presence of specific keys
    within that config file and that those keys have a specific data 
    type.
    
    args:
    file_name (str) : name of the config file

    returns:
    None
    """

    # Open config file
    with open(file_name, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

    # Parse out training and inferencing portions of the file
    train_config = config["train"]
    inf_config = config["inference"]

    # Instantiate the training schema we expect
    train_schema = {
                    "type": "object",
                    "properties": {
                                "sampling": {"type":"string",
                                            "enum": ["down","up"]},
                                "split": {"type": "object",
                                          "properties":{"mode": {"type":"string",
                                                                "enum": ["last_n_months","randomized"]}},                  
                                          "allOf": [{"if": {"properties":{"mode":{"const":"last_n_months"}}},
                                                     "then":{"required": ["months"],
                                                             "properties":{"months": {"type":"integer","minimum":1}}}},
                                                    {"if":{"properties":{"mode":{"const":"randomized"}}},
                                                     "then":{"required":["perc_test"],
                                                             "properties":{"perc_test": {"type":"number",
                                                                                         "exclusiveMinimum":0.0,
                                                                                         "exclusiveMaximum":1.0}}}}],
                                          "required": ["mode"],                                                
                                        },
                                "start_month": {"type": "string","pattern":"^[0-9]{4}-[0-9]{2}$"},
                                "max_month": {"type": "string","pattern":"^[0-9]{4}-[0-9]{2}$"},
                                "profiling": {"type": "boolean"},
                                "features": {"type": "array"},
                                "rename_cols": {"type": "object"},
                                "drop": {"type": "array"},
                                "dummy_cols": {"type": "array"},
                                "target": {"type": "string"},
                                "hyperparameters": {"type": "object",
                                                    "required": ["optimize", "num_tuning_rounds", "eta", "max_depth", 
                                                                "subsample", "gamma", "min_child_weight", "reg_alpha"],
                                                    "properties": {"optimize": {"type": "string",
                                                                                "enum":["auc","f1","accuracy"]},
                                                                    "num_tuning_rounds": {"type": "integer",
                                                                                          "minimum":1},
                                                                    "eta": {"type": "object",
                                                                            "required": ["min"],
                                                                            "properties": {
                                                                                        "min": {"type": "integer",
                                                                                                "maximum":0}}},
                                                                    "max_depth": {"type": "object",
                                                                                  "required": ["min", "max"],
                                                                                  "properties": {
                                                                                            "min": {"type": "integer",
                                                                                                        "minimum":1},
                                                                                            "max": {"type": "integer",
                                                                                                    "minimum":1}}},
                                                                    "subsample": {"type": "object",
                                                                                  "required": ["min", "max"],
                                                                                    "properties": {
                                                                                            "min": {"type": "number",
                                                                                                    "exclusiveMinimum":0,
                                                                                                    "maximum":1},
                                                                                            "max": {"type": "number",
                                                                                                    "exclusiveMinimum":0,
                                                                                                    "maximum":1}}},
                                                                    "gamma": {"type": "object",
                                                                              "required": ["min", "max"],
                                                                              "properties": {
                                                                                            "min": {"type": "integer",
                                                                                                    "minimum":0},
                                                                                            "max": {"type": "integer",
                                                                                                    "minimum":0}}},
                                                                    "min_child_weight": {"type": "object",
                                                                                         "required": ["min", "max"],
                                                                                         "properties": {
                                                                                            "min": {"type": "integer",
                                                                                                    "minimum":0},
                                                                                            "max": {"type": "integer",
                                                                                                    "minimum":0}}},
                                                                    "reg_alpha": {"type": "object",
                                                                                  "required": ["min", "max"],
                                                                                  "properties": {
                                                                                            "min": {"type": "integer",
                                                                                                    "minimum":0},
                                                                                            "max": {"type": "integer",
                                                                                                    "minimum":0}
                                                                                                }}
                                                                }
                                                    }
                                },
                    "required": ["sampling", "split", "start_month", "max_month", "profiling", "features", 
                                "rename_cols", "drop", "dummy_cols", "target", "hyperparameters"]
                    
                    }

    # Instantiate the inferencing schema we expect
    inf_schema = {
                    "type": "object",
                    "properties": {
                                "min_history_months": {"type":"integer","minimum":0,"maximum":6},
                                "num_month_forecast": {"type": "integer","minimum":1,"maximum":6},
                                "profiling": {"type": "boolean"}
                    },
                    "required": ["min_history_months", "num_month_forecast", "profiling"]
                }

    # Validate that both schemas are correctly formatted
    try:
        jsonschema.validate(train_config, train_schema)
        earliest_forecast_month = (datetime.today() + relativedelta(months=-config['inference']['min_history_months'])).strftime('%Y-%m')
        if config['train']['max_month']>=earliest_forecast_month:
            raise Exception("max month of training set overlaps with forecast period, please use earlier month")
        print("Train schema is compatible")
    except Exception as e:
        raise e
    
    try:
        jsonschema.validate(inf_config, inf_schema)
        print("Inference schema is compatible")
    except Exception as e:
        raise e

    return config

if __name__=="__main__":
    check("../config.yaml")