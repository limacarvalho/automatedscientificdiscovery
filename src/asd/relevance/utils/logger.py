import logging
import logging.config
import yaml
import warnings

from asd.relevance.utils import config
import os



# warnings.filterwarnings("ignore", category=ExperimentalWarning, module="optuna.multi_objective")
# warnings.filterwarnings('ignore', '.*multi_objective.*')

warnings.filterwarnings("ignore", module="distributed.utils_perf")
warnings.filterwarnings("ignore", module="LightGBM")
logging.getLogger('shap').setLevel(logging.WARNING) # turns off the "shap INFO" logs
warnings.filterwarnings("ignore", module="shap")


#with open(config.log_config_yaml, 'r') as f:
with open('/home/wasif/python-asd/xai/auto-learn/utils/config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)



