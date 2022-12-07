import logging
from logging import config

logfile = "logfile.log"

def empty_existing_logfile(filename: str=logfile):
    file = open(logfile,"r+")
    file.truncate(0)
    file.close()

log_config = {
    'version':1,
    'root':{
        'handlers' : ['stdout', 'file'],
        'level': 'INFO'
    },
    'handlers':{
        'stdout':{
            'formatter': 'main',
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        },
        'file':{
            'formatter': 'main',
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'filename': logfile,
            'mode': 'a+'
        },        
    },
    'formatters':{
        'main': {
            'format': '%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : Message : %(message)s',
            'datefmt': '%Y-%m-%dT%H:%M:%S%Z'
        }
    },
}

config.dictConfig(log_config)
logger = logging.getLogger(__name__)

