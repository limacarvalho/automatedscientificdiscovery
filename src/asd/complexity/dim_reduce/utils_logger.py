import logging
from logging import config

logfile = 'logfile.log'

def empty_existing_logfile(filename: str=logfile):
    '''
    removes content (previous logs) from logfile
    :param filename: path to filename
    '''
    file = open(filename,"r+")
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
            # 'format': '%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : Message : %(message)s',
            # 'datefmt': '%Y-%m-%dT%H:%M:%S%Z'
            'format': '%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(lineno)d : %(message)s',
            'datefmt': '%Y%m:%H:%M:%S'
        }
    },
}

config.dictConfig(log_config)
logger = logging.getLogger(__name__)