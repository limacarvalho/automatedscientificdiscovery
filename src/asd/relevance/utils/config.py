import pandas as pd
import os
import glob
import shutil
import uuid


dask_local_scheduler_port = 8787
rand_state = 0

# __file_path__ = os.path.dirname(__file__)

# print(__file_path__)

# main_dir = __file_path__ + '/home/wasif/python-asd/xai/app/ml/models/saved'
main_dir = '../'

# main_dir = __file_path__ + '/../ml/models/saved'
model_save_dir = main_dir + 'ml/models/saved'
# dataset_dir = "/home/wasif/python-asd/xai/app/test/data"
dataset_dir = main_dir +  "test/data"
project_name = '/covid_autolearn'
log_config_yaml = main_dir + 'utils/config.yaml'

# print(log_config_yaml)


### ray config params
working_dir = "/mnt/c/Users/rwmas/GitHub/xai/python-asd/src/asd/relevance"
pip = ["scikit-learn", 'optuna==2.3.0', 'xgboost', 'torch', 'captum', 'shap', 'datatable', 'hyperopt', 'ray[tune]', 
       'bayesian-optimization', 'xgboost_ray', 'lightgbm_ray', 'IPython', 'pyarrow', 'modin[ray]']
#addr = os.environ['RAYCLUSTER_AUTOSCALER_HEAD_SVC_PORT_10001_TCP_ADDR']
#port = os.environ['RAYCLUSTER_AUTOSCALER_HEAD_SVC_PORT_10001_TCP_PORT']

addr = '192.168.0.249'
port = '10001'



def create_study_name():
    str(uuid.uuid1(clock_seq=os.getpid()))


def clear_temp_folder(folder):
    files = glob.glob(folder+'/*')
    for f in files:
        os.remove(f)
        

def create_project_dirs(overwrite=False):
    project_dir = main_dir + project_name

    desired_permission = 0o755

    if not os.path.isdir(project_dir):
        os.makedirs(project_dir, desired_permission)


    base_dirs = ['/base/ann/slug','/base/ann/ensemble', '/base/xgboost/slug', '/base/xgboost/brisk', '/base/xgboost/ensemble']
    tmp_dirs = ['/tmp/ann/slug', '/tmp/xgboost/slug', '/tmp/xgboost/brisk']
    

    for dir in base_dirs:
        path = project_dir + dir
        
        if overwrite is True:
            if os.path.isdir(path):
                shutil.rmtree(path)
            

                
                
        if not os.path.isdir(path):
            os.makedirs(path, desired_permission)
        

    for dir in tmp_dirs:
        path = project_dir + dir   

        if overwrite is True:
            if os.path.isdir(path):
                shutil.rmtree(path)
        
        
        if not os.path.isdir(path):
            os.makedirs(path, desired_permission)

    if overwrite is True:
        print(f'Cleared the existing saved models')
        
    print(f'Directory structure for project {project_name} created successfully')


    
def create_dir_structure(model_name):

    # project_dir = model_save_dir + project_name + '/' + model_name
        
    project_dir = '/tmp' + project_name + '/' + model_name

    
    desired_permission = 0o755

    ### if exists then delete
    if os.path.isdir(project_dir):
        shutil.rmtree(project_dir)


    ### create main project directory
    if not os.path.isdir(project_dir):
        os.makedirs(project_dir, desired_permission)
    
    temp_path = project_dir + '/tmp'
    saved_path = project_dir + '/saved'

    os.makedirs(temp_path, desired_permission)
    os.makedirs(saved_path, desired_permission)

    
    # print(f'structure created under {project_dir}')
    return temp_path, saved_path