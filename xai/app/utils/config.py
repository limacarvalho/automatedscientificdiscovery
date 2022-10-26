import pandas as pd
import os
import glob
import shutil


dask_local_scheduler_port = 8787
rand_state = 0

main_dir = '/home/wasif/python-asd/xai/app/ml/models/saved'
dataset_dir = "/home/wasif/python-asd/xai/app/test/data"
project_name = '/covid_autolearn'



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



