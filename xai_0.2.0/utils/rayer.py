
import os
import ray
import json
import time
from utils import config


@ray.remote
def delay(x):
    time.sleep(1)
    return x


excludes = [#'../test/data/wetransfer_asd_team_2022-10-25_1332.zip', 
            #'../test/data/20220319_covid_merge_processed.csv', 
            #'../test/data/20220319_covid_merge.csv',
            # '*.csv', 
            '*.log','*.zip',
            #'/home/wasif/python-asd/xai/test/data/transaction-predictions/train.csv',
            #'/home/wasif/python-asd/xai/test/data/transaction-predictions/test.csv',
            #'/home/wasif/python-asd/xai/test/data/transaction-predictions/train.csv',
            #'/home/wasif/python-asd/xai/test/data/transaction-predictions/test.csv',

           ]


def get_local_cluster(working_dir=None, num_cpus=None, num_gpus=None):
    
    if working_dir is None:
        working_dir = config.working_dir
    if num_cpus is None:
        num_cpus = config.num_cpus
    if num_gpus is None:
        num_gpus = config.num_gpus
        
    # runtime_env = {"working_dir": working_dir, "conda": {"dependencies": ["torch", "pip", {"pip": ["scipy", "scikit-learn"]}]}}
    
    runtime_env = {"working_dir": working_dir, 'excludes': excludes} #'pip': config.pip}
    
    # print(runtime_env)
    
    ray.shutdown()
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus, runtime_env=runtime_env)

    return ray
    
def get_global_cluster(working_dir=None, num_cpus=None, num_gpus=None, ip=None):
    
    if working_dir is None:
        working_dir = config.working_dir
    if num_cpus is None:
        num_cpus = config.num_cpus
    if num_gpus is None:
        num_gpus = config.num_gpus
        
    if ip is None:
        ray_cluster_addr = config.addr
        ray_cluster_port = config.port
    
        
        
    # Get the Ray Cluster endpoint address
    ray_cluster_uri = f"ray://{ray_cluster_addr}:{ray_cluster_port}"

    # runtime_env = {"working_dir": working_dir, "conda": {"dependencies": ["torch", "pip", {"pip": ["scipy", "scikit-learn"]}]}}
    
    # Start Ray.
    try:
        ray.shutdown()
        ray.init(address=ray_cluster_uri, runtime_env={"working_dir": working_dir, 'pip': config.pip, 'excludes': excludes})
        
        # Start tasks in parallel.
        ray.autoscaler.sdk.request_resources(num_cpus=num_cpus)
        
    except RuntimeError as error_msg:
        #get_ray_cluster_info = !ray status --address='raycluster-autoscaler-head-svc.dev.svc.cluster.local:6379'
        #ray_cluster_info = "\n".join(get_ray_cluster_info)
        print(error_msg)

    
        
def stop_global_cluster():
    # Scale Ray worker nodes/pods
    ray.autoscaler.sdk.request_resources(num_cpus=0)
    
