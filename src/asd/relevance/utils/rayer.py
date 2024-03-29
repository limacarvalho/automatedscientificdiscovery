import time
import logging

import ray
from relevance.utils import config
from utils_logger import LoggerSetup

# Initialize logging object (Singleton class) if not already
LoggerSetup()


@ray.remote
def delay(x):
    time.sleep(1)
    return x


excludes = [
    # '*.csv',
    "*.log",
    "*.zip",
    "*.pickle",
]


def get_local_cluster(working_dir=None):
    if working_dir is None:
        working_dir = config.working_dir

    # runtime_env = {"working_dir": working_dir, "conda": {"dependencies": ["torch", "pip", {"pip": ["scipy", "scikit-learn"]}]}}

    runtime_env = {"working_dir": working_dir, "excludes": excludes}  #'pip': config.pip}

    # print(runtime_env)

    ray.shutdown()
    ray.init(runtime_env=runtime_env)

    return ray


def get_global_cluster(working_dir=None, ip=None):
    if working_dir is None:
        working_dir = config.working_dir

    if ip is None:
        ray_cluster_addr = config.addr
        ray_cluster_port = config.port

    # Get the Ray Cluster endpoint address
    ray_cluster_uri = f"ray://{ray_cluster_addr}:{ray_cluster_port}"

    # runtime_env = {"working_dir": working_dir, "conda": {"dependencies": ["torch", "pip", {"pip": ["scipy", "scikit-learn"]}]}}

    # Start Ray.
    try:
        ray.shutdown()
        ray.init(
            address=ray_cluster_uri,
            runtime_env={"working_dir": working_dir, "pip": config.pip, "excludes": excludes},
            # logging_level = logging.DEBUG
        )

        # Start tasks in parallel.
        ray.autoscaler.sdk.request_resources()

    except RuntimeError as error_msg:
        # get_ray_cluster_info = !ray status --address='raycluster-autoscaler-head-svc.dev.svc.cluster.local:6379'
        # ray_cluster_info = "\n".join(get_ray_cluster_info)
        print(error_msg)


def stop_global_cluster():
    # Scale Ray worker nodes/pods
    ray.autoscaler.sdk.request_resources(num_cpus=0)
