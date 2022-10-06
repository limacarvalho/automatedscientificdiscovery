from distributed import Client, LocalCluster
from dask import distributed
from utils import config
from dask.distributed import Client


def get_dask_client():

    client = distributed.client._get_global_client()

    if client is None:
        cluster = LocalCluster(scheduler_port=8786)
        # cluster = LocalCluster(scheduler_port=config.dask_local_scheduler_port)
        cl = Client(cluster)
        distributed.client._set_global_client(cl)
        return cl
    else:
        return client
    


def get_dask_client_cluster():            
    pass