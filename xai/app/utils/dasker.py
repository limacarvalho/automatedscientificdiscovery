from distributed import Client, LocalCluster
from dask import distributed
from utils import config
from dask.distributed import Client
from dask_gateway import Gateway


def get_local_client():
    client = distributed.client._get_global_client()

    if client is None:
        cluster = LocalCluster(scheduler_port=8786)
        # cluster = LocalCluster(scheduler_port=config.dask_local_scheduler_port)
        cl = Client(cluster)
        distributed.client._set_global_client(cl)
        return cl
    else:
        return client
    
    

def get_global_client():                
    client = None
    gateway = Gateway()
    clusters = gateway.list_clusters()

    if len(clusters) > 0:
        cluster_name = clusters[0].name
        print(f'connected to cluster {cluster_name}')
        cluster = gateway.connect(cluster_name=cluster_name)
        # client = cluster.get_client()
        client = Client(cluster)
        distributed.client._set_global_client(client)
        return client
    else:
        print('No cluster running')
        return None