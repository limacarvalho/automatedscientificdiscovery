from utils import dasker, config as util_config
from ml import KnockoffFilterSim, knockoffsettings
# from helper import consolidate_rejection_results
from . import helper
# from ml.models.brisk_models import BriskModels



def fit_models():

    client = dasker.get_dask_client()
    print(f'Local dask client created: localhost:{util_config.dask_local_scheduler_port}')

    # briskmodels = BriskModels(client)
    # briskmodels.fit_models()



def run_knockoffs(param, itr):

    client = dasker.get_dask_client()
    print(f'Local dask client created: localhost:{util_config.dask_local_scheduler_port}')
    ks = KnockoffFilterSim(client)

#    param['ksampler'] = 'metro'
#    print(f'running knockoff filter with iteration { itr} for  generic metropolized sampler.')
    ks.sim_knockoffs(param, itr)
    rejection_by_vote = ks.rejections


    resultset_dic = {
        'fdr': param['fdr'], 
        'itr': itr,
        'rejections': rejection_by_vote.tolist()
    }

    return resultset_dic
