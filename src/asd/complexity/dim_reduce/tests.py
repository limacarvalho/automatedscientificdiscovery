import numpy as np
from dimension_tools.dimension_suite.dim_reduce.steps.main import intrinsic_dimension
from helper_data.global_vars import *
from dimension_tools.dimension_suite.dim_reduce.helper_data.helper_data import Preprocess_data
import skdim
import time





###
def danco_pca(dim_is, data_id, data_high, lpca=True):
    # lpca
    if lpca == True:
        start = time.time()
        data_scale, status_preprocess = Preprocess_data().preprocess_scaling(data_high)
        ncols = data_scale.shape[1]
        lpca = skdim.id.lPCA().fit_pw(data_scale, n_neighbors=ncols, n_jobs = 1)
        dim_lpca = round(np.mean(lpca.dimension_pw_),2)
        time_lpca = round(time.time() - start, 2)
    else:
        dim_lpca, time_lpca = 'NaN', 'NaN'

    # asd
    start = time.time()
    dim_asd, df_summary = intrinsic_dimension(data_high=data_high,
                                          data_id=data_id,
                                          columns=None,
                                          cutoff_loss=0.99,
                                          functions=['all_functions']
                                          )
    time_asd = round(time.time() - start, 2)
    #
    print(df_summary)
    path = globalvar_path_dir_results
    df_summary.to_csv(path + data_id + '_results_dimred.csv', index=False)

    print('!!!!! RESULT:', dim_is,
          'lpca: ',   dim_lpca,
          'asd_dim_reduce:', dim_asd,
          'time_lpca', time_lpca,
          'time_asd', time_asd)


###
def manifold_generated_data():
    # dim is the number of columns and d the intrinsic dimension
    start = time.time()
    data_high = skdim.datasets.BenchmarkManifolds().generate(name='M12_Norm',
                                                             n=10000,
                                                             dim=300,
                                                             d=300,
                                                             noise=0.0)
    #
    # data_high = np.concatenate((data_high, data_high, data_high), axis=1)
    print(data_high.shape, 'data generation timit: ', round(time.time() - start, 2))

    # estimate dim with several methods
    rows = data_high.shape[0]
    cols = data_high.shape[1]
    count = 0
    dims = [2, 25, 50, 75, 98]
    noise = 1
    # dims = [50]
    str_shape = '_shape'+str(rows)+'x'+str(cols)
    # danco_pca(dim_is=100, data_id='manifold_dim0_noise'+str(noise)+str_shape, data_high=data_high)
    for i in range(1, cols, 1):
        count = count + 1
        rand  = np.random.randint(-999999, 999999)/1000000
        data_high[::, i] = data_high[::, 0] * rand * noise
        ## every one
        if count in dims:
            dim = cols - i
            data_id = 'manifold_dim_'+str(dim)+'_noise'+str(noise)+str_shape
            danco_pca(dim, data_id, data_high, lpca=False)

# manifold_generated_data()


'''
step 3:
- factor = 1 or higher steps in case of higher dimensions.
- take only the top functions in case of many functions coosen.

error step3:
there was one case (manifold 1000x100 dim 50) where the dim went 46, 47 and up to ncols=100.
at 47 there was no result >= cutoff

'''

'''
Try: This block will test the excepted error to occur
Except:  Here you can handle the error
Else: If there is no exception then this block will be executed
Finally: Finally block always gets executed either exception is generated or not
'''





## function for reducing data
# data_size = nrows * ncols
# if data_size <= lowest_limit:
#     data_reduced = data
# else:
#     # size = nrows * ncols
#     # size < lowest_limit: size_reduced = size
#     # size > lowest_limit: size_reduced = size / lowest_limit
#     # at size of 1.000.0000 = 5 percent of rows
#     nrows_reduced = min(int(lowest_limit / (data_size) * nrows), nrows)
#     idx_rows = random.sample(range(0, nrows-1), nrows_reduced)
#     data_reduced = data[idx_rows]