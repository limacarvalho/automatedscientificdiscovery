
## erased after extensive testing end 2022
# 'dictlearn': python.dictionary_learning(n=ndim)
# 'r_asi': rdim.funR_asi(), # takes too long, low loss
# 'r_elpp2': rdim.funR_elpp2(), # takes too long, low loss
# 'r_ldakm': rdim.funR_ldakm(), # takes too long, low loss
# 'r_lpp': rdim.funR_lpp(), # takes too long, low loss
# 'r_pflpp':rdim.funR_pflpp(), # takes too long, low loss
# 'r_sdlpp':rdim.funR_sdlpp(), # takes too long, low loss
# 'r_idmap':rdim.funR_idmap(), # takes too long, low loss
# 'r_keca':rdim.funR_keca(), # takes too long, low loss
# 'r_lapeig':rdim.funR_lapeig(), # takes too long, low loss
# 'r_rpca': rdim.funR_rpca(), # takes too long, low loss
# 'r_nnp': rdim.funR_nnp(), # takes too long, low loss
# 'r_spmds':rdim.funR_spmds() # takes too long, low loss and error: 'Q[, 1:dim_o]: subscript out of bounds'
# 'r_crca':rdim.funR_crca(), # implemented in python which is much faster
# 'r_crda':rdim.funR_crda(), # takes too long, very good loss, next: implement in python
# 'r_extlpp': rdim.funR_extlpp(), # takes too long, low loss
# 'r_lamp': rdim.funR_lamp(), # takes too long, low loss
# 'r_sammon': rdim.funR_sammon(), # python implementation is better and faster
# 'r_lmds': rdim.funR_lmds(), # takes too long, low loss and similar method mds much better
# 'r_npca': rdim.funR_npca(), # takes too long, low loss
# 'r_olpp': rdim.funR_olpp(), # takes too long, low loss
# 'r_rndproj': rdim.funR_rndproj(), # takes too long, low loss
# 'r_udp': rdim.funR_udp(), # slowest method
# 'r_cisomap': rdim.funR_cisomap(), # takes too long, low loss
# 'r_fastmap': rdim.funR_fastmap(), # takes too long, low loss
# 'r_spe': rdim.funR_spe(), # takes too long, low loss and ispe similar and better

# 'py_mds': python.py_mds(), r implementation faster, better and no hps
# 'py_spectral_embedding': python.spectral_embedding(),
# 'py_dictlearn_mb': python.dictionary_learning_mini_batch(), # takes too long, low loss
# 'py_tsne': python.tsne(), # takes too long, low loss
# 'py_pca_kernel': python.pca_kernel(), # many errors, no good results
# 'py_isomap':  python.isomap(), # good results, slow
# 'py_pca_sparse_mb': python.py_pca_sparse_mini_batch(), # takes too long, low loss
# 'py_fa': python.py_factor_analysis(),# takes too long, low loss
# 'py_fastica':  python.py_fast_ica(),# takes too long, low loss
# 'py_nmf': python.py_non_negative_matrix_factorization(),# takes too long, low loss
# 'py_lle': python.locally_linear_embedding(), # many errors, no good results


## erased after first tests early 2022
# 'cnpe', # slow
# 'spca', # slow
# 'dictlearn', # slow
# 'isoproj', # slow
# 'kudp', # slow
# 'lltsa', # slow
# 'spp', # not working covid, spectra
# 'nonpp', # not working covid, spectra
# 'bmds', # better than pca but 3000 times slower
# 'cisomap', # slow
# 'dm', # slow
# 'dve', # slow
# 'iltsa', # slow
# 'keca', # slow
# 'kpca', # slow
# 'mve', # slow
# 'mvu', # slow
# 'phate', # slow
# 'ree', # slow
# 'sne', # slow
# 'spp': rdim.funR_spp(), # very slow! 50sec (pca: 0.2), inacurate: covid: 0.772 (pca: 0.991)
# 'bmds':rdim.funR_bmds(), # covid: very slow 1400sec
# 'fssem', # max.k, preprocess, takes >30min for for unscaled data 1800/31feat
# 'lpe', # slow


# OTHERS, NOT USED
## 'latent_da': python.latent_ditrichlet_allocation(n=ndim), # Textverarbeitung Topics
## 'lisomap':rdim.funR_lisomap(), # only errors with covid
## 'umap' python, error implementation
## 'iltsa': rdim.funR_iltsa(),  # too many errors with cause propably ndim


# TEXTVERARBEITUNG
# 'latent_da', 'lda'
# LABEL REQUIERED
# if function == 'lda':  # check function for hyperparameters
#     fun = disc.linear_discriminant_analysis(n=ndim)
#
# if function == 'qda':  # no hyperparameters
#     fun = disc.quadratic_discriminant_analysis(n=ndim)


## NOT IMPLEMENTED
## 'smacov': python.smacof(n=ndim), # no fit function, dificult to implement
## 'umap': python.umap(n=ndim), # not installable on M1
## 'neural_network': nn.custom_neural_network(data, dim_out=ndim, epochs=500).neural_network_dim_reduce(),



## R methods evaluation (before testing): require label or response matrix or are available in python
#'ammc', #label
#'ammm', #label
#'cca', #data1, data2
#'dagdne', #label
#'dne', #label
#'dspp', #label
#'elde', #label
#'eslpp', #label
#'fa', sklern
#'ica', sklearn
#'kmvp', # label
#'lda',  # sklearn
#'lde', # label
#'ldp', # label
#'lea', # sklearn
#'lfda', # label
#'lpfda', #label
#'lqmi', #label
#'lsda', #label
#'lsir', #label-response
#'lspp', #label
#'mfa', # label, multiple factor analysis (MFA) large number of categorical variables
#'mlie', # label
#'mmc', # label
#'mmp', # label
#'mmsd', # label
#'modp', # label
#'msd', # label
#'mvp', # label
#'odp', # label
#'olda', # label
#'opls', # data1, data2
#'pca', # facil, sklearn
#'pls', # data1, data2
#'rlda', # label
#'rsir', # label response
#'sammc', # label
#'save', # label response
#'sda', # label
#'sir', # label response
#'slpe', # label
#'slpp', # label
#'spc', # label response
#'spca', # sklearn
#'ssldp', # label
#'ulda', # label
# 'cge', # label
# 'isomap', # sklearn
# 'klde', # label
# 'klfda', # label
# 'klsda', # label
# 'kmfa', # label
# 'kmmc', # label
# 'kqmi', # label
# 'ksda', # label
# 'splapeig', # label


# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess N, sigma, alpha
# 'lpmip',

# Orthogonal Locality Preserving Projections
# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess
# 'onpp',

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess
# 'lpca2006'

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: preprocess, lambda,
#'crp',  #  not working

# !!! not working !!! something with the datainput does not work here
# hyperparameters: type, preprocess
#'bpca'


# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess N, sigma, alpha
# 'lpmip',

# Orthogonal Locality Preserving Projections
# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess
# 'onpp',

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: type, preprocess
# 'lpca2006'

# !!! not working !!! imaginal numbers, [-11729.78145264+0.j,   -279.67931118+0.j],...
# hyperparameters: preprocess, lambda,
#'crp',  #  not working

# !!! not working !!! something with the datainput does not work here
# hyperparameters: type, preprocess
#'bpca'



## R methods testet
# 'adr', # facil
# 'pca', # scaling
# 'asi', # facil
# 'bpca', # facil
# 'bpca', # facil
# 'cnpe', # type, preprocess
# 'crp',  # preprocess, lambda
# 'elpp2', # preprocess
# 'extlpp', # preprocess, numk
# 'fssem', # max.k, preprocess
# 'isoproj', # type, symetric, preprocess
# 'kmvp', # preprocess, bandwidth
# 'kudp', # type, preprocess, bandwidth
# 'ldakm', # preprocess, maxiter = 10, abstol = 0.001
# 'llp', # type, symmetric, preprocess, t, lambda
# 'lltsa', #type, symmetric, preprocess
# 'lmds', #npoints
# 'lpca2006', #type, preprocess
# 'lpe', #preprocess, numk
# 'lpmip', #type, preprocess N, sigma, alpha=0.5
# 'lpp', #type, symmetric, preprocess, t
# 'mds', # facil
# 'nonpp', #type, preprocess
# 'npca', # facil
# 'npe', # type, symmetric, weight, preprocess, regtype, regparam
# 'olpp', # type, symmetric, weight, t
# 'onpp', # type, preprocess
# 'pflpp', # preprocess
# 'ppca', # facil
# 'rndproj', # type, preprocess N, s
# 'rpcag', # k, preprocess
# 'sdlpp', # t, preprocess
# 'spp', # preprocess, reltol
# 'udp', # type, preprocess

# # the customer has predefined the percentage of rows which are kept
# else:
#     if isinstance(percent_of_rows, int) and (0 < percent_of_rows <= 100):
#         if percent_of_rows == 100:
#             data_reduced = data
#         else:
#             nrows_reduced = min(int(nrows * int(percent_of_rows/100)), nrows)
#             data_reduced = self.reduce_helper(data, nrows-1, nrows_reduced)
#     else:
#         data_reduced = data
#         logger.error(msg='nth row must be integer, and 0...100, dataset is not reduced', exc_info=True)
# documentation



# import sys
# sys.path.append('/Users/kay/PycharmProjects/asd/dimension_tools/dimension_suite/extra')
# main_path = '/Users/kay/PycharmProjects/asd/dimension_tools/dimension_suite/dim_reduce'
# sys.path.append(main_path)
# sys.path.append(main_path+'/dim_reduction')
# sys.path.append(main_path+'/helper_data_')
# sys.path.append(main_path+'/helper_metrix')
# sys.path.append(main_path+'/hyperparameters')
# sys.path.append(main_path+'/steps')
# sys.path.append('.')
# sys.path.append('..')
# sys.path.append('...')
