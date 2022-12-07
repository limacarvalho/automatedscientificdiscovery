import hypertools as ht


class Dimred_functions_hypertools:
    '''
    # TODO modify
    dimensionality reduction functions from python sklearn.decomposition.
    Those and other functions were tested with several datasets in two steps:
    1) start with parameter ranges:
    - we performed a hyperparameter optimization with full range of values and
    tested for accuracy and speed.
    2) update 20221031:
    - we reduced the hyperparameter range, updated the default hyperparameters and/or
    ereased the hyperparameter from the list (not optimized during hyperparameter optimization).
    The later was the case for most hyperparameters, they had no or only marginal effects on loss
    and/or made the dim reduction very slow when set to a certain value.
    In this case we set a dummy variable.
    '''

    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols


    def py_rpcag(self):

            '''
             - - - DESCRIPTION - - -


            - - - INFORMATION - - -

            '''
            fun = ht.reduce(

            )

            hyperpars = {

            }
            params = fun.get_params()
            return fun, params, hyperpars

