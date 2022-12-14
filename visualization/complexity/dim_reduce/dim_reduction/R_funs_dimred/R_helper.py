from pathlib import Path
import numpy as np
from utils_logger import  logger
from helper_data.global_vars import *
import os
import pyper as pr
r = pr.R(use_pandas = True)
r('library(Rdimtools)')
r('Sys.setenv(LANGUAGE="en")')



class R_helper:
    '''
    class to run R functions in Python with the pyper package.
    thanks to Xiao-Qin Xia for the pyper package.

    Pyper repository:
    https://pypi.org/project/PypeR/
    Author: Xiao-Qin Xia
    '''

    def __init__(self):
        self.loss_fun = globalvar_loss_function


    def erase_file(self, path: str):
        '''
        remove file if exists
        :param path: str,
            filepath
        '''
        if os.path.exists(path):
            os.remove(path)


    def path_dir_errors(self) -> str:
        '''
        get relative path to directory where the r_errors file will be temporarily saved.
        :return: str,
            string with the path
        '''
        base_path = Path(__file__).parent
        path = (base_path /'r_errors').resolve()
        return str(path)


    def error_directory(self, path_dir_err: str):
        '''
        check if directory exist and create one if not exist.
        :param path_dir_err: str,
            path to directory where the r_errors file will be temporarily saved
        '''
        if os.path.isdir(path_dir_err):
            pass
        else:
            os.mkdir(path_dir_err)


    def r_error_init(self):
        '''
        Inititialization of R error handling.
        1) it creates a r_error directory if not exists.
        2) than it removes the rerror.txt file from this directory if exists.
        '''
        # relative path of r_error folder
        path_dir_err = self.path_dir_errors()
        # check if r_errors directory exists, if True: ok, if not create one
        self.error_directory(path_dir_err)
        # create path for error file
        self.path_error_file = str(path_dir_err + '/rerror.txt')
        # erase rerror.txt file from directory if exists.
        self.erase_file(self.path_error_file)


    def r_error_handler(self, fun_id: str):
        '''
        Workaround to catch and print R errors.
        In case of errors, they are catched from R and saved as a textfile.
        Then, the textfile is opened and the error is logged.
        In the last step, the textfile is removed.
        :param fun_id: str,
            function identifier for example: 'py_pca'
        '''
        if os.path.exists(self.path_error_file):
            f = open(self.path_error_file, 'r')
            file_contents = f.read()
            logger.error(msg=('R '+fun_id+' '+file_contents), exc_info=True)
            f.close()
            if os.path.exists(self.path_error_file):
                os.remove(self.path_error_file)


    def r_command_string(self, fun_id: str, default_parameters: str, arg_string: str, dim_low: int) -> str:
        '''
        Creates the R command in string format.
        We need to take special care of comas, special characters and empty spaces.
        If not set properly, R will no execute or throw an error.
        The command contains a try, catch expression to catch error messages from R.
        :param fun_id: str,
            function identifier for example: 'py_pca'.
        :param default_parameters: str,
            custom default paramaters.
        :param arg_string: str,
            string of dict with parameters and values of hyperparameters.
        :param dim_low: int,
            low dimension
        :return: str,
            R-command in string format
        '''
        # if argstring is empty (no hyperparameters) or is a dictionary is a dummy value in string.
        if 'empty' in arg_string or isinstance(arg_string, dict):
            arg_string = ''
        else:
            arg_string = arg_string

        # add a comma if there is an arg string.
        if arg_string == '':
            s = ' '
        else:
            s = ','

        # remove 'Rfun: ' R-function identifier from default_parameters
        default_parameters = default_parameters.replace('Rfun: ', '').replace('Rfun','')

        # R function, erase the r_ substring in the function identifier
        fun_id = str(fun_id).strip().replace('r_','')

        # build command string
        command_string = str('tryCatch(expr = {'
                            + str('low_r <- do.' + fun_id + '(as.matrix(rdata),'
                            + default_parameters + 'ndim=' + str(dim_low) + s + arg_string + ')')
                            + str('}, error = function(e){ write.table(as.character(e), file = "'
                            + self.path_error_file + '")})'))
        # example: 'tryCatch(expr = {low_r <- do.udp(X, ndim=2, type=c("proportion", 0.1)},
        #           error = function(e){ write.table(as.character(e), file="./r_errors/rerror.txt")})'
        logger.info(msg='start dimreduction: '+command_string)
        return command_string


    def r_function_exe(self,
                       fun_id: str,
                       default_parameters: str,
                       arg_string: str,
                       data_high: np.array,
                       dim_low: int) -> (np.array, dict):
        '''
        dimensionality reduction with R function and updated parameters.
        The R command string is provided to pyper which runs R and returns the low dimensional
        dataset as 'low_r$Y'.
        The Hyperparameter string is converted to a dictionary.
        :param default_parameters: str,
            string with hyperparameters
        :param fun_id: str,
            function identifier for example: 'py_pca'
        :param arg_string: str,
            R-command in string format
        :param data_high: np.array,
            high dimensional data
        :param dim_low: int,
            low dimension
        :return: (np.array, dict),
            low dimensional data and dict with hyperparameters
        '''
        # initialize the error file, more info above
        self.r_error_init()
        # build R command string
        arg_string_cmd = self.r_command_string(fun_id, default_parameters, arg_string=arg_string, dim_low=dim_low)
        # execute R command
        r('low_r <- 0')
        r.assign('rdata', data_high)
        r(arg_string_cmd)
        # get error message and erase the file
        self.r_error_handler(fun_id=fun_id)
        # get reduced data
        data_low = r.get('low_r$Y')
        # hyperparameters, convert from string into a dictionary
        hyperparameters = self.argstring_R_to_dict(arg_string)
        return data_low, hyperparameters



    def argstring_R_to_dict(self, arg_string: str) -> dict:
        '''
        convert the R string into a dictionary. The argstring contains the keys and
        values of the hyperparameters. USAGE: dim reductions and documentation.
        :param arg_string: str,
            R-command string with hyperparameter keys and values
        :return: dict,
            dictionary with hyperparameters and values
        '''
        # spe, ispe: substitute the '=' string by '---', because = is used for splitting parameters
        # below --- will be replaced by =
        if 'dist(x,method=' in arg_string or 'p=3' in arg_string:
            arg_string = arg_string.replace('dist(x,method=', 'dist(x,method---')
            arg_string = arg_string.replace('"minkowski",p=', '"minkowski",p---')
        try:
            # if its a string
            if isinstance(arg_string, str):
                if '=' in arg_string:
                    dict_argstring = dict(subString.split('=') for subString in arg_string.split(', '))
                    # spe, ispe: here we need to substitute the '---' again by =
                    for key, value in dict_argstring.items():
                        if '---' in value:
                            dict_argstring[key] = value.replace('---', '=')
                else:
                    dict_argstring = {'empty': 'empty'}
            # if its a dictionary
            else:
                dict_argstring = arg_string
        except:
            dict_argstring = {'empty': 'empty'}
            logger.error(msg='params datatype must be string or dict', exc_info=True)
        return dict_argstring


    def dict_to_r_string(self, dict_params: dict) -> str:
        '''
        converts dictionary of key, value items of hyperparameters into a string.
        :param dict_params: dict,
            with parameters and values of hyperparameters
        :return: str,
            dictionary with parameters and values as string
        '''
        arg_string = ''
        # loop through the dictionary
        for key, value in dict_params.items():
            # in case ther are no hyperparameters (empty)
            if key == 'empty':
                return arg_string
            # add the new key, value pair to the string
            else:
                arg_string = arg_string + str(key) + '=' + str(value) + ', '
        # in case of a extra coma, remove it
        if arg_string.endswith(', '):
            arg_string = arg_string[:-2]
        return arg_string