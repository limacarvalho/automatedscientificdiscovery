from pathlib import Path
import numpy as np
from helper_data.utils_data import Helper
from utils_logger import  logger
from helper_data.global_vars import *
import os
import pyper as pr
r = pr.R(use_pandas = True)
r('library(Rdimtools)')
r('Sys.setenv(LANGUAGE="en")')



class R_helper:
    '''
    class to run R functions on Python with the pyper package.

    Pyper repository:
    https://pypi.org/project/PypeR/
    Author: Xiao-Qin Xia
    LICENCE: GPL
    '''

    def __init__(self):
        self.loss_fun = globalvar_loss_function


    def path_dir_errors(self) -> str:
        '''
        get relative path to r_errors directory.
        :return: string with the path
        '''
        base_path = Path(__file__).parent
        path = (base_path /'r_errors').resolve()
        return str(path)


    def r_error_init(self):
        '''
        Inititialization of R error handler.
        creates a r_error directory if not exists.
        If exists it removes the rerror.txt file from this directory if exists.
        '''
        # relative path of r_error folder
        path_dir_err = self.path_dir_errors()
        # check if r_errors directory exists, if True: ok, if not create one
        Helper().check_if_dir_exists(path_dir_err)
        # create path for error file
        self.path_error_file = str(path_dir_err + '/rerror.txt')
        # erase rerror.txt file from directory if exists.
        Helper().erase_file(self.path_error_file)


    def r_error_handler(self, fun_id: str):
        '''
        Workaround to catch and print R errors.
        In case of errors, they are catched from R and saved as a textfile.
        In the next step the textfile is opened and the error is printed.
        Last step: erase textfile
        :param fun_id: str, function identifier for example: 'py_pca'
        '''
        if os.path.exists(self.path_error_file):
            f = open(self.path_error_file, 'r')
            file_contents = f.read()
            logger.error(msg=('R '+fun_id+' '+file_contents), exc_info=True)
            f.close()
            if os.path.exists(self.path_error_file):
                os.remove(self.path_error_file)


    def r_command_string(self, fun_id: str, default_parameters: str, arg_string: str, ndim: int) -> str:
        '''
        Creates the R command (a string) executed by pyper.
        We need to take special care of comas, special characters and empty spaces.
        If not set properly, R will no execute or throw an error.
        The command contains a try, catch expression to catch error messages from R.
         .
        :param default_parameters: custom default paramaters.
        :param fun_id: str, function identifier for example: 'py_pca'
        :param arg_string: str, string of dict with parameters and values of hyperparameters
        :param ndim: int, low dimension
        :return: str, R-command string
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
                            + default_parameters + 'ndim=' + str(ndim) + s + arg_string + ')')
                            + str('}, error = function(e){ write.table(as.character(e), file = "'
                            + self.path_error_file + '")})'))
        # example:  X=high dim data; type=.. hyperparameter; "asd/r_errors/rerror.txt" error handling
        # cmd: 'tryCatch(expr = {low_r <- do.udp(X, ndim=2, type=c("proportion", 0.1)},
        #       error = function(e){ write.table(as.character(e), file="asd/r_errors/rerror.txt")})'
        logger.info(msg=command_string)
        return command_string


    def r_function_exe(self,
                       fun_id: str,
                       default_parameters: str,
                       arg_string: str,
                       data_high: np.array,
                       dim_low: int) -> (np.array, dict):
        '''
         executes dimensionality reduction with a function from the R-dimtools package and
         custom parameters. Its executed through the Pyper package.
         It returns the low dimensional dataset as 'low_r$Y' from the R-dimtools package.
        :param default_parameters: strng with custom default parameters
        :param fun_id: str, function identifier for example: 'py_pca'
        :param arg_string: str, R-command
        :param data_high: np.array, high dimensional data
        :param dim_low: np.array, low dimensional data
        :return: (np.array, dict), low dimesnional data and dict with hyperparameters
        '''
        # initialize the error file, more info above
        self.r_error_init()

        # build R command string
        arg_string_cmd = self.r_command_string(fun_id, default_parameters, arg_string=arg_string, ndim=dim_low)

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
        convert the R string into a dictionary. The argstring contains the keys, values
        of the hyperparameters which wil be stored as a dict and used for other dim reductions
        and documentation.
        :param arg_string: str, R-command string with hyperparameter keys and values
        :return: dict, dictionary with hyperparameters and values
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
        :param dict_params: dict, with parameters and values of hyperparameters
        :return: str, dict with parameters and values as string
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