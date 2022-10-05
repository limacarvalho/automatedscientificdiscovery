import numpy as np
import traceback
from dimension_tools.dimension_suite.dim_reduce.helper_data.helper_data import Helper
from dimension_tools.dimension_suite.dim_reduce.helper_data.global_vars import *
import pyper as pr
import os

r = pr.R(use_pandas = True)
r('library(Rdimtools)')
r('Sys.setenv(LANGUAGE="en")')



class R_helper:

    def __init__(self):
        '''
        class to run r functions through the pyper function.
        '''
        self.loss_fun = globalvar_loss_function
        self.path_dir_error = globalvar_path_dir_r_errors



    def r_error_init(self):
        '''
        create r_error folder and remove rerror.txt file from directory if exists.
        Returns
        -------

        '''
        # check if r-error directory exists, if True: ok, if not create one and pass it to
        # globalvar_path_dir_r_errors
        Helper().check_if_dir_exists(self.path_dir_error)

        # crete path for error file
        self.path_error_file = str(self.path_dir_error + 'rerror.txt')

        # erase rerror.txt file from directory if exists.
        Helper().erase_file(self.path_error_file)



    def r_error_handler(self, fun_id: str):
        '''
        R error handler.
        In case of errors, they are catched from R and saved as a textfile.
        In the next step the textfile is opened and the error is printed.
        Last step: erase textfile
        I couldnt find a better way to handle R-errors from python.

        Parameters
        ----------
        fun_id : function identifier for example: 'py_pca'
        -------
        '''
        if os.path.exists(self.path_error_file) == True:
            f = open(self.path_error_file, 'r')
            file_contents = f.read()
            print('R-ERROR: ', fun_id, file_contents)
            f.close()
            os.remove(self.path_error_file)


    def r_command_string(self, fun_id: str, arg_string: str, ndim: int) -> str:
        '''
        the R command is a string which is converted to a R command and executed by pyper.
        We need to take special care of comas and any extra carachters. They might make
        interfere with the R command.
        The command itself is a try, catch expression in order to receive error messages
        from R and show them in the python console.

        Parameters
        ----------
        fun_id : function identifier
        arg_string : arg string is a string with hyperparameter information.
        ndim : low dimension

        Returns: string of R-command
        -------

        '''
        # if argstring is empty (no hyperparameters) or is a dictionary is a dummy value in string.
        if 'empty' in arg_string or isinstance(arg_string, dict):
            arg_string = ''
        else:
            arg_string = arg_string

        # comas are important in these strings, so if there has been a arg strin before
        # add a comma to seperate the one befor and the one here.
        if arg_string == '':
            s = ' '
        else:
            s = ','

        # R function, erase the r_ substring in the function identifier
        fun_id = str(fun_id).strip().replace('r_','')

        # build command string
        command_string = str('tryCatch(expr = {'
                                + str('low_r <- do.' + fun_id + '(as.matrix(rdata), ndim='
                                + str(ndim) + s + arg_string + ')')
                                + str('}, error = function(e){ write.table(as.character(e), file = "'
                                + self.path_error_file + '")})'))
        # example:  X=high dim data; type=.. hyperparameter; "asd/r_errors/rerror.txt" error handling
        # cmd: 'tryCatch(expr = {low_r <- do.udp(X, ndim=2, type=c("proportion", 0.1)},
        #                error = function(e){ write.table(as.character(e), file="asd/r_errors/rerror.txt")})'
        return command_string



    def r_function_exe(self, fun_id: str, arg_string: str, data: np.array, dim_low: int) -> tuple:
        '''
        executes R command which we have build in above function via the pyper function.
        The command is callling a dim reduce function which returns the low dimensional
        dataset as 'low_r$Y' from the R-dimtools package

        :param fun_string: R-command as string
        :param data: high dimensional data
        :return: array with reduced data
        '''
        # initialize the error file, more info above
        self.r_error_init()

        arg_string_cmd = self.r_command_string(fun_id, arg_string=arg_string, ndim=dim_low)

        # execute R command
        r('low_r <- 0')
        r.assign('rdata', data)
        r(arg_string_cmd)

        # get error message and erase the file
        self.r_error_handler(fun_id=fun_id)

        # get reduced data
        data_low = r.get('low_r$Y')

        # hyperparameters from string into a dictionary
        hyperparameters = self.argstring_R_to_dict(arg_string)
        return data_low, hyperparameters



    def argstring_R_to_dict(self, arg_string: str) -> dict:
        '''
        convert the R string into a dictionary. The argstring contains the keys, values
        of the hyperparameters.

        Parameters
        ----------
        arg_string : string of R-command containing the hyperparameter information

        Returns : dict with hyperparameters parameters as keys and values
        -------

        '''
        # spe, ispe: substitute the '=' string by '---', because = is used for splitting parameters
        if 'dist(x,method=' in arg_string:
            arg_string = arg_string.replace('dist(x,method=', 'dist(x,method---')

        try:
            # if its not empty or a dict
            if isinstance(arg_string, str):
                if '=' in arg_string:
                    dict_argstring = dict(subString.split('=') for subString in arg_string.split(', '))

                    # spe, ispe: here we need to substitute the '---' again by =
                    for key, value in dict_argstring.items():
                        if '---' in value:
                            dict_argstring[key] = value.replace('---', '=')

                else:
                    dict_argstring = {'empty': 'empty'}

            else:
                dict_argstring = arg_string
        except:
            dict_argstring = {'empty': 'empty'}
            print(globalstring_error + 'PARAMS DATATYPE MUST BE STRING OR DICT')
            print(traceback.format_exc())

        return dict_argstring


    def dict_to_r_string(self, dict_params: dict) -> str:
        '''
        convert a dict of key, value items of hyperparameters to a string.
        Parameters
        ----------
        dict_params : dict of key, value items of hyperparameters

        Returns: string of key, value items of hyperparameters
        -------

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