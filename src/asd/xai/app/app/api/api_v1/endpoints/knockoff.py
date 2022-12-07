

from fastapi import APIRouter, Depends, HTTPException,FastAPI, File, Form, UploadFile
from typing import List, Optional
import pandas as pd
import numpy as np
from io import BytesIO
from ml import wrapper
from ml.preprocess.data import SingletonDataSet



from utils import dasker, config as util_config
from app.core import settings
import json 

import logging
import traceback




router = APIRouter()


@router.get("/")
async def welcome():
    return {'Welcome, connection test to api.' + settings.API_V1_STR + 'successful'}


@router.post("/filter")
async def filter(
    #ksampler: str = Form(),
    sep: str = Form(),
    fdr: float = Form(),
    itr: int = Form(),
    pred_type = Form(),
    file: UploadFile = File()
):
    
    try:
            
        contents = await file.read()
        data = BytesIO(contents)
        df = pd.read_csv(data, sep=sep)
        data.close()

        if df is None:
            return {
                "Error": 'missing data'
            }

        columns = df.columns.str.lower()

        if 'y' not in columns:        
            return {
                "Error": 'target column ("y") missing'
            }            


        dataset = SingletonDataSet()
        dataset.load_data(df, itr, fdr, pred_type)
        # X, y = dataset.get_data()
        

        # param = {'df': df, 'ksampler': ksampler, 'fstat': fstat, 'fdr': fdr}
        param = {'df': df, 'fdr': fdr}

        # resulset =  wrapper.run_knockoffs(param, itr)

        resulset =  wrapper.fit_models()

        json_object = json.dumps(resulset, indent = 4)

        # wrapper.fit_models()

        return json_object
        # return resulset


    except Exception as e:
        logging.error(traceback.format_exc())




def filter_by_knockoff():
    client = dasker.get_dask_client()

    print(f'Local dask client created: localhost:{util_config.dask_local_scheduler_port}')

    ks = KnockoffFilterSim(client)

    print(f'running knockoff filter with iteration { config.itr} and seed {config.seed}')
    ks.sim_knockoffs(param, config.itr, rand=config.seed)

    return ks.rejections