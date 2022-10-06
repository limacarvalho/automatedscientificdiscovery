
import requests
import logging
import knockpy
from knockpy.knockoff_filter import KnockoffFilter
import pandas as pd
import numpy as np
import json


def test_deepknockoff():    

    url = 'http://127.0.0.1:8000/api/v1/knockoff/filter'
    fp = './data/sample.csv'


    files = {'file': open(fp,'rb')} # for sending a single file
    param = { 'ksampler': 'gaussian', 'fstat': 'lasso', 'fdr': 0.1, 'itr': 100}
    # token ={"name": "foo", "point": 0.13, "is_accepted": False}
    resp = requests.post(url=url, data=param, files=files)

    print(resp.json())
    #print(resp.request.headers['content-type'])

    # Directly from dictionary


def test_knockoff(i):
    #df = pd.read_csv('./data/sample.csv')

    url = 'http://127.0.0.1:8000/api/v1/knockoff/filter'
    fp = './data/20220319_covid_merge_processed.csv'


    files = {'file': open(fp,'rb')} # for sending a single file
    param = {'sep':',', 'fdr': 0.1, 'itr': 2, 'pred_type' : 'regression'}
    # token ={"name": "foo", "point": 0.13, "is_accepted": False}
    resp = requests.post(url=url, data=param, files=files)

    print(resp.json())

    file = 'resp_'+str(i)+'.json'
    with open(file, 'w') as outfile:
        json.dump(resp.json(), outfile)




if __name__ == '__main__':
    #for i in range(0, 20):
     #   test_knockoff(i)
    test_knockoff(0)