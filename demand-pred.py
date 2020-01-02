# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:21:05 2019

@author: jsyi
"""

import requests
import pandas as pd

def request_get(api_url, headers, params):
    response = requests.get(api_url, headers=headers, params=params)
    json_response = response.json()
    df = pd.DataFrame(json_response)
    
    return df

def getCommodityDataByYear():    
    url = 'https://apps.fas.usda.gov/PSDOnlineDataServices/api/CommodityData/GetCommodityDataByYear'
    headers = {'api_key': '64753A3D-5BF0-42D7-86D5-F649302D8C7D'}
    params = {
            'commodityCode': '0113000',
            'marketYear': '2019'
            }
    df = request_get(url, headers, params)
    
    return df

commodityData = getCommodityDataByYear()
