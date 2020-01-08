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

def getCommodities():
    url = 'https://apps.fas.usda.gov/PSDOnlineDataServices/api/LookupData/GetCommodities'
    headers = {'api_key': '64753A3D-5BF0-42D7-86D5-F649302D8C7D'}
    params = {}
    commodities = request_get(url, headers, params)
    
    return commodities

def getComodityCode(commodityName):
    commodities = getCommodities()
    code = commodities[commodities.CommodityName.str.strip().isin([commodityName])].CommodityCode.values[0]
    
    return code

def getCommodityDataByYear(code, year):
    url = 'https://apps.fas.usda.gov/PSDOnlineDataServices/api/CommodityData/GetCommodityDataByYear'
    headers = {'api_key': '64753A3D-5BF0-42D7-86D5-F649302D8C7D'}
    params = {
            'commodityCode': code,
            'marketYear': year
            }
    df = request_get(url, headers, params)
    
    return df

code = getComodityCode('Meat, Swine')
year = '2019'
commodityData = getCommodityDataByYear(code, year)
