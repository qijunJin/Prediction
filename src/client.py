#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Gerardo Parrello"
__version__ = "0.0.1"
__status__ = "Prototype"

"""
client.py: Description of what client.py does.
"""

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Just add logger.debug('My message with %s', 'variable data') where you need data

import configparser as cfg
import pandas as pd
import requests as re
import datetime as dt
import json


def submit_predictions(config_file, df):

    """
    """
    
    columns = [
        'customer',
        'date',
        'billing',
    ]
    for col in columns:
        if col not in df.columns:
            raise ValueError(
                f"Column {col} is missing from dataframe."
            )

    if df.shape[1] != 3:
        raise ValueError(
            f"Wrong number of columns, should be 3."
        )

    if df.empty:
        raise ValueError("you passed an empty dataframe")

    total_customer = 1190 # total number of rows in the validation set
    if len(df.customer) < total_customer:
        raise ValueError("you have less customer than needed")

    if len(df.customer) != len(df.customer.unique()):
        raise ValueError("you have non-unique customer")

    config = cfg.ConfigParser()
    config.read(config_file)

    protocol = 'http://'
    host = config['DEFAULT']['host']
    token = config['DEFAULT']['token']

    # post predictions
    endpoint = '/predictions'
    url = protocol + host + endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(token),
        "Prefer": "return=representation",
    }
    payload = df[[
        'customer',
        'date',
        'billing',
    ]].to_json(orient='records', date_format='iso')
    r = re.post(url, data=payload, headers=headers)
    
    return(r.status_code)
