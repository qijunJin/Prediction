#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Gerardo Parrello"
__version__ = "0.0.1"
__status__ = "Prototype"

"""This script uploads a sample submission to the bitphy-ubiqum datathon live
accuracy tracker.
"""
import pandas as pd
import client

df = pd.read_csv('../datasets/sample_submission.csv')
status = client.submit_predictions('./config.ini', df)
print(status)
