''' Transforming interest rate data into zeros and forwards '''



# Assume we have the entire X datafame in memory

#Load up interest rate data
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt

import os
import pickle

os.chdir('..')
X = pickle.load(open("data/interest_rate_data", "rb" ) )

X = X/100
zeros = X[['3 MO', '6 MO', '1 YR']].copy()

''' calculating the forward rates for the two year '''
price = 100


coupons = price * X['2 YR'] / 100