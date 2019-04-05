''' Basic interest rate model


Check that the forward time series is stationary or not

BASIC MODEL: Change in rate equals mean change in rate + noise
rt+1 - rt = + mu + sigma


STOCHASTIC PROCESS MODEL
rt+1 = rt + alpha *dt + simga * sqrt(dt) * dx

    where
        rt+1 =  the rate at time t+1
        rt  =   the rate at time t
        alpha = drift (set equal to forward rate)
        dt =    1/260 (trading days)
        sigma = historical volatility of the rate
        dx    = standard normal noise

CALIBRATION:
    alpha = slope of the forward curve (fowards in X periods - current spot rate)
    sigma = historical std deviation of the spot rate (assumes this is constant over the peiord)

STEPS:
    1. Calculate slope of forward curve and adjust to be an annualized rate
        Do this in a separate dataframe (before split of training and cross valudation) after calculating the forwards
        Call it drift_df
    2. Calculate the std deviation of the entire series between test start and now
    3. Demonstrate the one period change (NOTE: THIS IS NOT REALLY A FACTOR FOR THE ONE PERIOD CHANGE)

    4. Demonstrate the expected value of interest rates with this model (no noise)
    5. Calculate difference from forecast and actual and store this data
    6. Create plots of these variables


'''

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyflux as pf
import datetime as datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle



X = pickle.load(open("../data/interest_rate_data", "rb" ))
X_fwds = pickle.load(open('../data/forward_rates', 'rb'))
X_zeros = pickle.load(open('../data/zero_rates', 'rb'))



#df_FX = pickle.load( open( "data/FX_data", "rb" ) )

# Loading up the federal reserve metrics and incorporating into our dataframes
fed_metrics = pickle.load( open( "../data/mvp_cosine_sim", "rb" ) )
cos_last = fed_metrics['cos_last']
cos_avg_n = fed_metrics['cos_avg_n']
ed_last = fed_metrics['ed_last']
ed_avg_n = fed_metrics['ed_avg_n']
fed_dates = fed_metrics['dates']

# USING THE PD MERGE BRANDON TAUGHT
avgstats = pd.DataFrame({'date':fed_dates,
                        'ed_last': ed_last,
                        'ed_avg_n': ed_avg_n,
                        'cos_last': cos_last,
                        'cos_avg_n': cos_avg_n}).groupby('date').mean()
avgstats.index = pd.to_datetime(avgstats.index)

X = X.merge(avgstats, how='left', left_index = True, right_index = True)
X_fwds = X_fwds.merge(avgstats, how='left', left_index = True, right_index = True)
X_zeros = X_zeros.merge(avgstats, how = 'left', left_index = True, right_index = True)

X.fillna(value=0, inplace=True)
X_fwds.fillna(value=0, inplace=True)
X_zeros.fillna(value=0, inplace=True)


# Train/Test Split for time series model
total_obs = len(X)
train_int = int(round(total_obs*.7, 0))
cv_int = int(round(total_obs*.85, 0))

fwd_train = X_fwds[0:train_int]
fwd_cv = X_fwds[train_int:cv_int]
fwd_test = X_fwds[cv_int:]

zero_train = X_zeros[0:train_int]
zero_cv = X_zeros[train_int:cv_int]
zero_test = X_zeros[cv_int:]

X_train = X[0:train_int]
X_cv = X[train_int:cv_int]
X_test = X[cv_int:]

def plot_interest_rates(X_train):
    fig = figure(siz)
    plt.title("Historical US Treasury Interest Rates")
    plt.plot(X_train['three_m', 'six_m', 'one_y', 'two_y','three_y', 'five_y', 'seven_y', 'ten_y'])