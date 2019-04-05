import numpy as np
import pandas as pd
import pyflux as pf
import datetime as datetime
import matplotlib.pyplot as plt

import os
import pickle

X_fwds = pickle.load(open('../data/forward_rates', 'rb'))


#df_FX = pickle.load( open( "data/FX_data", "rb" ) )
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

X_fwds = X_fwds.merge(avgstats, how='left', left_index = True, right_index = True)

X_fwds.fillna(value=0, inplace=True)

total_obs = len(X_fwds)
train_int = int(round(total_obs*.7, 0))
cv_int = int(round(total_obs*.85, 0))

fwd_train = X_fwds[0:train_int]
fwd_cv = X_fwds[train_int:cv_int]
fwd_test = X_fwds[cv_int:]

# now working on setting up the model dictionary

dict_params = {'ar':1, 'ma': 1, "diff_ord": 1, 'target':'10 YR'}