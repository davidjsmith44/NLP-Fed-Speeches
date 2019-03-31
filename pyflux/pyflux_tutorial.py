''' pyflux_tutorial.py '''

import pandas as pd
import numpy as np
import pandas_datareader as pdr
from datetime import datetime
import matplotlib.pyplot as plt
import pyflux as pf

a = pdr.DataReader('ADBE',  'yahoo', datetime(2005,3,30), datetime(2019,3,29))
a_prices = pd.DataFrame(a['Adj Close'].values)
a_returns = pd.DataFrame(np.diff(np.log(a['Adj Close'].values)))
a_returns.index = a.index.values[1:a.index.values.shape[0]]
a_returns.columns = ["JPM Returns"]
a_returns.head()

# Step 1: Visualization
# creating a plot of Adobe Returns
fig, axs = plt.subplots(2,1, figsize=(12,12))
axs[0].plot(a_prices)
axs[0].set_title("ADBE price since Dave Joined")
axs[1].plot(a_returns)
axs[1].set_title("ADBE returns since Dave Joined")
plt.show()

pf.acf_plot(a_returns.values.T[0], max_lag=260)
pf.acf_plot(np.square(a_returns.values.T[0]), max_lag = 260)

# Step 2: Propose a model
my_model = pf.GARCH(p=1, q=1, data=a_returns)
print(my_model.latent_variables)

#Step 3: Inference
# Using Metropolis-Hastings for approximate inference on GARCH mode
result = my_model.fit('M-H', nsims=20000)
# Ploting latent variabes alpha and beta
my_model.plot_z([1,2])

# Step 4: Evaluate Model Fit
# Plotting the series versus its predicted values
# Can check out of sample performance
# plot the fit of the GARCH model and observe that it is picking
# up volatility clustering in the series
my_model.plot_fit(figsize=(15,5))

# plotting the posterior predictive density
my_model.plot_sample(nsims=10, figsize=(15,7))

# Performing a posterior predictive check (PPC) on the features of
# the generated series, for example kurtosis
from scipy.stats import kurtosis
my_model.plot_ppc(T=kurtosis)

# Step 5: Analyse and Predict
my_model.plot_predict(h=30, figsize=(15,5))