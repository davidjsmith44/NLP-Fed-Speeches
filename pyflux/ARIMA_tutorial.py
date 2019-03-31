'''
Reviewing Pyflux ARIMA model

Using Sun Spot Data

'''

import numpy as np
import pandas as pd
import pyflux as pf
import datetime as datetime
import matplotlib.pyplot as plt

data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv')
data.index = data['time'].values

plt.figure(figsize=(15,5))
plt.plot(data.index,data['value'])
plt.ylabel('Sunspots')
plt.title('Yearly Sunspot Data');
plt.show()

# Specifying arbitrary ARIMA(4,0,4) model
model = pf.ARIMA(data=data, ar=4, ma=4, target='value', family = pf.Normal())

# Estimate the latent variables with a maximum likelihood estimation
x = model.fit("MLE")
x.summary()

# plot the latent variables using plot_z
model.plot_z(figsize=(15,5))

#Now getting an idea of performance using a rolling in-sample prediction
# through the plot_predict_is() method
model.plot_predict_is(h=50, figsize=(15,5))

# If we want predictions, we can use the plot_predict(): method
model.plot_predict(h=20, past_values= 20, figsize=(15,5))

ss_df = model.predict(h=15)

''' ARIMA CLASS DESCRIPTION

Parameter	Type	                Description
data	    pd.DataFrame or np.ndarray	Contains the univariate time series
ar	        int	                    The number of autoregressive lags
ma	        int	                    The number of moving average lags
integ	    int	                    How many times to difference the data (default: 0)
target	    string or int	        Which column of DataFrame/array to use.
family	    pf.Family instance	    The distribution for the time series, e.g pf.Normal()

Attributes
latent_variables  (use print('model.latent_variables))

adjust_prior(index, prior) -adjusts the priors for the latent variables
    Latent variables and their indices can be viewed by printing latent_variables
    index  (int) = index of the latent variable to change
    prior  (pf.Family instance) = prior distribution (ex pf.Normal())

fit(method, **kwargs)   Estimates latent variables for the model
    -returns an object that is the fit model
    -method is a string (ex 'MLE' or 'M-H')

plot_fit(**kwargs) = plots the fit of the model against the data

plot_ppc(T, nsims) - plots a historgram of the prior predictive check with
            a decrepancy measure of the users choice (the T param)
            Method only works if have fitted using Bayesian inference
        T is a function (ex np.mean, np.max)
        nsims is an int (how many simulations for the PPC)

plot_predict(h, past_values, intervals, **kwags)
    plots predictions of the model, along with intervals
    h is an int - how many steps forward to forecast
    past_values is an int - how many past values to plot
    intervals is a boolean - whether to plot intervals or not
    NOTE: If you use MLE or Variational Inference the intervals will
    not reflect latent variable uncertainty - only Metropolis-Hastings will
    give you fully Bayesian prediction intervals.

plot_predict_is(h, fit_once, fit_method, **Kwargs)
    Plots in-sample rolling predictions for the model. (Pretends the
    last subsection of data was out of sample and forecasts each period
    and assesses how well they did)
        h is an int - how many previous timesteps to use
        fit_one is a boolean -> whether to fit once, or for every timestep
        fit_method is a string -> which inference option to use (ex 'MLE')

plot_sample(nsims, plot_date=True)
    Plots samples form the posterior predictive density of the model if you
    fit using Bayesian inference
        nsims is an int -> how many samples to draw
        plot_data is a boolean -> whether to plot the real data as well

plot_z(indices, figsize) -retuns a plot of the latent variables and
    their associated uncertainty
        indices is an int or list - which latent variables to plot

ppc(T, nsims) - returns a p-value for a posterior predictive check
    NOTE: This method only works if you have fitted using Bayesian inference
        T is a function for discepance (eg. np.mean or np.max)
        nsims is an int - how many simulations for the PPC

predct(h, intervales = False) - returns a dataframe of the model precitions
    h is an int ->how many steps ahead to forecast
    intervals is a boolean - whethe to return prediction intervals
    NOTE: MLE and Variational Inference will NOT show reflect latent
          variable uncertainty! Only Metropois-Hastings will give full
          Bayesian prediction intervals.
    retuns a dataframe of the predictions

predict_is(h, fit_once, fit_method) -> returns a dataframe of in-sample rolling predictions
    of the model.
    h is an int -> how many previous timesteps to use
    fit_once is a boolean -> whether to fit once or at every timestep
    fit_method is a string -> which interfence option to use ex: 'MLE'


sample(nsims) - returns np.ndarray of draws of the data from the posterior
    predictive density. (Only works if fit with Bayesian inference)



 '''
