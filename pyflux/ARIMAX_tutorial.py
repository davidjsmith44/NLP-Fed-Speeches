''' PyFlux ARIMAX totorial '''

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
#%matplotlib inline

''' Tutorial uses monthly UK driver deaths '''
data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/drivers.csv")
data.index = data['time'];

# Below are the dummy variables for strucrtural change
data.loc[(data['time']>=1983.05), 'seat_belt'] = 1;
data.loc[(data['time']<1983.05), 'seat_belt'] = 0;
data.loc[(data['time']>=1974.00), 'oil_crisis'] = 1;
data.loc[(data['time']<1974.00), 'oil_crisis'] = 0;
plt.figure(figsize=(15,5));
plt.plot(data.index,data['value']);
plt.ylabel('Driver Deaths');
plt.title('Deaths of Car Drivers in Great Britain 1969-84');
plt.plot();

'''
NOTE: Should I use a dummy variable for structural change based on the point in
economic cycle we are in?
    Possibly a dummy for the financial crisis
    Maybe a dummy variable for after the financial crisis
'''

# NOTE: They use 'patsy' notation for the model
model = pf.ARIMAX(data=data, formula = 'value~1+seat_belt+oil_crisis',
                ar=4, ma=4, family=pf.Normal())
x = model.fit("MLE")
x.summary()

# plotting the in-sample fit
model.plot_fit(figzier=(15,10))

# NOTE: To forecast we need exogenous variavbles into the future.
# Since the inteventions carry forward we can use a slice of the existing data frame
# and use plot_predict
model.plot_predict(h=10, oos_data = data.iloc[-12:], past_values =100, figsize=(15,15))

model.plot_predict_is(h=10, fit_once=False, fit_method = 'MLE')
# Attribures:  latent_variables - this is where the information is stored
#                                   about the model

model.predict(h = 3, oos_data = data.iloc[-12:], intervals=False)
'''
METHODS
    adjust_prior() - allows for adjusting the prior distribution

    fit() - estimates the latent variables
            if using 'inference' option, the method returns a results object
                (EX: Baysian or Classical - see docs)
                returns pf.Results instance

    plot_fit - plots the firt of the model against the data

    plot_ppc(T, nsims) - plots a histogram for the posterior predictive check
                        with a discrepancy measure of the user's choosing
                        This method only works if using Baysian Inference

    plot_predict(h, oos_data, past_values, intervals **kwards)
            h is an int for how many steps to forecast ahead
            oos_data is a dataframe of exogeneous variables in a frame for h steps
                    MUST BE SAME SIZE AS THE INITIAL DATAFRAME
            past_values is an int of how many past values to plot
            intervals is a boolean for whether to plot intervals or not

To be clear, the oos_data argument should be a DataFrame in the same format as the initial dataframe used to initialize the model instance. The reason is that to predict future values, you need to specify assumptions about exogenous variables for the future. For example, if you predict h steps ahead, the method will take the h first rows from oos_data and take the values for the exogenous variables that you asked for in the patsy formula.

Optional arguments include figsize - the dimensions of the figure to plot. Please note that if you use Maximum Likelihood or Variational Inference, the intervals shown will not reflect latent variable uncertainty. Only Metropolis-Hastings will give you fully Bayesian prediction intervals. Bayesian intervals with variational inference are not shown because of the limitation of mean-field inference in not accounting for posterior correlations.

    plot_presict_is(h,, fit_once, fit_method, **kwargs)
        plots in sample rolling predictions for the mode
        (user pretends last subsection of data is out-of-sample and forcasts
        after each period assess how well they did)
        fit_once is a boolean - if false it fits every time-step
        h is an int that indicates how many steps to simulate

    plot_sample(nsims, plot_data = True)
        plots samples from the posterior predictive density of the model
        This method only works if you fitted the model using Bayesian inference

