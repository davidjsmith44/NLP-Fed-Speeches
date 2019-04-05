''' ForecastModel Class '''
import pandas as pd
import numpy as np
import math
from collections import Counter
import pyflux as pf


class ForecastModel(object):
    ''' The forecast model includes all hpyerparameters of the class and methods to estimate forwards


    this_name = 'Normal ARIMA(2,1,2)'
    hyper_params= {'ar':2, 'ma': 2, "diff_ord": 1, 'target':'ten_y'}
    forecast = 0
    model_inputs = {'model_type': 'ARIMA',
                    'name': this_name,
                    'target': this_target,  -used for ARIMA models ignored for ARIMAX
                    'hyper_params': hyper_params,
                    'foreacst': forecast}


    The init function takes a dictionary parm_dict with the following keys
    'this_name' = the name of the model ex: ARIMA(2,1,2)
    'hyper_params'      -a dictionary of hyperparameters with the following keys
                            'ar'
                            'ma'
                            'diff_ord'
    'num_components'    -an int containing the number of principal components to use
    'forecast' an empty array the size of the cross validation set that gets populated
                in the forecast function

    The initial classes are created with the init function

    The fit method is called with new data each new date
    The then the predict_one function is updated

    This classes predict_one will be called outside of this class
    '''
    def __init__(self, model_inputs):
        '''
        Initialize a time series model class
        takes a dictionary model_inputs and builds parameters to be used to estimate model
        NOTE: The time series model object is not initialized here,
        since this class will be called multiple times during cross validation.
        The fit method will do three things
            1. Receive a unique dataset (passed into the method)
            2. Initialize the ts model (for example ARIMAX) on the data
            3. Fit the method

        The predict_one method will take the fitted class and predict a one day forecast
        '''
        self.model_class    = model_inputs['model_class']
        self.model_name     = model_inputs['name']
        self.target         = model_inputs['target']

        if self.model_class == 'ARIMA':
            self.model_type     = model_inputs['model_type']
            hyper_params = model_inputs['hyper_params']
            self.ar = hyper_params['ar']
            self.ma = hyper_params['ma']
            self.diff_order = hyper_params['diff_ord']
            self.family = pf.Normal()
            self.formula = None
            self.num_components = None

        elif self.model_class == 'ARIMAX':
            self.model_type  = model_inputs['model_type']
            hyper_params = model_inputs['hyper_params']
            self.ar = hyper_params['ar']
            self.ma = hyper_params['ma']
            self.diff_order = hyper_params['diff_ord']
            self.family = pf.Normal()
            self.formula = model_inputs['formula']
            self.num_components = None

        else: # this will be gaussian and later PCA!
            self.model_type     = None
            self.ar = None
            self.ma = None
            self.diff_order = None
            self.family = None
            self.formula = None
            self.num_components = None
            self.formula = None   # This is the string for ARIMAX models that needs to be used

    def fit(self, X):
        '''
        Takes the model and initialized the time series object to it with the dataframe X
        Then fits the model using the dataframe X
        '''
        if self.model_class == 'ARIMA':
            model = self.model_type(data = X,
                ar= self.ar,
                ma= self.ma,
                integ= self.diff_order,
                target = self.target,
                family=self.family)
            model.fit('MLE')
            # m = model.fit('MLE')
            # m.summary()
            self.model = model

        elif self.model_class == 'ARIMAX':
            model = self.model_type(data = X,
                formula = self.formula,
                ar= self.ar,
                ma= self.ma,
                integ= self.diff_order,
                family=self.family)
            model.fit('MLE')
            # m = model.fit('MLE')
            # m.summary()
            self.model = model

        else:  # The case where we have Gaussian model
            model = np.mean(X[self.target])
            self.model = model

    def predict_one(self, X):
        ''' This method predicts one day forward on the variables. It must first
        call the create_oos_data() method to create a dataframe to be used for
        the forecast. '''

        if self.model_class == 'ARIMA':
            return self.model.predict(h=1, intervals=False)

        elif self.model_class == 'ARIMAX':
            oos_data = self.create_oos_data(X)
            return self.model.predict(h=1, oos_data = oos_data)

        else:   # This is the Gaussian Model
            # the gaussian self.model contains the mean change in the rate
            return self.model

    def create_oos_data(self, X):
        '''This method build the data needed for a one period forecast and alters the
            the speech data to be zeros (so one period forecast with no speech data'''
        oos_data =  X.iloc[[-1]]
        # now reset all of the fed speech data to be zero'
        # oos_data[['ed_last', 'ed_avg_n', 'cos_last', 'cos_avg_n']] = 0
        return oos_data
