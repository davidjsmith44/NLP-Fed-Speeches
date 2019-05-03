''' ForecastModel Class '''
import pandas as pd
import numpy as np
import math
from collections import Counter
import pyflux as pf
from sklearn.decomposition import PCA

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
        self.dep_vars       = model_inputs['dep_vars']
        self.pca            = None
        self.components     = None
        self.shocks         = None

        if self.model_class == 'ARIMA':
            self.model_type     = model_inputs['model_type']
            hyper_params = model_inputs['hyper_params']
            self.ar = hyper_params['ar']
            self.ma = hyper_params['ma']
            self.diff_order = hyper_params['diff_ord']
            self.family = pf.Normal()
            self.formula = None
            self.num_components = model_inputs['num_components']

        elif self.model_class == 'ARIMAX':
            self.model_type  = model_inputs['model_type']
            hyper_params = model_inputs['hyper_params']
            self.ar = hyper_params['ar']
            self.ma = hyper_params['ma']
            self.diff_order = hyper_params['diff_ord']
            self.family = pf.Normal()
            self.formula = model_inputs['formula']
            self.num_components = model_inputs['num_components']


        else: # this will be gaussian
            self.model_type  = None
            self.ar = None
            self.ma = None
            self.diff_order = None
            self.family = None
            self.formula = None
            self.formula = None   # This is the string for ARIMAX models that needs to be used
            self.num_components = model_inputs['num_components']

    def fit(self, X):
        '''
        Takes the model and initialized the time series object to it with the dataframe X
        Then fits the model using the dataframe X
        '''

        if self.target != 'PCA':
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

        else:   # case where we have a PCA model
                tenors = [0.5, 1, 2, 3, 5, 7, 10]
                test = PCA(n_components = self.num_components, random_state=44)
                test.fit(X[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']])
                # NEED TO STORE LATENT VARIABLES AND PERCENTAGE EXPLAINED
                self.pca = test
                self.pct_var_expl = test.explained_variance_ratio_
                self.components = test.components_
                shocks = test.fit_transform(X[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']])
                self.shocks = shocks
                #print('BELOW ARE THE SHOCKS: ',shocks)
                # need to fit three components to this so self.model is a list
                self.model = []
                if self.model_class == 'ARIMA':
                    for i in range(self.num_components):
                        model = self.model_type(data = self.shocks[:,i],
                            ar= self.ar,
                            ma= self.ma,
                            integ= self.diff_order,
                            target = self.shocks[:,i],
                            family=self.family)
                        model.fit('MLE')
                        # m = model.fit('MLE')
                        # m.summary()
                        self.model.append(model)

                elif self.model_class == 'ARIMAX':
                    # here we need to join X with the shocks one by one
                    # based on the number of components
                    for i in range(self.num_components):
                        X_arima = X.copy()
                        X_arima['shock']=shocks[:,i]
                        #model['target'] = 'shock'
                        func_str = self.formula
                        func_list = func_str.split(sep='~')
                        self.formula = 'shock' + '~' + func_list[1]

                        # NOTE: May need to adjust if I am using levels here!
                        model = self.model_type(data = X_arima,
                            formula = self.formula,
                            ar= self.ar,
                            ma= self.ma,
                            integ= self.diff_order,
                            family=self.family)
                        model.fit('MLE')
                        self.model.append(model)

                else:  # The case where we have Gaussian model
                    #for i in range(self.num_components):
                    #    model = np.mean(X[self.shocks[:,i]], axis=0)
                    #    self.model.append(model)
                    self.model = np.mean(self.shocks, axis = 0)
                    # need to transform the shocks
                    # some plotting?
                    # what do I need to store here to have everything I need


    def predict_one(self, X):
        ''' This method predicts one day forward on the variables. It must first
        call the create_oos_data() method to create a dataframe to be used for
        the forecast. '''

        if self.target != 'PCA':
            if self.model_class == 'ARIMA':
                this_pred = self.model.predict(h=1, intervals=False)
                return this_pred.iloc[0,0]
                #return self.model.predict(h=1, intervals=False)

            elif self.model_class == 'ARIMAX':
                oos_data = self.create_oos_data(X)
                this_pred = self.model.predict(h=1, oos_data = oos_data)
                return this_pred.iloc[0,0]
                #return self.model.predict(h=1, oos_data = oos_data)

            else:   # This is the Gaussian Model
                # the gaussian self.model contains the mean change in the rate
                return self.model

        # now handeling the PCA cases to predict one.
        else:
            this_prediction = np.zeros(shape=(1,7))
            if self.model_class == 'ARIMA':
                for i in range(self.num_components):
                    # below is the prediciton for the shock
                    this_shock = self.model[i].predict(h=1, intervals=False)
                    #print('in pca model with shocks')
                    #print('this is shock i: ', this_shock)
                    this_shock = this_shock.iloc[0,0]
                    this_impact = this_shock * self.components[i,:]
                    this_prediction += this_impact
                return this_prediction

            elif self.model_class == 'ARIMAX':
                for i in range(self.num_components):
                    X_arima = X.copy()
                    X_arima['shock']=self.shocks[:,i]
                    oos_data = self.create_oos_data(X_arima)
                    # below is the prediciton for the shock
                    this_shock = self.model[i].predict(h=1, oos_data=oos_data)
                    this_shock = this_shock.iloc[0,0]
                    print('here is the shock for ARIMAX', this_shock)
                    print('here are the ARIMAX components', self.components)
                    this_impact = this_shock * self.components[i,:]
                    this_prediction += this_impact
                return this_prediction

            else:   #case where we have a Gaussian model
                for i in range(self.num_components):
                    # below is the prediciton for the shock
                    this_shock = self.model[i]
                    this_impact = this_shock * self.components[i,:]
                    this_prediction += this_impact
                return this_prediction

    def create_oos_data(self, X):
        '''This method build the data needed for a one period forecast and alters the
            the speech data to be zeros (so one period forecast with no speech data'''
        oos_data =  X.iloc[[-1]]
        # now reset all of the fed speech data to be zero'
        # oos_data[['ed_last', 'ed_avg_n', 'cos_last', 'cos_avg_n']] = 0
        return oos_data
