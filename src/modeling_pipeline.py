''' Post Data Pull Workflow

At this point in the program we have pulled and processed both the Fed speeches and interest rate date.
FED SPEECHES
    The results of this process have been saved in a pickle file in the data subdirectory called
        'ts_cosine_sim.p'
        It contains a list with the following three variables [ts_cos_last, ts_cos_avg_n, ts_date]
            ts_cos_last     contains the cosine similarity of the last fed speech to the ones on the ts_date
            ts_cos_avg_n    contains the cos. sim of the last 50 speeches to the ones made on ts_date
            ts_date is      contains the date in a crappy np.datetime64 object

    os.chdir("..")
    pickle_out = open('../data/ts_cosine_sim', 'wb')
    pickle.dump([ts_cos_last, ts_cos_avg_n, ts_dates], pickle_out)
    pickle_out.close()

INTEREST RATE DATA
    The interest rate data has been pulled from quandl and pre-processed. The data sits in a file called 'interest_rate_data.p'


STEPS:
    1. Load up the interest rate data
    2. Split into train/test
    3. Convert into three datasets
        raw yields
        discrete forwards
        cont.comp forwards
    4. EDA on the data in the training + cv set
    5. build model pipeline
        -estimate parameters of the model over the training set
        -build CV update parameters/etc
    6. Compare the results

'''



''' Working on the Gaussian Model for the MVP on base level rates '''
def forecast_gaussian(X):
    ''' Mean zero interest rates we just take the change here '''
    fcst = 0
    return fcst

''' ARIMA MODEL ON BASE RATES
For now lets just look at the 10 year
'''

def build_ARIMA_model(X, ar, ma, diff_ord, target):

    model = pf.ARIMA(data=X, ar=4, ma=4, integ = diff_ord, target=target, family = pf.Normal())

    # Estimate the latent variables with a maximum likelihood estimation
    model.fit("MLE")
    #x.summary()
    pred = model.predict(h=1)
    last_rate = X['10 YR'][-1]
    this_shock = pred['Differenced 10 YR'].iloc[0]
    next_rate = last_rate + this_shock

    return next_rate

def update_cv_data(X_train, X_cv, i):

    temp = X_cv[0:i]
    frames = [X_train, temp]
    X_this_cv = pd.concat(frames)

    return X_this_cv

def create_cv_forecasts(X_train, X_cv, dict_params):
    cv_len = len(X_cv)
    forecasts = np.zeros(shape=[cv_len,1])
    ar = dict_params['ar']
    ma = dict_params['ma']
    diff_ord = dict_params['diff_ord']
    target = dict_params[target]
    for i in range(cv_len):
        print(i)
        this_X = update_cv_data(X_train, X_cv, i)
        forecasts[i] = build_ARIMA_model(this_X, ar, ma, diff_ord,
                            target, family = pf.Normal())
    return forecasts

def cross_validate_models(model_list, X_train, X_cv):
    '''
    Building a overlaying function that handles the cross validation

    Will take as an input model_list that includes the model type
    and all of the hyper parameters of the model

    INPUTS:
        X_train -   the dataframe containing the training dataset
        X_cv -      the dataframe containing the cross_val dataset
        model_list  A dictionary containing the hyper parameters
                     of the model

    OUTPUTS
        The forecast of the model (one day interest rate forecast) will be stored
        in the model_list['foreacst'] section

 'model_type'    {'Gaussian', 'ARIMA', 'ARIMAX'}
    'name'          name given to model for charting purposes
    'target_class'  {'rates', 'forwards', 'cc_forwards'}
    'hyper_parms'    A dictoinary containing the hyperparmeters of the model
    'forecast'      A zero, initially that get populated with the daily forecasts over CV period

    '''
    # setting up the initial arima model
    for idx, item in enumerate(model_list):
        # Making sure we use the correct interest rate transformation

        if item['target_class']=='rates':
            print('This one uses rates')

        elif item['target_class']=='forwards':
            print('This model uses forwards')
        elif item['target_class']=='cc_forwards':
            print("This model uses continuously compounded forwards")

        if item['model_type']== 'ARIMA':
            model_list[i]['forecast'] = create_cv_forecasts(X_train, X_cv, item['hyper_params'])

    return model_list





#Load up interest rate data
import numpy as np
import pandas as pd
import pyflux as pf
import datetime as datetime
import matplotlib.pyplot as plt
import os
import pickle
from ForecastModel import ForecastModel

if __name__ == '__main__':

    X_fwds = pickle.load(open('../data/forward_rates', 'rb'))
    X_fwds = df_add_first_diff(X_fwds)

    #df_FX = pickle.load( open( "data/FX_data", "rb" ) )
    # Loading up the federal reserve speech data
    fed_metrics = pickle.load( open( "../data/mvp_cosine_sim", "rb" ) )
    cos_last = fed_metrics['cos_last']
    cos_avg_n = fed_metrics['cos_avg_n']
    ed_last = fed_metrics['ed_last']
    ed_avg_n = fed_metrics['ed_avg_n']
    fed_dates = fed_metrics['dates']

    #grouping by date (some dates had multiple speeches)
    avgstats = pd.DataFrame({'date':fed_dates,
                            'ed_last': ed_last,
                            'ed_avg_n': ed_avg_n,
                            'cos_last': cos_last,
                            'cos_avg_n': cos_avg_n}).groupby('date').mean()
    avgstats.index = pd.to_datetime(avgstats.index)

    X_fwds = X_fwds.merge(avgstats, how='left', left_index = True, right_index = True)
    X_fwds.fillna(value=0, inplace=True)

    # the first row of X_fwds contains zeros for the differenced rates, clear them here
    X_fwds = X_fwds.drop(X_fwds.index[0])

    # Creating training, cross-validation and test datasets
    total_obs = len(X_fwds)
    train_int = int(round(total_obs*.7, 0))
    cv_int = int(round(total_obs*.85, 0))

    fwd_train = X_fwds[0:train_int]
    fwd_cv = X_fwds[train_int:cv_int]
    fwd_test = X_fwds[cv_int:]

    forecast_matrix = np.zeros(shape=(len(fwd_cv), 7))
    # Base models to be estimated
    model_list= []
    ''' ARIMA model'''
    this_name = 'Normal ARIMA(1,1,1)'
    model_type = pf.ARIMA
    model_class = 'ARIMA'
    model_target= 'd_ten_y'
    hyper_params= {'ar':1, 'ma': 1, "diff_ord": 0}
    num_components = 1
    model_inputs = {'model_type': model_type,
                    'model_class': model_class,
                    'name': this_name,
                    'target': model_target,
                    'hyper_params': hyper_params,
                    'num_components': num_components,
                    'forecast': forecast_matrix}
    model_list.append(model_inputs)

    ''' ARMIAX model '''
    this_name = 'Normal ARIMAX(1,0,1)'
    model_type = pf.ARIMAX
    model_class = 'ARIMAX'
    model_target= 'd_ten_y'
    hyper_params= {'ar':1, 'ma': 1, "diff_ord": 0}
    num_components = 1
    model_inputs = {'model_type': model_type,
                    'model_class': model_class,
                    'name': this_name,
                    'target': model_target,
                    'hyper_params': hyper_params,
                    'num_components': num_components,
                    'formula':'d_ten_y~1+ed_last',
                    'forecast': forecast_matrix}
    model_list.append(model_inputs)

    ''' Gaussian Model '''
    this_name = 'Gaussian'
    model_class = 'Gaussian'
    model_target= 'd_ten_y'
    model_inputs = {'model_class': model_class,
                    'name': this_name,
                    'target': model_target,
                    'forecast': forecast_matrix}
    model_list.append(model_inputs)

    # create the list of column names to go over
    col_names = ['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']

    # now that we have the basic models, we need to run these models for every forward rate and
    # for all of the dates in the cv dataset

    #for i in col_names:
    i = 0

    # adjust the models to reflect this particular forward
    model_list[0]['target']=col_names[i] # ARIMA model
    model_list[1]['target']= col_names[i] # ARIMAX model
    model_list[2]['target']= col_names[i] # Gaussian
    # adjust the patsy function for the arimax model
    func_str = model_list[1]['formula']
    func_list = func_str.split(sep='~')
    model_list[1]['formula'] = col_names[i] + '~' + func_list[1]

    #initialize the models
    base_models = []
    for m in model_list:
        base_models.append(fc.ForecastModel(m))

    # start the loop for the dates
    for d in range(len(fwd_cv)):
        # updating the time series by one day
        X = update_cv_data(fwd_train, fwd_cv, d)
        print(d)

        for j, m in enumerate(model_list):

            this_model = base_models[j]
            this_model.fit(X)
            this_prediction = this_model.predict_one(X)
            if type(this_prediction)== np.float64:
                model_list[j]['forecast'][d,i]= this_prediction
            else:
                model_list[j]['forecast'][d,i] = this_prediction.iloc[0,0]
            print(this_prediction)









import ForecastModel as fc
# to relaod the foreacst model type below
reload(fc)

this_model = fc.ForecastModel(model_inputs)

# need to clear out the first row of the change in X
#fwd_train = fwd_train.drop(fwd_train.index[0])

#this_model.fit(fwd_train)
this_model.fit(fwd_train)
prediction = this_model.predict_one(fwd_train)

# create a list of the models

