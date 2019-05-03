'''
This file contains the list of parameters that will be run through
the interest rate simulation model.

The models are stored as elements in a list

Each element is a dictionary containing the model information
and the hyperparameters to implement.

Gaussian - mean change in the dataset is the expected change
ARIMA
ARIMAX


'''
import pyflux as pf
model_list = []
if __name__ == '__main__':
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
                    'num_components': num_components
                    }

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
                    'formula':'d_ten_y~1+ed_last'}

    model_list.append(model_inputs)

    ''' Gaussian Model '''

    this_name = 'Gaussian'
    model_class = 'Gaussian'
    model_target= 'd_ten_y'
    model_inputs = {'model_class': model_class,
                    'name': this_name,
                    'target': model_target}

    model_list.append(model_inputs)