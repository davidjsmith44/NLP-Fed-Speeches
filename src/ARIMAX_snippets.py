''' ARMIMAX and EDA snippets '''

''' First ARIMAX model '''
#NOTE: We need to store the formula for the ARIMA model in the hyper_params
this_name = 'Normal ARIMAX(1,1,1)'
hyper_params = {'ar': 1, 'ma':1, 'diff_ord':1, 'target': 'ten_y',
                'formula':'ten_y~1+ed_last'}
forecast = 0
model_inputs = {'model_type': 'ARIMAX',
                'name': this_name,
                'target_class': 'rates',
                'hyper_params': hyper_params,
                'foreacst': forecast}
model_list.append(model_inputs)

### trying like they have in the documentation
model = pf.ARIMAX(data = X, formula = 'ten_y~1+ed_last',
        ar=1, ma=1, integ=1, family=pf.Normal())

test_fit = model.fit("MLE")
test_fit.summary()