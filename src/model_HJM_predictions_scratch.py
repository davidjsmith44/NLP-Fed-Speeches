''' Working on redoing the base_HJM_predictions function to incorporate my ARIMAX model

NEED TO MAKE THE FOLLOWING CHANGES
1. Recreate historical shocks to the term structure for this dataset
2. Fit ARIMAX model on the historical shocks to the first principal component shocks
3. Predict one period forward change in the shock
4. Feed this one period shock back into the HJM simulation function
5. Return the new data series

'''
def model_HJM_projection(df, model_dict):
    ''' Project one day of a simulation for the yield curve'
    INPUT:
        df          containing forward rates, change in forward rates and Fed speeches
        model_dict  dictionary with all of the terms of this model

    '''
    # BEGIN: below is the old code for PCA
    X = df[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']]
    # need to drop the first column of the dataframe since it contains blanks for the change in rates
    X=X.iloc[1:]
    n_comp = 3
    test = PCA(n_components = n_comp, random_state=44)

    test.fit(X)
    shocks = test.fit_transform(X)

    # HERE WE NEED TO DO A MODEL ON THE SHOCKS
    # we are going to add the shocks to the dataframe
    df['shock1']= 0
    df['shock2']= 0
    df['shock3']= 0

    df['shock1'] = shocks[:,0]
    df['shock2'] = shocks[:,1]
    df['shock3'] = shocks[:,2]

    #import pyflux as pf
    model = pf.ARIMAX(data = df.iloc[:-1], formula = 'shock1~1+ed_last+shock2+shock3',
            ar=4, ma=5, integ=0, family=pf.Normal())

    test_fit = model.fit("MLE")
    test_fit.summary()
    test_fit = model.fit("MLE")
    test_fit.summary()
    oos_data = df.iloc[-1]
    oos_data['ed_last'] = 0
    oos_data['ed_avg_n'] = 0
    oos_data['cos_last'] = 0
    oos_data['cos_avg_n'] = 0
    oos_data['shock1']= 0
    oos_data['shock2']= 0
    oos_data['shock3']= 0

    prediction = model.predict(h = 10, oos_data, intervals=False)

    # END: old code for PCA

    # NOTE: The PCA explained variance is different than the np.linalg.eig
    # To address this, we are going to do both. I need to return the shocks to the changes in rates for the
    # ARIMAX model - so I am going to use PCA to get the historical shocks and do my estimation and then return
    # to the np.linalg.eig method to get the volatility

    # END: code for PCA
    sigma = np.cov(X.transpose())
    sigma  *= 252

    eigval, eigvec = np.linalg.eig(sigma)
    eigvec = np.matrix(eigvec)
    assert type(eigval)==np.ndarray
    assert type(eigvec)==np.matrix

    ''' find three largest principal components '''
    factors=3
    index_eigvec = list(reversed(eigval.argsort()))[0:factors]   # highest principal component first in the array
    princ_eigval =np.array([eigval[i] for i in index_eigvec])
    princ_comp = np.hstack([eigvec[:,i] for i in index_eigvec])

    ''' Calculate discretized volatility function from principal components '''
    sqrt_eigval = np.matrix(princ_eigval ** .5)
    tmp_m = np.vstack([sqrt_eigval for i in range(princ_comp.shape[0])])  # resize matrix (1,factors) to (n, factors)
    vols = np.multiply(tmp_m, princ_comp) # multiply matrice element-wise

    # fitting the volatility functions as a polynomial
    fitted_vols = []
    fit_volatility(0, 3, '1st component', fitted_vols, vols)
    fit_volatility(1, 3, '2nd component', fitted_vols, vols)
    fit_volatility(2, 3, '3rd component', fitted_vols, vols)
    ''' NOTE: This fitting of a volatility function may be overkill for a one day simulation '''

    mc_tenors = np.array([0.5, 1, 2, 3, 5, 7, 10])
    # Discretize fitted volfuncs for the purpose of monte carlo simulation
    mc_vols = np.matrix([[fitted_vol.calc(tenor) for tenor in mc_tenors] for fitted_vol in fitted_vols]).transpose()


    ''' AT THIS POINT I DO NOT UNDERSTAND THE SHAPE OF MC_VOLS (7,9) '''
    mc_drift = np.array([m(tau, fitted_vols) for tau in mc_tenors])

    hist_rates =np.matrix(df[['six_m', 'one_y', 'two_y', 'three_y', 'five_y', 'seven_y','ten_y']])
    curve_spot = np.array(hist_rates[-1,:].flatten())[0]


    proj_rates = []
    proj_timeline = np.linspace(0,1/260,2)
    for i, (t, f) in enumerate(simulation(curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline)):
        #progressbar.update(i)
        proj_rates.append(f)

    proj_rates = np.matrix(proj_rates)
    #new_rates = proj_rates[-1,:0]
    #return new_rates
    return proj_rates



