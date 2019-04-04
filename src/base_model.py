''' Basic interest rate model

rt+1 = rt + alpha *dt + simga * sqrt(dt) * dx

    where
        rt+1 =  the rate at time t+1
        rt  =   the rate at time t
        alpha = drift (set equal to forward rate)
        dt =    1/260 (trading days)
        sigma = historical volatility of the rate
        dx    = standard normal noise

CALIBRATION:
    alpha = slope of the forward curve (fowards in X periods - current spot rate)
    sigma = historical std deviation of the spot rate (assumes this is constant over the peiord)

STEPS:
    1. Calculate slope of forward curve and adjust to be an annualized rate
        Do this in a separate dataframe (before split of training and cross valudation) after calculating the forwards
        Call it drift_df
    2. Calculate the std deviation of the entire series between test start and now
    3. Demonstrate the one period change (NOTE: THIS IS NOT REALLY A FACTOR FOR THE ONE PERIOD CHANGE)

    4. Demonstrate the expected value of interest rates with this model (no noise)
    5. Calculate difference from forecast and actual and store this data
    6. Create plots of these variables


'''

# LETS USE HIS FOR THE BASELINE MODEL
def df_add_first_diff(df):
    ' Adds the first differenced columns to the dataframe'
    diff_df = df.diff()
    df['d_six_m'] = diff_df['six_m']
    df['d_one_y'] = diff_df['one_y']
    df['d_two_y'] = diff_df['two_y']
    df['d_three_y'] = diff_df['three_y']
    df['d_five_y'] = diff_df['five_y']
    df['d_seven_y'] = diff_df['seven_y']
    df['d_ten_y'] = diff_df['ten_y']
    return df

''' Volatility Fitting '''

def get_matrix_column(mat,i):
    return np.array(mat[:,i].flatten())[0]

class PolynomialInterpolator:
    def __init__(self, params):
        assert type(params) == np.ndarray
        self.params = params
    def calc(self, x):
        n = len(self.params)
        C = self.params
        X = np.array([x**i for i in reversed(range(n))])
        return sum(np.multiply(X, C))

def fit_volatility(i, degree, title, fitted_vols, vols):
    vol = get_matrix_column(vols, i)
    fitted_vol = PolynomialInterpolator(np.polyfit(tenors, vol, degree))
    #plt.plot(tenors, vol, marker='.', label='Discretized volatility')
    #plt.plot(tenors, [fitted_vol.calc(x) for x in tenors], label='Fitted volatility')
    #plt.title(title)
    #plt.xlabel(r'Time $t$')
    #plt.legend()
    fitted_vols.append(fitted_vol)

def integrate(f, x0, x1, dx):
    n = (x1-x0)/dx+1
    out = 0
    for i, x in enumerate(np.linspace(x0, x1, n)):
        if i==0 or i==n-1:
            out += 0.5 * f(x)
        else:
            out += f(x)  # not adjusted by *0.5 because of repeating terms x1...xn-1 - see trapezoidal rule
    out *= dx
    return out

def m(tau, fitted_vols):
    #This funciton carries out integration for all principal factors.
    #It uses the fact that volatility is function of time in HJM model
    out = 0.
    for fitted_vol in fitted_vols:
        assert isinstance(fitted_vol, PolynomialInterpolator)
        out += integrate(fitted_vol.calc, 0, tau, 0.01) * fitted_vol.calc(tau)
    return out

def simulation(f, tenors, drift, vols, timeline):
    assert type(tenors)==np.ndarray
    assert type(f)==np.ndarray
    assert type(drift)==np.ndarray
    assert type(timeline)==np.ndarray
    assert len(f)==len(tenors)
    vols = np.array(vols.transpose())  # 3 rows, T columns
    len_tenors = len(tenors)
    len_vols = len(vols)
    yield timeline[0], copylib.copy(f)
    for it in range(1, len(timeline)):
        t = timeline[it]
        dt = t - timeline[it-1]
        sqrt_dt = np.sqrt(dt)
        fprev = f
        f = copylib.copy(f)
        #random_numbers = [np.random.normal() for i in range(len_vols)]
        for iT in range(len_tenors):
            val = fprev[iT] + drift[iT] * dt
            #
            sum = 0
            # Here we are assuming no shocks so not adjusting for vols
            # for iVol, vol in enumerate(vols):
            #     #sum += vol[iT] * random_numbers[iVol]
            #     sum += vol[iT] * random_numbers[iVol]
            # val += sum * sqrt_dt
            #
            iT1 = iT+1 if iT<len_tenors-1 else iT-1   # if we can't take right difference, take left difference
            dfdT = (fprev[iT1] - fprev[iT]) / (iT1 - iT)
            val += dfdT * dt
            #
            f[iT] = val
        yield t,f

def base_HJM_projection(df):
    ''' Project one day of a simulation for the yield curve'
    INPUT: df containing forward rates and change in forward rates
    '''
    sigma = np.cov(df[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']].transpose())
    sigma  *= 252

    eigval, eigvec = np.linalg.eig(sigma)
    eigvec = np.matrix(eigvec)
    #assert type(eigval)==np.ndarray
    #assert type(eigvec)==np.matrix
    #print(eigval)

    ''' find three largest principal components '''
    factors=3
    index_eigvec = list(reversed(eigval.argsort()))[0:factors]   # highest principal component first in the array
    princ_eigval =np.array([eigval[i] for i in index_eigvec])
    princ_comp = np.hstack([eigvec[:,i] for i in index_eigvec])
    #print("Principal eigenvalues")
    #print(princ_eigval)
    #print()
    #print("Principal eigenvectors")
    #print(princ_comp)
    #plt.plot(princ_comp, marker='.'),
    #plt.title('Principal components')
    #plt.xlabel(r'Time $t$');

    ''' Calculate discretized volatility function from principal components '''
    sqrt_eigval = np.matrix(princ_eigval ** .5)
    tmp_m = np.vstack([sqrt_eigval for i in range(princ_comp.shape[0])])  # resize matrix (1,factors) to (n, factors)
    vols = np.multiply(tmp_m, princ_comp) # multiply matrice element-wise
    #print('vols shape: ' + str(vols.shape))
    #plt.plot(vols, marker='.')
    #plt.xlabel(r'Time $t$')
    #plt.ylabel(r'Volatility $\sigma$')
    #plt.title('Discretized volatilities');

    # fitting the volatility functions as a polynomial
    fitted_vols = []
    fit_volatility(0, 3, '1st component', fitted_vols, vols)
    fit_volatility(1, 3, '2nd component', fitted_vols, vols)
    fit_volatility(2, 3, '3rd component', fitted_vols, vols)
    ''' NOTE: This fitting of a volatility function may be overkill for a one day simulation '''
    #plt.pubplot(1, 3, 1), fit_volatility(0, 3, '1st component');
    #plt.subplot(1, 3, 2), fit_volatility(1, 3, '2nd component');
    #plt.subplot(1, 3, 3), fit_volatility(2, 3, '3rd component');


    #mc_tenors = linspace(0,25,51)
    mc_tenors = np.array([0.5, 1, 2, 3, 5, 7, 10])
    # Discretize fitted volfuncs for the purpose of monte carlo simulation
    mc_vols = np.matrix([[fitted_vol.calc(tenor) for tenor in mc_tenors] for fitted_vol in fitted_vols]).transpose()
    #plt.plot(mc_tenors, mc_vols, marker='.')
    #plt.xlabel(r'Time $t$')
    #plt.title('Volatilities')
    # NOTE: Maybe just use base vols here from eigenvalues!


    ''' AT THIS POINT I DO NOT UNDERSTAND THE SHAPE OF MC_VOLS (7,9) '''
    mc_drift = np.array([m(tau, fitted_vols) for tau in mc_tenors])
    #plt.plot(mc_drift, marker='.')
    #plt.xlabel(r'Time $t$')
    #plt.title('Risk-neutral drift');
    #plt.show()


    ''' QUESTION: Does this drift include all principal components?'''
    hist_rates =np.matrix(fwd_cv[['six_m', 'one_y', 'two_y', 'three_y', 'five_y', 'seven_y','ten_y']])
    curve_spot = np.array(hist_rates[-1,:].flatten())[0]
    # plt.plot(mc_tenors, curve_spot.transpose(), marker='.')
    # plt.ylabel('$f(t_0,T)$')
    # plt.xlabel("$T$");


    proj_rates = []
    proj_timeline = np.linspace(0,1/260,2)
    for i, (t, f) in enumerate(simulation(curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline)):
        #progressbar.update(i)
        proj_rates.append(f)

    proj_rates = np.matrix(proj_rates)
    #new_rates = proj_rates[-1,:0]
    #return new_rates
    return proj_rates

if __name__ == __main__:

import copy as copylib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyflux as pf
import datetime as datetime
import matplotlib.pyplot as plt

import os
import pickle



#os.chdir('..')
X = pickle.load(open("../data/interest_rate_data", "rb" ))
X_fwds = pickle.load(open('../data/forward_rates', 'rb'))
X_zeros = pickle.load(open('../data/zero_rates', 'rb'))

# First difference the time series data for stationarity
X = df_add_first_diff(X)
X_fwds = df_add_first_diff(X_fwds)
X_zeros = df_add_first_diff(X_zeros)


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

X = X.merge(avgstats, how='left', left_index = True, right_index = True)
X_fwds = X_fwds.merge(avgstats, how='left', left_index = True, right_index = True)
X_zeros = X_zeros.merge(avgstats, how = 'left', left_index = True, right_index = True)

X.fillna(value=0, inplace=True)
X_fwds.fillna(value=0, inplace=True)
X_zeros.fillna(value=0, inplace=True)

# cannot use train/test split on this because it is time series
total_obs = len(X)
train_int = int(round(total_obs*.7, 0))
cv_int = int(round(total_obs*.85, 0))

fwd_train = X_fwds[0:train_int]
fwd_cv = X_fwds[train_int:cv_int]
fwd_test = X_fwds[cv_int:]

zero_train = X_zeros[0:train_int]
zero_cv = X_zeros[train_int:cv_int]
zero_test = X_zeros[cv_int:]

X_train = X[0:train_int]
X_cv = X[train_int:cv_int]
X_test = X[cv_int:]


tenors = [0.5, 1, 2, 3, 5, 7, 10]
mc_tenors = np.linspace(0,25,51)

# Looping through the CV data to create cross val data sets
fcst_array = np.zeros(shape = (len(fwd_cv),len(tenors)))
for i in range(len(fwd_cv)):
    fwd_train = fwd_train.append(fwd_cv.iloc[i])
    #print(fwd_train.tail())

    fcst_array[i,:] = base_HJM_projection(fwd_train)[1,:]

# now we have the fcst_array and can use this to compare to
# the actual change in fwds
    actual_array = fwd_cv[['d_six_m', 'd_one_y', 'd_two_y',
        'd_three_y', 'd_five_y', 'd_seven_y', "d_ten_y"]].as_matrix()

    fcst_error = fcst_array - actual_array