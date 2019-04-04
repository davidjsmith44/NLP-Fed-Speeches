''' testing hjm '''



import copy as copylib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
# put this into a function later
tenors = [0.5, 1, 2, 3, 5, 7, 10]
test = PCA(n_components = 7, random_state=0)
test.fit(X_fwds[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']])

print("Explained variance of first pc: {0:2.2f}".format(test.explained_variance_ratio_[0]))
print("Explained variance of second pc: {0:2.2f}".format(test.explained_variance_ratio_[1]))
print("Explained variance of third pc: {0:2.2f}".format(test.explained_variance_ratio_[2]))
print("Explained variance of forth pc: {0:2.2f}".format(test.explained_variance_ratio_[3]))
print("Explained variance of fith pc: {0:2.2f}".format(test.explained_variance_ratio_[4]))
#test.explained_variance_ratio_

# This is how I recover the shocks
shocks = test.fit_transform(X_fwds[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']])

# LETS USE HIS FOR THE BASELINE MODEL
sigma = np.cov(X_fwds[['d_six_m', 'd_one_y', 'd_two_y', 'd_three_y', 'd_five_y', 'd_seven_y', 'd_ten_y']].transpose())
print("sigma shape :", str(sigma.shape))

sigma  *= 252

eigval, eigvec = np.linalg.eig(sigma)
eigvec = np.matrix(eigvec)
assert type(eigval)==np.ndarray
assert type(eigvec)==np.matrix
print(eigval)

''' find three largest principal components '''
factors=3
index_eigvec = list(reversed(eigval.argsort()))[0:factors]   # highest principal component first in the array
princ_eigval =np.array([eigval[i] for i in index_eigvec])
princ_comp = np.hstack([eigvec[:,i] for i in index_eigvec])
print("Principal eigenvalues")
print(princ_eigval)
print()
print("Principal eigenvectors")
print(princ_comp)
plt.plot(princ_comp, marker='.'),
plt.title('Principal components')
plt.xlabel(r'Time $t$');

''' Calculate discretized volatility function from principal components '''
sqrt_eigval = np.matrix(princ_eigval ** .5)
tmp_m = np.vstack([sqrt_eigval for i in range(princ_comp.shape[0])])  # resize matrix (1,factors) to (n, factors)
vols = np.multiply(tmp_m, princ_comp) # multiply matrice element-wise
print('vols shape: ' + str(vols.shape))
plt.plot(vols, marker='.')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Volatility $\sigma$')
plt.title('Discretized volatilities');

''' Volatility Fitting '''
def get_matrix_column(mat, i):
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

fitted_vols = []
def fit_volatility(i, degree, title):
    vol = get_matrix_column(vols, i)
    fitted_vol = PolynomialInterpolator(np.polyfit(tenors, vol, degree))
    plt.plot(tenors, vol, marker='.', label='Discretized volatility')
    plt.plot(tenors, [fitted_vol.calc(x) for x in tenors], label='Fitted volatility')
    plt.title(title)
    plt.xlabel(r'Time $t$')
    plt.legend()
    fitted_vols.append(fitted_vol)

plt.subplot(1, 3, 1), fit_volatility(0, 3, '1st component');
plt.subplot(1, 3, 2), fit_volatility(1, 3, '2nd component');
plt.subplot(1, 3, 3), fit_volatility(2, 3, '3rd component');

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

#mc_tenors = linspace(0,25,51)
mc_tenors = np.array([0.5, 1, 2, 3, 5, 7, 10])
# Discretize fitted volfuncs for the purpose of monte carlo simulation
mc_vols = np.matrix([[fitted_vol.calc(tenor) for tenor in mc_tenors] for fitted_vol in fitted_vols]).transpose()
plt.plot(mc_tenors, mc_vols, marker='.')
plt.xlabel(r'Time $t$')
plt.title('Volatilities')


''' AT THIS POINT I DO NOT UNDERSTAND THE SHAPE OF MC_VOLS (7,9) '''
def m(tau, fitted_vols):
    #This funciton carries out integration for all principal factors.
    #It uses the fact that volatility is function of time in HJM model
    out = 0.
    for fitted_vol in fitted_vols:
        assert isinstance(fitted_vol, PolynomialInterpolator)
        out += integrate(fitted_vol.calc, 0, tau, 0.01) * fitted_vol.calc(tau)
    return out

mc_drift = np.array([m(tau, fitted_vols) for tau in mc_tenors])
plt.plot(mc_drift, marker='.')
plt.xlabel(r'Time $t$')
plt.title('Risk-neutral drift');
plt.show()

# Somewhere here we define historical rates???

''' QUESTION: Does this drift include all principal components?'''
hist_rates =np.matrix(X_fwds[['six_m', 'one_y', 'two_y', 'three_y', 'five_y', 'seven_y','ten_y']])
curve_spot = np.array(hist_rates[-1,:].flatten())[0]
plt.plot(mc_tenors, curve_spot.transpose(), marker='.')
plt.ylabel('$f(t_0,T)$')
plt.xlabel("$T$");

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
        random_numbers = [np.random.normal() for i in range(len_vols)]
        for iT in range(len_tenors):
            val = fprev[iT] + drift[iT] * dt
            #
            sum = 0
            for iVol, vol in enumerate(vols):
                #sum += vol[iT] * random_numbers[iVol]
                sum += vol[iT] * random_numbers[iVol]
            val += sum * sqrt_dt
            #
            iT1 = iT+1 if iT<len_tenors-1 else iT-1   # if we can't take right difference, take left difference
            dfdT = (fprev[iT1] - fprev[iT]) / (iT1 - iT)
            val += dfdT * dt
            #
            f[iT] = val
        yield t,f

proj_rates = []
proj_timeline = np.linspace(0,1, 260)
#progressbar = ProgressBar("One simulation path", len(proj_timeline))
for i, (t, f) in enumerate(simulation(curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline)):
    #progressbar.update(i)
    proj_rates.append(f)
proj_rates = np.matrix(proj_rates)
plt.plot(proj_timeline.transpose(), proj_rates)
plt.xlabel(r'Time $t$')
plt.ylabel(r'Rate $f(t,\tau)$');
plt.title(r'Simulated $f(t,\tau)$ by $t$')
plt.show()
plt.plot(mc_tenors, proj_rates.transpose())
plt.xlabel(r'Tenor $\tau$')
plt.ylabel(r'Rate $f(t,\tau)$')
plt.title(r'Simulated $f(t,\tau)$ by $\tau$')
plt.show()
