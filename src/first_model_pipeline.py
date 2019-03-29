
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
from sklearn.decomposition import PCA



def plot_3_pcs(comps, shocks, filename=None):
    ''' TO DO ON THIS PLOT
        1. make the X axis for charts on right the date
        2. optional title with start and end dates
    '''
    fig, axs = plt.subplots(3,2, figsize = (12,12))

    axs[0,0].plot(comps[0, :], color='k')
    axs[0,0].grid()
    axs[0,0].set_facecolor("whitesmoke")
    axs[0,0].set_title("First Principal Component Vector")
    axs[0,1].plot(shocks[:,0], color='r')
    axs[0,1].grid()
    axs[0,1].set_facecolor("whitesmoke")
    axs[0,1].set_title("Shocks due to 1st Component")

    axs[1,0].plot(comps[1, :], color='k')
    axs[1,0].grid()
    axs[1,0].set_facecolor("whitesmoke")
    axs[1,0].set_title("Second Principal Component Vector")
    axs[1,1].plot(shocks[:,1], color='b')
    axs[1,1].grid()
    axs[1,1].set_facecolor("whitesmoke")
    axs[1,1].set_title("Shocks due to 2nd Component")

    axs[2,0].plot(comps[2, :], color='k')
    axs[2,0].grid()
    axs[2,0].set_facecolor("whitesmoke")
    axs[2,0].set_title("Third Principal Component Vector")
    axs[2,1].plot(shocks[:,2], color='g')
    axs[2,1].grid()
    axs[2,1].set_facecolor("whitesmoke")
    axs[2,1].set_title("Shocks due to 3rd Component")

    fig.suptitle("Principal Components of the US Treasury Yield Curve", fontsize = 30, color = 'k')
    fig.show()

    if filename:
        plt.savefig(filename)



def sample_from_data(X, start_date, end_date):
    start_date = '2010-01-01'
    end_date ='2016-01-01'

    return  X[ (X.index >= start_date) & (X.index < '2010-12-30')]




def run_model_on_sample(X_samp, n_comp, random_state=44):
    ''' take a sample and runs a principal component analysis
    on the sample
    RETURNS a dictionary with component_vectors,
                              percent_variance_explained
                              shocks
    '''

    pca = PCA(n_components = n_comp, random_state = random_state)
    pca.fit(X_samp)


    # recovering the components (vectors)
    component_vectors = pca.components_
    pct_var_explained = pca.explained_variance_ratio_
    shocks            = pca.fit_transform(X)

    return_dict = {'component_vectors': component_vectors,
                    'pct_var_explained': pct_var_explained,
                    'shocks': shocks}
    return return_dict


df_treas = quandl.get("USTREASURY/YIELD", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")
df_treas['3 MO'] = df_treas['3 MO'].fillna(0.0001)

# now creating the X features to be run through PCA
X = df_treas.copy()
X = X.drop(['1 MO', '2 MO', '20 YR', '30 YR'], axis=1)


# create initial sample of the model
# need to filter the data here
start_date = '2010-01-01'
end_date = '2016-01-01'
X_samp = sample_from_data(X, start_date, end_date)

# now working on the PCA model
n_comp = 5
#test = PCA(n_components = n_comp, random_state=44)



#test.fit(X)
#print("Explained variance: ", test.explained_variance_ratio_)

#plt_x = np.arange(0, n_comp)
#plt.bar(plt_x, test.explained_variance_ratio_)
#plt.show()


pca_dict = run_model_on_sample(X_samp, 5)
f_name= '../principal_components.png'
#plot_3_pcs(pca_dict['component_vectors'], pca_dict['shocks'])
plot_3_pcs(pca_dict['component_vectors'], pca_dict['shocks'], f_name)

# NOW LETS DO AUTOCORRELATION STUFF ON THE HISTORICAL SHOCKS
# start looking at the shocks
import statsmodels as sm
# test = sm.tsa.stattools.adfuller(series)

def series_and_lagged(series, lag=1):
    truncated = np.copy(series)[lag:]
    lagged = np.copy(series)[:(len(truncated))]
    return truncated, lagged

def compute_autocorrelation(series, lag=1):
    series, lagged = series_and_lagged(series, lag=lag)
    return np.corrcoef(series, lagged)[0, 1]

def plot_series_and_lagged(axs, series, title, lag=1):
    series.iloc[:-lag].plot(ax=axs[0], marker='.')
    series.iloc[lag:].plot(ax=axs[1], marker='.')
    axs[0].set_title(title)
    axs[1].set_title(title + ' lagged by {}'.format(lag))

fig, axs = plt.subplots(2, figsize=(14, 4))
plot_series_and_difference(axs, google_trends['baseball'], 'baseball')
fig.tight_layout()