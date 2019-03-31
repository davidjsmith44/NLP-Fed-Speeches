''' LOAD_FINANCIAL_DATA.PY

This file
    1. downloads interest rate and FX data from quandl.com
    2. Cleans the interest rate data and places it into a dataframe X
        X columns = ['3 MO', '6 MO', etc]
    3. The intererst rate data is saved in a pickle file called 'interest_rate_data.p' in the data directory
    4. Downloads FX data for EUR, GBP, JPY and places that data into a dataframe called FX
        FX columns = ['EUR', 'GBP', 'JPY']
        NOTE: No preprocessing has been done on this file yet!
    5. The FX data is saved in a pickle file called 'FX_data.p'

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import quandl
import pickle
import os

df_treas = quandl.get("USTREASURY/YIELD", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")


# The 3 month Treasury has three dates at the end of 2008
# where the rate is zero. Pandas is treating this as Nan
# I am replacing these with zero, because they are zeros.
df_treas['3 MO'] = df_treas['3 MO'].fillna(0.0001)

# now creating the X features to be run through PCA
X = df_treas.copy()

X = X.drop(['1 MO', '2 MO', '20 YR', '30 YR'], axis=1)

# saving the df to a pickle file
pickle_out = open('../data/interest_rate_data', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()


''' Saving this for later if we need it '''
# daily EUR_USD
df_EUR = quandl.get("FED/RXI_US_N_B_EU", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")

# daily GBP / USD
df_GBP = quandl.get("FED/RXI_US_N_B_UK", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")

# daily USD/JPY
df_JPY = quandl.get("FED/RXI_N_B_JA", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")

df_FX = df_EUR
df_FX.rename(index=str, columns={"Value": "EUR"})
df_FX['GBP'] = df_GBP['Value']
df_FX['JPY'] = df_JPY['Value']

pickle_out = open('../data/FX_data', 'wb')
pickle.dump(df_FX, pickle_out)
pickle_out.close()



'''
n_comp = 5
test = PCA(n_components = n_comp, random_state=44)

test.fit(X)
print("Explained variance: ", test.explained_variance_ratio_)

plt_x = np.arange(0, n_comp)
plt.bar(plt_x, test.explained_variance_ratio_)
plt.show()

# recovering the components (vectors)
comp_vects = test.components_


shocks = test.fit_transform(X)


def plot_3_pcs(comps, shocks):
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

'''
''' To Do list
1. make the PCA analysis more of a workflow
    -select dates to use
        start_date
        end_date
    - fit the PCA model
    - recover all of the relevant information
    - recover the shocks
    - perform autocorrelations on the data

2. HOW DO I BUILD THE FINAL MODEL???


'''