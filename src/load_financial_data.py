'''
This file downloads data from quandl.com and places
it into a datafile

DATA INCLUDED
    US Treasury Yields
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import quandl


df_treas = quandl.get("USTREASURY/YIELD", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")

# daily EUR_USD

df_EUR_USD = quandl.get("FED/RXI_US_N_B_EU", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")
# daily GBP / USD
df_GBP = quandl.get("FED/RXI_US_N_B_UK", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")
# daily USD/JPY
df_JPY = quandl.get("FED/RXI_N_B_JA", authtoken="zBYbsY7fujcHokgXQdsY",
    start_date = "2006-01-01", end_date="2019-03-28")

''' NOW I NEED TO MESS WITH THE DATES '''

# The 3 month Treasury has three dates at the end of 2008
# where the rate is zero. Pandas is treating this as Nan
# I am replacing these with zero, because they are zeros.
df_treas['3 MO'] = df_treas['3 MO'].fillna(0.0001)

# now creating the X features to be run through PCA
X = df_treas.copy()

X = X.drop(['1 MO', '2 MO', '20 YR', '30 YR'], axis=1)

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