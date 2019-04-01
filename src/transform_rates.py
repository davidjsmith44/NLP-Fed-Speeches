''' Transforming interest rate data into zeros and forwards '''



# Assume we have the entire X datafame in memory

#Load up interest rate data
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt

import os
import pickle
from src.bootstrap import BootstrapYieldCurve

os.chdir('..')
X = pickle.load(open("data/interest_rate_data", "rb" ) )

X = X/100
zeros = X[['3 MO', '6 MO', '1 YR']].copy()

'''
STEPS
1. Calculate zero coupon bond prices for the discount securities
    -3 MO,
    -6 MO
    -1 YR
2. Create loop to build spot rates

3. Calculate forwards from the spot rates (or redo zero coupon bond prices)



'''

def zero_coupon_bond_price(par, ytm, time):
    ''' Takes the par price, ytm and time to maturity and returns the spot price of the bond'''
    return par / (1 + ytm/2) ** (time*2)


''' calculating the forward rates for the two year '''
'''
The bootstrapping class requires us to add instruments with the following
    par
    T (term)
    coup
    price
'''
price = 100
par = 100

bond_prices_3M = zero_coupon_bond_price(par, X['3 MO'], time=0.25)
bond_prices_6M = zero_coupon_bond_price(par, X['6 MO'], time=0.5)
bond_prices_1YR = zero_coupon_bond_price(par, X['1 YR'], time=1.0)

'''
X['P3M'] = bond_prices_3M
X['P6M'] = bond_prices_6M
X['P1YR'] = bond_prices_1YR
'''

# NOW USING LOOP TO BUILD SPOT RATES ONE AT A TIME
# 2 year bond prices

# start by working on one

yc = BootstrapYieldCurve()
# NOTE the fields are par, term, coupon and price
i = 0
yc.add_instrument(par, 0.25, 0, bond_prices_3M[0])
yc.add_instrument(par, 0.5, 0, bond_prices_6M[0])
yc.add_instrument(par, 1.0, 0, bond_prices_1YR[0])
# now adding the two year
yc.add_instrument(100, 2.0, X['2 YR'].iloc[i],100)
yc.add_instrument(100, 3.0, X['3 YR'].iloc[i],100)
yc.add_instrument(100, 5.0, X['5 YR'].iloc[i],100)
yc.add_instrument(100, 7.0, X['7 YR'].iloc[i],100)
yc.add_instrument(100, 10.0, X['10 YR'].iloc[i],100)


y = yc.get_zero_rates()
x = yc.get_maturities()

#import math

'''
yield_curve = BootstrapYieldCurve()
yield_curve.add_instrument(100, 0.25, 0., 97.5)
yield_curve.add_instrument(100, 0.5, 0., 94.9)
yield_curve.add_instrument(100, 1.0, 0., 90.)
yield_curve.add_instrument(100, 1.5, 8, 96., 2)
yield_curve.add_instrument(100, 2., 12, 101.6, 2)
'''


y = yc.get_zero_rates()
x = yc.get_maturities()



''' BUILDING MY OWN FUNCTION TO SOLVE FOR BOOTSTRAPPING '''
# 2 year
def bootstrap_2yr(X):
    spots = np.zeros(shape=(len(X), 1))
    par = 100

    for i in range(len(X)):
        rate = 0
        r6 = X['6 MO'].iloc[i]
        r12 = X['1 YR'].iloc[i]
        cpn = X['2 YR'].iloc[i] * par/2
        while True:
            rate += 0.0001
            delta = 100 - (cpn/((1+(r6/2))**1)) \
                    - (cpn/((1+(r12/2))**2)) \
                    - (cpn/((1+(rate / 2))**3)) \
                    - (cpn + par)/((1 + (rate / 2))**4)

            if delta >= 0:
                break
        spots[i] = rate
    return spots

def bootstrap_3yr(X, X_zeros):
    spots = np.zeros(shape=(len(X), 1))
    par = 100

    for i in range(len(X)):
        rate = 0
        r6 = X['6 MO'].iloc[i]
        r12 = X['1 YR'].iloc[i]
        r18 = X_zeros['2 YR'].iloc[i]
        r24 = X_zeros['2 YR'].iloc[i]
        cpn = X['3 YR'].iloc[i] * par/2
        while True:
            rate += 0.0001
            delta = 100 - (cpn/((1+(r6/2))**1)) \
                    - (cpn/((1+(r12/2))**2)) \
                    - (cpn/((1+(r18 / 2))**3)) \
                    - (cpn/((1+(r24 / 2))**4)) \
                    - (cpn/((1+(rate / 2))**5)) \
                    - (cpn + par)/((1 + (rate / 2))**6)
            if delta >= 0:
                break
        spots[i] = rate
    return spots

def bootstrap_5yr(X, X_zeros):
    spots = np.zeros(shape=(len(X), 1))
    par = 100

    for i in range(len(X)):
        rate = 0
        r6 = X['6 MO'].iloc[i]
        r12 = X['1 YR'].iloc[i]
        r18 = X_zeros['2 YR'].iloc[i]
        r24 = X_zeros['2 YR'].iloc[i]
        r30 = X_zeros['3 YR'].iloc[i]
        r36 = X_zeros['3 YR'].iloc[i]

        cpn = X['5 YR'].iloc[i] * par/2
        while True:
            rate += 0.00001
            delta = 100 - (cpn/((1+(r6/2))**1)) \
                    - (cpn/((1+(r12/2))**2)) \
                    - (cpn/((1+(r18 / 2))**3)) \
                    - (cpn/((1+(r24 / 2))**4)) \
                    - (cpn/((1+(r30 / 2))**5)) \
                    - (cpn/((1+(r36 / 2))**6)) \
                    - (cpn/((1+(rate / 2))**7)) \
                    - (cpn/((1+(rate / 2))**8)) \
                    - (cpn/((1+(rate / 2))**9)) \
                    - (cpn + par)/((1 + (rate / 2))**10)
            if delta >= 0:
                break
        spots[i] = rate
    return spots

def bootstrap_7yr(X, X_zeros):
    spots = np.zeros(shape=(len(X), 1))
    par = 100

    for i in range(len(X)):
        rate = 0
        r6 = X['6 MO'].iloc[i]
        r12 = X['1 YR'].iloc[i]
        r18 = X_zeros['2 YR'].iloc[i]
        r24 = X_zeros['2 YR'].iloc[i]
        r30 = X_zeros['3 YR'].iloc[i]
        r36 = X_zeros['3 YR'].iloc[i]
        r42 = X_zeros['5 YR'].iloc[i]
        r48 = X_zeros['5 YR'].iloc[i]
        r54 = X_zeros['5 YR'].iloc[i]
        r60 = X_zeros['5 YR'].iloc[i]

        cpn = X['7 YR'].iloc[i] * par/2
        while True:
            rate += 0.00001
            delta = 100 - (cpn/((1+(r6/2))**1)) \
                    - (cpn/((1+(r12/2))**2)) \
                    - (cpn/((1+(r18 / 2))**3)) \
                    - (cpn/((1+(r24 / 2))**4)) \
                    - (cpn/((1+(r30 / 2))**5)) \
                    - (cpn/((1+(r36 / 2))**6)) \
                    - (cpn/((1+(r42 / 2))**7)) \
                    - (cpn/((1+(r48 / 2))**8)) \
                    - (cpn/((1+(r54 / 2))**9)) \
                    - (cpn/((1+(r60 / 2))**10)) \
                    - (cpn/((1+(rate / 2))**11)) \
                    - (cpn/((1+(rate / 2))**12)) \
                    - (cpn/((1+(rate / 2))**13)) \
                    - (cpn + par)/((1 + (rate / 2))**14)
            if delta >= 0:
                break
        spots[i] = rate
    return spots

def bootstrap_10yr(X, X_zeros):
    spots = np.zeros(shape=(len(X), 1))
    par = 100

    for i in range(len(X)):
        rate = 0
        r6 = X['6 MO'].iloc[i]
        r12 = X['1 YR'].iloc[i]
        r18 = X_zeros['2 YR'].iloc[i]
        r24 = X_zeros['2 YR'].iloc[i]
        r30 = X_zeros['3 YR'].iloc[i]
        r36 = X_zeros['3 YR'].iloc[i]
        r42 = X_zeros['5 YR'].iloc[i]
        r48 = X_zeros['5 YR'].iloc[i]
        r54 = X_zeros['5 YR'].iloc[i]
        r60 = X_zeros['5 YR'].iloc[i]
        r66 = X_zeros['7 YR'].iloc[i]
        r72 = X_zeros['7 YR'].iloc[i]
        r78 = X_zeros['7 YR'].iloc[i]
        r84 = X_zeros['7 YR'].iloc[i]

        cpn = X['10 YR'].iloc[i] * par/2
        while True:
            rate += 0.00001
            delta = 100 - (cpn/((1+(r6/2))**1)) \
                    - (cpn/((1+(r12/2))**2)) \
                    - (cpn/((1+(r18 / 2))**3)) \
                    - (cpn/((1+(r24 / 2))**4)) \
                    - (cpn/((1+(r30 / 2))**5)) \
                    - (cpn/((1+(r36 / 2))**6)) \
                    - (cpn/((1+(r42 / 2))**7)) \
                    - (cpn/((1+(r48 / 2))**8)) \
                    - (cpn/((1+(r54 / 2))**9)) \
                    - (cpn/((1+(r60 / 2))**10)) \
                    - (cpn/((1+(r66 / 2))**11)) \
                    - (cpn/((1+(r72 / 2))**12)) \
                    - (cpn/((1+(r78 / 2))**13)) \
                    - (cpn/((1+(r84 / 2))**14)) \
                    - (cpn/((1+(rate / 2))**15)) \
                    - (cpn/((1+(rate / 2))**16)) \
                    - (cpn/((1+(rate / 2))**17)) \
                    - (cpn/((1+(rate / 2))**18)) \
                    - (cpn/((1+(rate / 2))**19)) \
                    - (cpn + par)/((1 + (rate / 2))**20)
            if delta >= 0:
                break
        spots[i] = rate
    return spots


X_zeros = X[['3 MO', '6 MO', '1 YR']].copy()
zeros_2yr = bootstrap_2yr(X)
X_zeros['2 YR'] = zeros_2yr

zeros_3yr = bootstrap_3yr(X, X_zeros)
X_zeros['3 YR'] = zeros_3yr

zeros_5yr = bootstrap_5yr(X, X_zeros)
X_zeros['5 YR'] = zeros_5yr

zeros_7yr = bootstrap_7yr(X, X_zeros)
X_zeros['7 YR'] = zeros_7yr

zeros_10yr = bootstrap_10yr(X, X_zeros)
X_zeros['10 YR'] = zeros_10yr


# saving the df to a pickle file
pickle_out = open('data/zero_rates', 'wb')
pickle.dump(X_zeros, pickle_out)
pickle_out.close()

''' now working on forward rates
START BY IGNORING THE 3M RATE
 we will have annualized forward rates fro
    6 mo
    6-12 mo
    1-2 yrs
    2-3 yrs
    3-5 yers
    5-7 yrs
    7-10 yrs
Need to express these as annualized rates

#then take logs
'''
z_pr_6m  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['6 MO'],
                                    time= 0.5)

z_pr_1y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['1 YR'],
                                    time= 1.0)

z_pr_2y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['2 YR'],
                                    time= 2.0)

z_pr_3y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['3 YR'],
                                    time= 3.0)

z_pr_5y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['5 YR'],
                                    time= 5.0)

z_pr_7y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['7 YR'],
                                    time= 7.0)

z_pr_10y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['10 YR'],
                                    time= 10.0)

fwd_6_12 = z_pr_6m / z_pr_1y
fwd_1_2= z_pr_1y/z_pr_2y
fwd_2_3 =z_pr_2y/z_pr_3y
fwd_3_5 =z_pr_3y/z_pr_5y
fwd_5_7 =z_pr_5y/z_pr_7y
fwd_7_10=z_pr_7y/z_pr_10y

# nowwe need to account for periods that are two years and adjust by
# taking the sqare root
fwd_3_5 =np.sqrt(fwd_3_5)
fwd_5_7 =np.sqrt(fwd_5_7)
fwd_7_10=fwd_7_10**(1/3)

# annualize the first ????
X_fwds = X_zeros[['6 MO', '1 YR']].copy()
X_fwds['2 YR'] = fwd_1_2 -1
X_fwds['3 YR'] = fwd_2_3 -1
X_fwds['5 YR'] = fwd_3_5 -1
X_fwds['7 YR'] = fwd_5_7 -1
X_fwds['10 YR'] = fwd_7_10 -1

# take care of the 6 month rates to annualize
#X_fwds['1 YR'] = fwd_6_12

pickle_out = open('data/forward_rates', 'wb')
pickle.dump(X_fwds, pickle_out)
pickle_out.close()