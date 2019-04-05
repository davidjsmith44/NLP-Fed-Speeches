''' Transforming interest rate data into zeros and forwards '''


def zero_coupon_bond_price(par, ytm, time):
    ''' Takes the par price, ytm and time to maturity and returns the spot price of the bond'''
    return par / (1 + ytm/2) ** (time*2)


    # price = 100
    # par = 100

    # bond_prices_3M = zero_coupon_bond_price(par, X['three_m'], time=0.25)
    # bond_prices_6M = zero_coupon_bond_price(par, X['six_m'], time=0.5)
    # bond_prices_1YR = zero_coupon_bond_price(par, X['one_y'], time=1.0)

    # return bond_prices_3M, bond_prices_6M, bond_prices_1YR

# NOW USING LOOP TO BUILD SPOT RATES ONE AT A TIME
# 2 year bond prices

''' BUILDING MY OWN FUNCTION TO SOLVE FOR BOOTSTRAPPING '''
# 2 year
def bootstrap_2yr(X):
    spots = np.zeros(shape=(len(X), 1))
    par = 100

    for i in range(len(X)):
        rate = 0
        r6 = X['six_m'].iloc[i]
        r12 = X['one_y'].iloc[i]
        cpn = X['two_y'].iloc[i] * par/2
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
        r6 = X['six_m'].iloc[i]
        r12 = X['one_y'].iloc[i]
        r18 = X_zeros['two_y'].iloc[i]
        r24 = X_zeros['two_y'].iloc[i]
        cpn = X['three_y'].iloc[i] * par/2
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
        r6 = X['six_m'].iloc[i]
        r12 = X['one_y'].iloc[i]
        r18 = X_zeros['two_y'].iloc[i]
        r24 = X_zeros['two_y'].iloc[i]
        r30 = X_zeros['three_y'].iloc[i]
        r36 = X_zeros['three_y'].iloc[i]

        cpn = X['five_y'].iloc[i] * par/2
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
        r6 = X['six_m'].iloc[i]
        r12 = X['one_y'].iloc[i]
        r18 = X_zeros['two_y'].iloc[i]
        r24 = X_zeros['two_y'].iloc[i]
        r30 = X_zeros['three_y'].iloc[i]
        r36 = X_zeros['three_y'].iloc[i]
        r42 = X_zeros['five_y'].iloc[i]
        r48 = X_zeros['five_y'].iloc[i]
        r54 = X_zeros['five_y'].iloc[i]
        r60 = X_zeros['five_y'].iloc[i]

        cpn = X['seven_y'].iloc[i] * par/2
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
        r6 = X['six_m'].iloc[i]
        r12 = X['one_y'].iloc[i]
        r18 = X_zeros['two_y'].iloc[i]
        r24 = X_zeros['two_y'].iloc[i]
        r30 = X_zeros['three_y'].iloc[i]
        r36 = X_zeros['three_y'].iloc[i]
        r42 = X_zeros['five_y'].iloc[i]
        r48 = X_zeros['five_y'].iloc[i]
        r54 = X_zeros['five_y'].iloc[i]
        r60 = X_zeros['five_y'].iloc[i]
        r66 = X_zeros['seven_y'].iloc[i]
        r72 = X_zeros['seven_y'].iloc[i]
        r78 = X_zeros['seven_y'].iloc[i]
        r84 = X_zeros['seven_y'].iloc[i]

        cpn = X['ten_y'].iloc[i] * par/2
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


''' BELOW WILL BE IN THE MAIN SECTION '''
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
import os
import pickle

# Load up the interest rate data
X = pickle.load(open("../data/interest_rate_data", "rb" ) )

# Start building zero rate curves (spot rates)
X_zeros = X[['three_m', 'six_m', 'one_y']].copy()
zeros_2yr = bootstrap_2yr(X)
X_zeros['two_y'] = zeros_2yr

zeros_3yr = bootstrap_3yr(X, X_zeros)
X_zeros['three_y'] = zeros_3yr

zeros_5yr = bootstrap_5yr(X, X_zeros)
X_zeros['five_y'] = zeros_5yr

zeros_7yr = bootstrap_7yr(X, X_zeros)
X_zeros['seven_y'] = zeros_7yr

zeros_10yr = bootstrap_10yr(X, X_zeros)
X_zeros['ten_y'] = zeros_10yr


# Saving the X_zeros to a pickle file
pickle_out = open('../data/zero_rates', 'wb')
pickle.dump(X_zeros, pickle_out)
pickle_out.close()

''' now working on forward rates
START BY IGNORING THE 3M RATE
 we will have annualized forward rates from
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
#bond_prices_3M, bond_prices_6M, bond_prices_1YR
z_pr_6m  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['six_m'],
                                    time= 0.5)

z_pr_1y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['one_y'],
                                    time= 1.0)

z_pr_2y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['two_y'],
                                    time= 2.0)

z_pr_3y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['three_y'],
                                    time= 3.0)

z_pr_5y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['five_y'],
                                    time= 5.0)

z_pr_7y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['seven_y'],
                                    time= 7.0)

z_pr_10y  = zero_coupon_bond_price(par = 100,
                                    ytm = X_zeros['ten_y'],
                                    time= 10.0)

# taking the forward rates from the zero prices
fwd_6_12 = z_pr_6m / z_pr_1y
fwd_1_2= z_pr_1y/z_pr_2y
fwd_2_3 =z_pr_2y/z_pr_3y
fwd_3_5 =z_pr_3y/z_pr_5y
fwd_5_7 =z_pr_5y/z_pr_7y
fwd_7_10=z_pr_7y/z_pr_10y

# now we need to account for periods that are two years and adjust by
# taking the sqare root
fwd_3_5 =np.sqrt(fwd_3_5)
fwd_5_7 =np.sqrt(fwd_5_7)
fwd_7_10=fwd_7_10**(1/3)

# annualize the first ????
X_fwds = X_zeros[['six_m', 'one_y']].copy()
X_fwds['two_y'] = fwd_1_2 -1
X_fwds['three_y'] = fwd_2_3 -1
X_fwds['five_y'] = fwd_3_5 -1
X_fwds['seven_y'] = fwd_5_7 -1
X_fwds['ten_y'] = fwd_7_10 -1

# take care of the 6 month rates to annualize
#X_fwds['one_y'] = fwd_6_12

pickle_out = open('../data/forward_rates', 'wb')
pickle.dump(X_fwds, pickle_out)
pickle_out.close()
