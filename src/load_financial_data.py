''' LOAD_FINANCIAL_DATA.PY

This file
    1. Downloads historical interest rate data from quandl.com
    2. Cleans the data and saves it to a dataframe
        X columns = ['3 MO', '6 MO', etc]
    3. Calculates Zero Coupon (spot) rates and saves to a file
    4. Calculates forward rates and saves to a file
    5. Downloads FX data for EUR, GBP, JPY and places that data into a dataframe called FX
        FX columns = ['EUR', 'GBP', 'JPY']
        NOTE: No preprocessing has been done on this file yet!
    6. The FX data is saved in a pickle file called 'FX_data.p'

'''

def zero_coupon_bond_price(par, ytm, time):
    ''' Takes the par price, ytm and time to maturity and returns the spot price of the bond'''
    return par / (1 + ytm/2) ** (time*2)

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

def build_zeros_and_forwards(X):
    # Discount rates are already par rates
    X_zeros = X[['three_m', 'six_m', 'one_y']].copy()
    X_fwds = X[['six_m', 'one_y']].copy() # will overwrite one year forwards below

    # Start building zero rate curves (spot rates)
    X_zeros['two_y'] = bootstrap_2yr(X)
    X_zeros['three_y'] = bootstrap_3yr(X, X_zeros)
    X_zeros['five_y'] = bootstrap_5yr(X, X_zeros)
    X_zeros['seven_y'] = bootstrap_7yr(X, X_zeros)
    X_zeros['ten_y'] = bootstrap_10yr(X, X_zeros)

    #bond_prices_3M, bond_prices_6M, bond_prices_1YR
    z_pr_6m  = zero_coupon_bond_price(par = 100, ytm = X_zeros['six_m'], time= 0.5)
    z_pr_1y  = zero_coupon_bond_price(par = 100, ytm = X_zeros['one_y'], time= 1.0)
    z_pr_2y  = zero_coupon_bond_price(par = 100, ytm = X_zeros['two_y'], time= 2.0)
    z_pr_3y  = zero_coupon_bond_price(par = 100, ytm = X_zeros['three_y'],time= 3.0)
    z_pr_5y  = zero_coupon_bond_price(par = 100, ytm = X_zeros['five_y'],time= 5.0)
    z_pr_7y  = zero_coupon_bond_price(par = 100, ytm = X_zeros['seven_y'], time= 7.0)
    z_pr_10y  = zero_coupon_bond_price(par = 100, ytm = X_zeros['ten_y'], time= 10.0)

    # taking the forward rates from the zero prices
    fwd_6_12 = z_pr_6m / z_pr_1y
    fwd_1_2= z_pr_1y/z_pr_2y
    fwd_2_3 =z_pr_2y/z_pr_3y
    fwd_3_5 =z_pr_3y/z_pr_5y
    fwd_5_7 =z_pr_5y/z_pr_7y
    fwd_7_10=z_pr_7y/z_pr_10y

    # now we need to account for periods that are not one year and adjusting to annualized rates
    fwd_6_12 = fwd_6_12**2
    fwd_3_5 =np.sqrt(fwd_3_5)
    fwd_5_7 =np.sqrt(fwd_5_7)
    fwd_7_10=fwd_7_10**(1/3)

    # Changing these back into interet rates from total returns
    X_fwds['one_y'] = fwd_6_12 - 1
    X_fwds['two_y'] = fwd_1_2 -1
    X_fwds['three_y'] = fwd_2_3 -1
    X_fwds['five_y'] = fwd_3_5 -1
    X_fwds['seven_y'] = fwd_5_7 -1
    X_fwds['ten_y'] = fwd_7_10 -1

    return X_fwds, X_zeros

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


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import quandl
    import pickle
    import datetime as datetime

    df_treas = quandl.get("USTREASURY/YIELD", authtoken="zBYbsY7fujcHokgXQdsY",
        start_date = "2006-01-01", end_date="2019-03-28")

    # The 3 month Treasury has three dates at the end of 2008
    # where the rate is zero. Pandas is treating this as Nan
    # I am replacing these with zero, because they are zeros.
    df_treas['3 MO'] = df_treas['3 MO'].fillna(0.0001)

    X = df_treas.copy()
    # The following series are incomplete over the sample period and are removed
    X = X.drop(['1 MO', '2 MO', '20 YR', '30 YR'], axis=1)

    # transform to interest rates
    X = X/100

    X = X.rename(columns = {'3 MO': 'three_m',
                        '6 MO': 'six_m',
                        '1 YR': 'one_y',
                        '2 YR': 'two_y',
                        '3 YR': 'three_y',
                        '5 YR': 'five_y',
                        '7 YR': 'seven_y',
                        '10 YR': 'ten_y'})

    #build the forwards and zeros
    X_fwds, X_zeros  = build_zeros_and_forwards(X)

    # add first differences
    X = df_add_first_diff(X)
    X_zeros = df_add_first_diff(X_zeros)
    X_fwds = df_add_first_diff(X_fwds)

    # Saving the X_zeros to a pickle file in case we need later
    pickle_out = open('../data/zero_rates', 'wb')
    pickle.dump(X_zeros, pickle_out)
    pickle_out.close()

    # Saving the forward rates
    pickle_out = open('../data/forward_rates', 'wb')
    pickle.dump(X_fwds, pickle_out)
    pickle_out.close()

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

