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
    