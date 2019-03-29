Capstone Schedule



    '''
    Thursday March 29th Initial List
    Completed Tasks
    1. determine how to do cosine similarity of a new speech relative
        to the last n speeches (out of sample forecast)


    2. Finally make the call in what type of speeches we are going to
        use in this model (FOMC only?)
    3. Create data pipeline for the quandl interest rate pull and
        publish initial results to github
    4. Look at initial autocorrelations for this model of just interest rates
    5. Get out the histocial eigenvalues and start plotting them
    6. Plot the impact on interest rates due to the eignvalues
    7. Determine game plan to close the loop!!!


    NLP Clean up
    1. Clean up this code to make it publishable
    2. Verity how it works when there are 4 speeches at a time
        a. Do I need to take a max of the speeches vs average?
        b. Should this be returning everything so that I can use all as hyper parameters?
    3. Should I store the most important words from the simulation for use later in      explaining the larger cosine similarity?
    4. Hanle the datetime64 and timestamp data differently
    5. Add speaker titles and possibly look at just 'Economic Outlook' of speeches
    6. Is there an interactive way to present the data over time

    Presentation
    1. Create a summary document with the following
        -train test split
        -graphs of top features
        -explaination of what I need to do to the yeild curves

    FRIDAY March 29th
    FIRST test that the pickle file worked!
    1. Create file that loads up the financial data from quandl for a specific date range
        -include FX rates from the majors
    2. Create initial plots in Jupyter notebook using Seaborn for presentation
    3. Transform the interest rate data into forward rates
        -provide a brief explaination of why
        -create charts that show what this looks like
    4. Get dataset together into one file
        -interest rates
        -Fed speeches
        -FX rates (and any other variables)
    5. Extract the time series of principal components over time
    6. Create graphs of these things
    7. Convert these into dataframe
    8. Train, test, cv split
    9. Begin doing partial autocorrelations on the data
    10. Create initial pipeline for a time series model
        Delta PC = f (past delta_PC, changes in FX rates (lagged), fed_speech_similarity)
    11. Prediction/Complete the initial loop?
            -What am I really trying to do here
            -How do I do it
            -How can I demonstrate the cumulative impact of the speeches
                (cannot let the speeches bleed into one another)
                Possibly - if cosine_sim = X, here is the models cumulative impact on PCs
                        -here is the resulting impact on the forward rates
                        -here is the resulting impact on Treasury yields


Completed Tasks


