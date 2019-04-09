
April 9th last day
1. Use federal resereve policy statements and scrape the new web page
2. Shorten the time horizon
3. Make a tail for the model (only estimate X days past)
4. redo for ARIMAX
5. work on presentation



1. Create PCA based models                                                               
2. Add levels to the ARIMAX models
3. Redo speeches to only include 'outlook' or economy speeches
 
FOR THE PRESENTATION 
1. Some metric that describes how well these models perform
    -could be overlapping ditributions
    -means of each
    -look at the coefficients of the model (possibly have these returned to the class when fit)
2. Soundbites - Fed speeches will move the level of interest rates on avg XX based on how different they are
3. Some demonstration of how different the speeches are from one to the other
4. Better code for github (just clean this shit)

APRIL 4th List
1. Create the pipeline
    -starts with current training set
    -take PCA on the training set
    -ARIMAX on first element of shocks
    -take one period changes to the shocks
    -punch back into term structure model
    move on to next period



HIGHEST PRIORITY TO GET THE MVP
WTF - I am using longer set of interest rates than I have FED speeches. No wonder we are seeing something wrong!
2. Create function to map forwards back into interest rates
3. TAKE THE ARIMAX tutorial and make sure you know how to use this
4. Map the speeches (the whole current shit show) into a dataframe with cosine sim
5. Run an initial ARIMAX model
6. Create plotting functions for the histograms and reports on mean and std deviation and kurtosis


    2. Finally make the call in what type of speeches we are going to
        use in this model (FOMC only?)
    Put interest rate pull information on github



    NLP Clean up
    1. Clean up this code to make it publishable
    2. Verity how it works when there are 4 speeches at a time
        a. Do I need to take a max of the speeches vs average?
        b. Should this be returning everything so that I can use all as hyper parameters?
    3. Should I store the most important words from the simulation for use later in      explaining the larger cosine similarity?
    5. Add speaker titles and possibly look at just 'Economic Outlook' of speeches
    6. Is there an interactive way to present the data over time

    Presentation
    1. Create a summary document with the following
        -train test split
        -graphs of top features
        -explaination of what I need to do to the yeild curves

    FRIDAY March 29th
    FIRST test that the pickle file worked!
    3. Transform the interest rate data into forward rates
        -provide a brief explaination of why
        -create charts that show what this looks like
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

# TO BE DONE April 2, 2018
1. NLP pipeline - change number of words, create several different vectors for the data based on hyperparameters
    -change how many features
    -lemmatize
    -how many speeches to compare them to
2. Do I just look at the speeches labeled 'Outlook' or 'Policy' - try this as a model enhancement

# DATA WORK

2. What exactly do we want to do here with the output
    -create one period forecasts for each model
    -look at the distributions for the model
    -do all of this in the training datasets

3. Build something to compare forwards to zeros to yields
    -base model to run them all through based on my results from the train method
    -incorporate the PCA as well

4. Look into other forms of interence besided MLE!

5. What is the target? 10 year, 5 year, whole yield curve?
    Is there a way to compare this across all of the yield curves?
        -PCA will give us this by iteself.
        -Do I fit all of the models together for each part of the term structure and run each model like that?

6. Clean up code
7. Move on to SQL and something else.





Completed Tasks
    1. Create file that loads up the financial data from quandl for a specific date range
        -include FX rates from the majors
    2. Create initial plots in Jupyter notebook using Seaborn for presentation

    5. Extract the time series of principal components over time
    6. Create graphs of these things
    4. Look at initial autocorrelations for this model of just interest rates

NLP
    1. determine how to do cosine similarity of a new speech relative
        to the last n speeches (out of sample forecast)


    3. Create data pipeline for the quandl interest rate pull and
        publish initial results to github
    5. Get out the histocial eigenvalues and start plotting them
    6. Plot the impact on interest rates due to the eignvalues
    7. Determine game plan to close the loop!!!

    4. Get dataset together into one file
        -interest rates
        -Fed speeches
        -FX rates (and any other variables)

    9. Begin doing partial autocorrelations on the data

