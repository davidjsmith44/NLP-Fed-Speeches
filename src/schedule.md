

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




