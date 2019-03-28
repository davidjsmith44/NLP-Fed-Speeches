''' Initial NLP Pipeline '''


def create_speech_list(df, date, numb_speeches):
    '''
    For a given date, the most recent speeches are returned in a dataframe

    INPUTS:
        df - the dataframe of all fed speeches
        date - the date needed to include no speeches before
        num_speeches - the number of most recent speeches to include
    OUTPUT:
        dataframe - this is a subset of the original dataframe with same columns
    '''
    filtered_df = df[df['date']<= date]
    if len(filtered_df)> numb_speeches:
        filtered_df = filtered_df.iloc[-1:-numb_speeches-1:-1]
    return filtered_df


def implement_ftidf_model(df):
    '''
    This takes the smaller version of the speeches and retuns the tf_idf calculations
    NOTE: NEED TO ADD HYPERPARAMETER DICTIONARY TO THIS
    '''
    import string
    import numpy as np
    import pandas as pd

    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer
    from nltk.stem.porter import PorterStemmer

    from nltk.tokenize import word_tokenize
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    doc_list = list(df['text'])
    tfidvect =TfidfVectorizer(lowercase=True,
                          stop_words='english',
                          max_features = 1000,
                          norm = 'l2',
                          use_idf = True,
                          smooth_idf=True,
                          sublinear_tf = False)
    tfidf_vectorized = tfidvect.fit_transform(doc_list).toarray()
    tfidvect.fit_transform(doc_list)
    # can this just be tfidvect.fit_transform(doc_list) without the assignment??
    return tfidvect, tfidf_vectorized





# Importing all of the Fed Speeches
import pickle
df = pickle.load( open( "../data/all_fed_speeches", "rb" ) )
df.info()

# create a date, filter the df and run the word processing on this date
this_date = datetime.datetime(2017, 1, 15)
recent_df = create_speech_list(df, date, numb_speeches)
recent_model = implement_ftidf_model(df)


# working on the cosine similarities !!!
from sklearn.metrics.pairwise import linear_kernel
# create the new dataframe of the next speech = called df_new
new_text = df_new['text']
# fit this dataframe to the last vectorization
new_tokens = tfidvect.transform(new_text)
cosine_sims = linear_kernel(new_tokens, tfidf_vectorized)
cosine_sims.shape



'''
NEED TO DO TODAY
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

'''