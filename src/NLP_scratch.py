''' Natural Language Processing using Nieve Bayes 


    # saving the df to a pickle file
    pickle_out = open('all_fed_speeches', 'wb')
    pickle.dump(df, pickle_out)
    pickle_out.close()


STEPS:
1. Load dataframe with pickle
2. follow steps from lecture to
    remove punctuation
    lowercase
    stump words
3. Vectorize

4. Topic modeling and look at titles


'''

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from nktk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 1. Load dataframe
import pickle
df = pickle.load( open( "../data/all_fed_speeches", "rb" ) )
df.info()

# 2. Creating text pipeline to remove caps, punctuation, stop words and to stem and lemmazied
''' Using nltk tokenizer '''
tokenized = [word_tokenize(content.lower()) for content in df['text']]

''' removing all stop words in the documents '''
stop = set(stopwords.words('english'))
docs = [[word for word in words if word not in stop] for words in tokenized]

''' NOW STEMMING and LEMITIZATION '''
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

# create the document lists
docs_porter = [[porter.stem(word) for word in words]
               for words in docs]
docs_snowball = [[snowball.stem(word) for word in words]
                 for words in docs]
docs_wordnet = [[wordnet.lemmatize(word) for word in words]
                for words in docs]

#test printing out the words and the different changes from lem/stem
print("%16s %16s %16s %16s" % ("word", "porter", "snowball", "lemmatizer"))
for i in range(min(len(docs_porter[0]), len(docs_snowball[0]), len(docs_wordnet[0]))):
    p, s, w = docs_porter[0][i], docs_snowball[0][i], docs_wordnet[0][i]
    if len(set((p, s, w))) != 1:
        print("%16s %16s %16s %16s" % (docs[0][i], p, s, w))

# picking one
my_docs = docs_porter

# create the vocabulary
vocab_set = set()
[[vocab_set.add(token) for token in tokens] for tokens in my_docs]
vocab = list(vocab_set)

# create a reverse lookup vocab list
vocab_dict = {word: i for i, word in enumerate(vocab)}

# all in one swath with sklearn
def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings

    Tokenize and stem/lemmatize the document.
    '''
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

# applying count vectorizer to this corpus

count_vect = CountVectorizer(stop_words='english', tokenizer = tokenize)
count_vectorized = countvect.fit_transform(documents)

# apply TfidVectorizer
tfidvect =TfidVectorizer(stop_words='english', tokenizer=tokenize)
tfidf_vectorized = tfidvect.fit_transform(documents)

## Cosine Similarity
from sklearn.metrics.pairwise import linear_kernel
cosine_similarities = linear_kernel(tfidf_vectorized, tfidf_vectorized)

#now iterate tover all possible pairs print the cosine similarieis of their tfidf 
# scores for each documents bag of words

for i, doc1 in enumerate(docs):
    for j, doc2 in enumerate(docs):
        print(i, j, cosine_similarities[i,j])


'''    NOTES AFTER A CRAPPY DAY. 
1. Do not like the way the stemmers work. Need to see if I can get
the TFidVectorizer to work with the lemmatizer.

2. If this works, I still have a lot of hyperparameters to 
deal with. (number of features, )

3. How do I want to compare the cosine similiarities? Is this to 
all other documents, or do I create a time series of these things?

The time series sounds like a good way to start
SIMPLIST MODEL
-just take speeches made by the Chair and compare them to the other
speeches made by the chair.
    Compare the average similarity of the new speech to the other group
    Use this as a variable to explain the changes in the yield curve

Possibly look at the impact of the speech on the steepness of the yield
curve. Thinking that the first principal component will have a smaller
impact than the longer term consequences?


PLAN
1. Get some sort of crap together for baseline speeches
2. create rolling cosine similarity of one speech versus last 10
3. Use these as a dummy variable for the PCA
FINISH? THIS INITIAL TEXT CRAP THIS WEEK

4. Start working on doing the EDA on the interest rates
5. PCA estimation (over training set)
6. Can I extract the base shocks over time based on these?
    -Want a plot of what they look like
    -Want to do some sort of autocorrelation (partial on them)
    -Want to do impulse responses and variance decomposition

