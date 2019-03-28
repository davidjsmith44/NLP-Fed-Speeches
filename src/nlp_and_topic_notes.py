''' NLP and TOPIC Modeling notes '''
# 1. Load dataframe
import pandas as pd
import numpy as np 
import pickle
df = pickle.load( open( "../data/all_fed_speeches", "rb" ) )
df.info()

# 2. Creating text pipeline to remove caps, punctuation, stop words and to stem and lemmazied
docs_cleaned = [our_text_pipeline(doc, stops=stop_words, lemmatize=True) 
 for doc in corpus]

def our_text_pipeline(doc, stops={}, lemmatize=False):
    '''
    Args:
        doc (str): the text to be tokenized
        stops (set): an optional set of words (tokens) to exclude
        lemmatize (bool): if True, lemmatize the words
    
    Returns: 
        tokens (list of strings)
    '''
    doc = doc.lower().split()
    punct = set(string.punctuation)
    tokens = [''.join([char for char in tok if char not in punct]) 
              for tok in doc]
    if stops:
        tokens = [tok for tok in tokens if (tok not in stops)]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return tokens

def our_count_vectorizer(docs):
    '''
    Args:
        docs (list of lists of strings): corpus
    Returns:
        X_count (numpy array): count vectors
        vocab (list of strings): alphabetical list 
                                 of unique words
    '''
    vocab_set = set()
    for doc in docs:
        vocab_set.update(doc)

    vocab = sorted(vocab_set)

    X_count = np.zeros(shape=(len(docs), len(vocab)))

    for i, doc in enumerate(docs):
        for word in doc:
            j = vocab.index(word)
            X_count[i,j] += 1
    return X_count, vocab  

X_count, vocab = our_count_vectorizer(docs_cleaned)

count_vectors = pd.DataFrame(data=X_count, columns=vocab)
count_vectors

# 𝑇𝐹𝑤𝑜𝑟𝑑,𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡=#_𝑜𝑓_𝑡𝑖𝑚𝑒𝑠_𝑤𝑜𝑟𝑑_𝑎𝑝𝑝𝑒𝑎𝑟𝑠_𝑖𝑛_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡 / 𝑡𝑜𝑡𝑎𝑙_#_𝑜𝑓_𝑤𝑜𝑟𝑑𝑠_𝑖𝑛_𝑑𝑜𝑐𝑢𝑚𝑒𝑛𝑡
# Term_Freq(word, doc) = # times word appears in doc / total # of words in the document

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
count_vectorizer = CountVectorizer(tokenizer=our_text_pipeline, stop_words='english')

print(count_vectorizer.vocabulary_)
print(count_vectorizer.get_feature_names())

# JUST USING NLTK HERE AND BELOW
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

# tokenize documents
from nltk.tokenize import word_tokenize
tokenized = [word_tokenize(content.lower()) for content in documents]

# removing stop words
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
docs = [[word for word in words if word not in stop]
        for words in tokenized]

# Stemming/Lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

docs_porter = [[porter.stem(word) for word in words]
               for words in docs]
docs_snowball = [[snowball.stem(word) for word in words]
                 for words in docs]
docs_wordnet = [[wordnet.lemmatize(word) for word in words]
                for words in docs]

## Print the stemmed and lemmatized words from the first document
print("%16s %16s %16s %16s" % ("word", "porter", "snowball", "lemmatizer"))
for i in range(min(len(docs_porter[0]), len(docs_snowball[0]), len(docs_wordnet[0]))):
    p, s, w = docs_porter[0][i], docs_snowball[0][i], docs_wordnet[0][i]
    if len(set((p, s, w))) != 1:
        print("%16s %16s %16s %16s" % (docs[0][i], p, s, w))

## choosing one
my_docs = docs_snowball

## Bag of Words and TFIDF

# 1. Create your vocab, a set of words UNIQUE over the whole corpus (list of documents which are lists of strings). A set is a good datatype for this since it doesn't allow duplicates. At the end you'll want to convert it to a list so that we can deal with our words in a consistent order.

vocab_set = set()
[[vocab_set.add(token) for token in tokens] for tokens in my_docs]
vocab = list(vocab_set)
# 2. Create a reverse lookup for the vocab list. This is a dictionary whose keys are the words and values are the indices of the words (the word id). This will make things much faster than using the list index function.

vocab_dict = {word: i for i, word in enumerate(vocab)}

# 3. Now let's create our word count vectors manually. Create a numpy matrix where each row corresponds to a document and each column a word. The value should be the count of the number of times that word appeared in that document.

import numpy as np

word_counts = np.zeros((len(docs), len(vocab)))
for doc_id, words in enumerate(my_docs):
    for word in words:
        word_id = vocab_dict[word]
        word_counts[doc_id][word_id] += 1

# 4. Create the document frequencies. For each word, get a count of the number of documents the word appears in (different from the number of times the word appears!).

df = np.sum(word_counts > 0, axis=0)
# 5. Normalize the word count matrix to get the term frequencies. This means dividing each count by the L1 norm (the sum of all the counts). This makes each vector a vector of term frequencies.

tf_norm = word_counts.sum(axis=1)
tf_norm[tf_norm == 0] = 1
tf = word_counts / tf_norm.reshape(len(my_docs), 1)


# 6. Multiply the term frequency matrix by the log of the inverse of the document frequences to get the tf-idf matrix.

idf = np.log((len(my_docs) + 1.) / (1. + df)) + 1.
tfidf = tf * idf
# 7. Normalize the tf-idf matrix as well by dividing by the l2 norm.

tfidf_norm = np.sqrt((tfidf ** 2).sum(axis=1))
tfidf_norm[tfidf_norm == 0] = 1
tfidf_normed = tfidf / tfidf_norm.reshape(len(my_docs), 1)




''' Using sklearn  '''
# 1. Write the tokenize function. It should use nltk's word_tokenize as well as the stemmer or lemmatizer that you chose to use.

def tokenize(doc):
    '''
    INPUT: string
    OUTPUT: list of strings

    Tokenize and stem/lemmatize the document.
    '''
    return [snowball.stem(word) for word in word_tokenize(doc.lower())]

# 2. Apply the CountVectorizer on the whole corpus. Use your tokenize function from above. Do you get the same results as you did when you created this by hand?

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

countvect = CountVectorizer(stop_words='english', tokenizer=tokenize)
count_vectorized = countvect.fit_transform(documents)

#Compare my results:

words = countvect.get_feature_names()
print("sklearn count of 'dinner':", count_vectorized[0, words.index('dinner')])
print("my count of 'dinner':", word_counts[0, vocab_dict['dinner']])

# These are both 2.

# 3. Apply the TfidfVectorizer.

tfidfvect = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
tfidf_vectorized = tfidfvect.fit_transform(documents)

# compare my results
words_tfidf = tfidfvect.get_feature_names()
print("sklearn tfidf of 'dinner':", tfidf_vectorized[0, words_tfidf.index('dinner')])
print("my tfidf of 'dinner':", tfidf[0, vocab_dict['dinner']])

''' extra cretid 1: Cosine Similarity using TFIDF '''
#Now that we're comfortable with tokenizing documents, let's use the cosine similarity to find similar documents.

# 1. Use sklearn's linear_kernel to compute the cosine similarity between two documents.

from sklearn.metrics.pairwise import linear_kernel

cosine_similarities = linear_kernel(tfidf_vectorized, tfidf_vectorized)

# 2. Now iterate over all possible pairs (as in 2 for loops iterating over the same list of documents) print the cosine similarities of their tfidf scores for each documents bag of words.

for i, doc1 in enumerate(docs):
    for j, doc2 in enumerate(docs):
        print(i, j, cosine_similarities[i, j])

''' Extra Credit 2: Part of speech tagging '''
# As a side note, let's take a quick look at Part of speech tagging. These part of speech tags can be used as features.

# You can see the documentation on the part of speech tagger in the nltk book ch 5

# 1. Since part of speech tagging takes a long time, pick off a single document.

# 2. Create a part of speech tagged version of the document. Which version of your documents should you use? The original tokenized one, the one with stop words removed, or the stemmed version? Try all of them and take a look at the results to see which one performs the best.

# 3. What happens if I part of speech tag my bag of words? Does it perform well? Why or why not?



## From the topic modeling section
'''
This script provides functions to read & analyze the contents of NYT articles
using our custom NMF class -and- using the NMF implementation from scikit-learn.
For a discussion of this script, see `pair.ipynb`.
'''

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF as NMF_sklearn

from my_nmf import NMF


def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):
    '''
    Build and return a **callable** for transforming text documents to vectors,
    as well as a vocabulary to map document-vector indices to words from the
    corpus. The vectorizer will be trained from the text documents in the
    `contents` argument. If `use_tfidf` is True, then the vectorizer will use
    the Tf-Idf algorithm, otherwise a Bag-of-Words vectorizer will be used.
    The text will be tokenized by words, and each word will be stemmed iff
    `use_stemmer` is True. If `max_features` is not None, then the vocabulary
    will be limited to the `max_features` most common words in the corpus.
    '''
    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    stop_set = set(stopwords.words('english'))

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary


def softmax(v, temperature=1.0):
    '''
    A heuristic to convert arbitrary positive values into probabilities.
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s


def hand_label_topics(H, vocabulary):
    '''
    Print the most influential words of each latent topic, and prompt the user
    to label each topic. The user should use their humanness to figure out what
    each latent topic is capturing.
    '''
    hand_labels = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:20]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_five]))
        label = input('please label this topic: ')
        hand_labels.append(label)
        print()
    return hand_labels


def analyze_article(article_index, contents, web_urls, W, hand_labels):
    '''
    Print an analysis of a single NYT articles, including the article text
    and a summary of which topics it represents. The topics are identified
    via the hand-labels which were assigned by the user.
    '''
    print(web_urls[article_index])
    print(contents[article_index])
    probs = softmax(W[article_index], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        print('--> {:.2f}% {}'.format(prob * 100, label))
    print()


def main():
    '''
    Run the unsupervised analysis of the NYT corpus, using NMF to find latent
    topics. The user will be prompted to label each latent topic, then a few
    articles will be analyzed to see which topics they contain.
    '''
    # Load the corpus.
    df = pd.read_pickle("data/articles.pkl")
    contents = df.content
    web_urls = df.web_url

    # Build our text-to-vector vectorizer, then vectorize our corpus.
    vectorizer, vocabulary = build_text_vectorizer(contents,
                                 use_tfidf=True,
                                 use_stemmer=False,
                                 max_features=5000)
    X = vectorizer(contents)

    # We'd like to see consistent results, so set the seed.
    np.random.seed(12345)

    # Find latent topics using our NMF model.
    factorizer = NMF(k=7, max_iters=35, alpha=0.5)
    W, H = factorizer.fit(X, verbose=True)

    # Label topics and analyze a few NYT articles.
    # Btw, if you haven't modified anything, the seven topics which should
    # pop out are:  (you should type these as the labels when prompted)
    #  1. "football",
    #  2. "arts",
    #  3. "baseball",
    #  4. "world news (middle eastern?)",
    #  5. "politics",
    #  6. "world news (war?)",
    #  7. "economics"
    hand_labels = hand_label_topics(H, vocabulary)
    rand_articles = np.random.choice(list(range(len(W))), 15)
    for i in rand_articles:
        analyze_article(i, contents, web_urls, W, hand_labels)

    # Do it all again, this time using scikit-learn.
    nmf = NMF_sklearn(n_components=7, max_iter=100, random_state=12345, alpha=0.0)
    W = nmf.fit_transform(X)
    H = nmf.components_
    print('reconstruction error:', nmf.reconstruction_err_)
    hand_labels = hand_label_topics(H, vocabulary)
    for i in rand_articles:
        analyze_article(i, contents, web_urls, W, hand_labels)


if __name__ == '__main__':

    main()





