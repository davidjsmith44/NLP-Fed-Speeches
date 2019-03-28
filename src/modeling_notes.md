Modeling notes


NLP

First hyperparameter I need to decide on is whether to use lemmatized or stemmer on the speeches

Lemmatizer = will keep the root words actual words but will take more time
            Also the lemmatizer needs to be given a part of speech to process

Stemming = will simply chop words into their root, but the resulting root may              not be an actual word. 

Lemmatizer  ->  from nklk.stem.wordnet import WordNetLemmatizer
                wordnet = WordNetLemmatizer()
                wordnet.lemmatize(word)   - within a double loop of docs and                                words

Stemmer         from nltk.stem.porter import PorterStemmer
                porter = PorterStemmer()
                porter.stem(word)

                from nltk.stem.snowball import SnowballStemmer
                snowball = SnowballStemmer('english')
                snowball.stem(word)
                