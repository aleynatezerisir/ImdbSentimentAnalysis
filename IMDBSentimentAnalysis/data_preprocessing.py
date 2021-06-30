# import numpy as np
# import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# convert txt files to python list
reviews_train = []
reviews_test = []

for line in open('C:/Users/KASA/PycharmProjects/BitirmeTezi/movie_data/full_train.txt', 'r', encoding="utf8"):
    reviews_train.append(line.strip())

for line in open('C:/Users/KASA/PycharmProjects/BitirmeTezi/movie_data/full_test.txt', 'r', encoding="utf8"):
    reviews_test.append(line.strip())
target = [1 if i < 12500 else 0 for i in range(25000)]

# CLEANING THE DATA

# text processing like removing punctuation and HTML tags and making everything lower-case.
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

# removing stop words
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in english_stop_words])
        )
    return removed_stop_words

no_stop_words_train = remove_stop_words(reviews_train_clean)
no_stop_words_test = remove_stop_words(reviews_test_clean)

# lemmatization
def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

lemmatized_reviews_train = get_lemmatized_text(no_stop_words_train)
lemmatized_reviews_test = get_lemmatized_text(no_stop_words_test)

# stemming
def get_stemmed_text(corpus):
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

stemmed_reviews_train = get_stemmed_text(no_stop_words_train)
stemmed_reviews_test = get_stemmed_text(no_stop_words_test)



