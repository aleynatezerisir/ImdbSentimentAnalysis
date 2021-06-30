# stemmed, count_vectorizer, n_gram(1,2)
import data_preprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
# count vectorizer with stemmed data
ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))
ngram_vectorizer.fit(data_preprocessing.stemmed_reviews_train)

X = ngram_vectorizer.transform(data_preprocessing.stemmed_reviews_train)
X_test = ngram_vectorizer.transform(data_preprocessing.stemmed_reviews_test)

print("Stemmed data and n-gram(1,2) and count vectorizer (bag of words) : ")

lg = LogisticRegression(max_iter = 1000)
lg.fit(X, data_preprocessing.target)
print("Logistic regression Accuracy : %s"
          % accuracy_score(data_preprocessing.target, lg.predict(X_test)))

svm = LinearSVC()
svm.fit(X, data_preprocessing.target)
print("SVM Accuracy: %s"
          % accuracy_score(data_preprocessing.target, svm.predict(X_test)))

nb = MultinomialNB()
nb.fit(X, data_preprocessing.target)
print("Naive Bayes Accuracy: %s"
          % accuracy_score(data_preprocessing.target, nb.predict(X_test)))

# Stemmed data and n-gram(1,2) and count vectorizer (bag of words) :
# Logistic regression Accuracy : 0.8878
# SVM Accuracy: 0.88436
# Naive Bayes Accuracy: 0.85688