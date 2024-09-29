import pandas as pd
import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow as tf
import joblib
from sklearn.naive_bayes import BernoulliNB


# Load dataset
dataset = pd.read_csv('tweets.csv', encoding='ISO-8859-1')

# Preprocess function
import utils

import re

# Preprocess dataset
corpus = [] #list of cleaned reviews
y = []
print('stemming reviews...')
for i in range(0, 1600000, 16):
   corpus.append(utils.preprocess_review(dataset.iloc[i,5]))
  #  corpus.append(dataset.iloc[i,5])
   y.append(dataset.iloc[i,0])

# Vectorize text
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()


# Train-test split
print('separating data into test and training sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3)

# Train Naive Bayes model
model = BernoulliNB()
model.fit(X_train, y_train)
print('training model...')

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'model.pkl')
joblib.dump(cv, 'vectorizer.pkl')
