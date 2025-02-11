# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Importing the dataset
dataset = pd.read_csv(r"Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Cleaning the texts
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Training the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculating accuracy
ac = accuracy_score(y_test, y_pred)
print(f"Accuracy: {ac}")

# Bias and variance
bias = classifier.score(X_train, y_train)
variance = classifier.score(X_test, y_test)
print(f"Bias: {bias}")
print(f"Variance: {variance}")

# Save the trained MultinomialNB model to a file using pickle
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
print("Model saved to naive_bayes_model.pkl")

# Save the TfidfVectorizer to a pickle file
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(cv, file)
print("TfidfVectorizer has been saved to tfidf_vectorizer.pkl")
