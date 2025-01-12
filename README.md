# Sentiment-Analysis
---
## 📜 Project Description:
 - This project focuses on building a sentiment analysis application that can predict the sentiment of restaurant reviews as either positive or negative. The user can input a restaurant 
   review, select one of several machine learning models for sentiment classification, and receive a prediction of whether the sentiment is positive or negative. The application is 
   built using Streamlit for the frontend and various machine learning algorithms for the backend.
---
## 📝 Overview:
 - This project aims to perform sentiment analysis on restaurant reviews to determine whether the reviews are positive or negative. Sentiment analysis is a key application of Natural 
   Language Processing (NLP) and helps in understanding customer opinions and improving services based on feedback.
---
## 📦 Dataset:
 - **Dataset Name:** Restaurant Reviews Dataset
 - **Source:** The dataset contains a collection of restaurant reviews, where each review is labeled as either positive or negative.
 - **Dataset Format:** The dataset is stored in a tab-separated value (TSV) file, where each row contains:
     - **Review:** A text field containing the restaurant review.
     - **Sentiment:** A binary target value (1 for positive, 0 for negative).
  The dataset has 1000 entries, which were used to train and test the models.
---
## 🤖 Technologies Used:
 - `Python` - The primary programming language used for both backend and machine learning tasks.
 - `Streamlit` - Used to build the interactive frontend for the sentiment analysis tool.
 - `Scikit-learn` - Machine learning library used for building, training, and evaluating models.
 - `NLTK` - Natural Language Toolkit used for text preprocessing, including tokenization, stopword removal, and lemmatization.
 - `Pickle` - Used to save the trained models and the vectorizer.
---
## ⚙ Algorithm Implemented:
 - `Naive Bayes (Multinomial)`
---
## 🔍 Data Preprocessing:
 -  Text Cleaning:
    - Removed all non-alphabetical characters.
    - Converted text to lowercase.
    - Tokenized text into words.
    - Removed stopwords.
    - Performed stemming using the Porter Stemmer.

 -  Feature Extraction:
    - Used TfidfVectorizer to convert the text into a numerical feature matrix.

 -  Model Training:
    - Split the dataset into 80% training and 20% testing sets.
    - Trained the Naive Bayes model and evaluated its performance.

 -  Frontend Building:
    - Developed a simple user interface using Streamlit.
    - The user can input a restaurant review, and the application predicts whether the review is positive or negative.
---
## 📈 Results:
 - The Naive Bayes classifier achieved the highest accuracy for this task.
 - Performance Metrics for Naive Bayes:
   - Accuracy: 76.5%
   - Bias: 94.75%
   - Variance: 76.5%
---
## 🎯 Conclusion:
- The Naive Bayes model proved to be the most effective in this sentiment analysis project. Its simplicity and effectiveness in text classification make it a suitable choice for real- 
  time applications. This project demonstrates the application of NLP techniques and machine learning to extract meaningful insights from textual data.
---
## 🌐 Streamlit (Frontend)=()
---
