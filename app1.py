import streamlit as st
import pickle
import numpy as np
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .stMarkdownContainer {{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set up background
image_base64 = get_base64_image("background.jpg")  # Update the path to your background image
set_background(image_base64)
# Load the Naive Bayes model
with open('naive_bayes_model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Load the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Title and Description
st.title("Sentiment Analysis")
st.write("This app analyzes the sentiment of restaurant reviews. Enter a review to see whether the sentiment is positive or negative.")

# User Input
user_input = st.text_area("Enter your review here:")

# Button for Prediction
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess and transform the user input
        transformed_input = tfidf_vectorizer.transform([user_input]).toarray()
        
        # Predict sentiment
        prediction = classifier.predict(transformed_input)
        
        # Display the result
        if prediction[0] == 1:
            st.success("The review sentiment is Positive 😊")
        else:
            st.error("The review sentiment is Negative 😞")
    else:
        st.warning("Please enter a review to analyze.")
