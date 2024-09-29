import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import contractions
import tensorflow as tf

def preprocess_review(review):
    # Expand contractions (e.g., "don't" -> "do not")
    review = contractions.fix(review)
    
    # Convert emojis to text (e.g., 😊 -> ":smiling_face:")
    review = emoji.demojize(review, delimiters=(" ", " "))
    
    # Remove URLs, mentions, and hashtags
    review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE) # URLs
    review = re.sub(r'@\w+', '', review) # Mentions
    review = re.sub(r'#\w+', '', review) # Hashtags
    
    # Remove special characters and digits, keeping only letters
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    
    # Convert to lowercase
    review = review.lower()
    
    # Tokenize the review
    review = review.split()
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.remove('not')  # Keep 'not' for negations
    
    # Lemmatize and remove stopwords
    review = [lemmatizer.lemmatize(word) for word in review if word not in all_stopwords]
    
    # Join the words back into a single string
    review = ' '.join(review)
    
    return review

def tokenize(dataset):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(dataset)
    return tokenizer