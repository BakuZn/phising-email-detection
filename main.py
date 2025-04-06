import pandas as pd

df=pd.read_csv('spam.csv')
print(df.head())
print("\nColumns:",df.columns)
print("\nshapes:",df.shape)

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Split into words
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords + stem
    return ' '.join(words)