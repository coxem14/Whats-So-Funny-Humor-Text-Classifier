import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 0)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.neural_network import MLPClassifier

import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as PS

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



def clean_corpus(X):
    corpus = [text.lower() for text in X]
    tokenizer = RegexpTokenizer("[\w']+")
    tokens = list(map(tokenizer.tokenize, corpus))
    
    punc = set(string.punctuation)
    tokens_no_punc = [[word for word in words if word not in punc]
        for words in tokens]
   
    s_words = set(stopwords.words('english'))
    tokens_no_sw = [[word for word in words if word not in s_words]
        for words in tokens_no_punc]
    
    porter = PS()
    tokens_stemmed = [[porter.stem(word) for word in text] for text in tokens_no_sw]
    
    new_corpus = [' '.join(word) for word in tokens_stemmed]
    
    return new_corpus


def get_predictions(text, model):
    prediction = model.predict([text])
    return prediction