import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 0)

from os import path
from PIL import Image
import os

from collections import Counter

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_curve, silhouette_samples, silhouette_score
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

from pprint import pprint

import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer as PS

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from wordcloud import WordCloud, ImageColorGenerator

'''Text Cleaning Helper Function'''

def clean_corpus(X):
    '''
    Takes in a series or list of strings.
    Returns one string of cleaned text.
    '''
    # lowercase the strings
    corpus = [text.lower() for text in X] 

    # declare regular expression tokenizer
    # split strings into words while keeping contractions together
    tokenizer = RegexpTokenizer("[\w']+")
    tokens = list(map(tokenizer.tokenize, corpus)) 
    
    # remove punctuation
    punc = set(string.punctuation)
    tokens_no_punc = [[word for word in words if word not in punc]
        for words in tokens]
   
    # remove stopwords
    s_words = set(stopwords.words('english'))
    tokens_no_sw = [[word for word in words if word not in s_words]
        for words in tokens_no_punc]
    
    # stem the words to get rid of multiple forms of the same word
    porter = PS()
    tokens_stemmed = [[porter.stem(word) for word in text] for text in tokens_no_sw]
    
    # join all words into one string
    cleaned_corpus = [' '.join(word) for word in tokens_stemmed]
    
    return cleaned_corpus



'''Word Cloud Helper Functions'''

def make_word_cloud(corpus, wordcloud_obj):
    cleaned_corpus = ' '.join(clean_corpus(corpus))
    word_cloud = wordcloud_obj
    word_cloud.generate(cleaned_corpus)
    return word_cloud

def get_top_words(word_cloud, num_words=10):
    word_list = list(word_cloud.words_)
    top_words = word_list[:num_words]
    return f'The top {num_words} words are: {", ".join(top_words)}'



'''Model Fitting and Evaluation Helper Functions'''

#Used sklearn's pipeline helper function to create the pipelines.

# mnb_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
#                         ('model', MultinomialNB())])

def fit_predict_model_accuracy(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    accuracy = np.mean(predicted == y_test)
    print("\nThe accuracy on the test set is {0:0.3f}.".format(accuracy))
    return predicted


def create_confusion_matrix(y_test, predicted, targets, ax):
    cm = confusion_matrix(y_test, predicted)
    sns.set(font_scale=1.7)
    sns.heatmap(cm.T, square = True, annot = True, fmt = 'd', 
            xticklabels = targets, yticklabels = targets, ax=ax)


def get_predictions(text, model):
    prediction = model.predict([text])
    return prediction













if __name__ == "__main__":
    pass