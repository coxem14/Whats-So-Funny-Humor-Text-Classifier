**********************************************
# What's So Funny? - Humor Text Classifier
**********************************************

#### Erin Cox
#### https://github.com/coxem14/Capstone-2
*Last update: 12/4/2020*
***

<p align = 'center'>
    <img src = ''>
</p>

## Table of Contents
1. [Background](#Background)
2. [Data](#Data)
3. [Analysis](#Analysis)
    * [](#)
4. [Future Ideas](#)

* https://www.kaggle.com/moradnejad/200k-short-texts-for-humor-detection
* Build model(s) to classify text as humorous or serious
* Identify different clusters/topics within humorous text
* Identify different clusters/topics of serious text
* Create a program to generate jokes

## Background

Intro for why text classification is useful and in particular applications for humor detection

## Dataset



https://www.kaggle.com/moradnejad/200k-short-texts-for-humor-detection

https://arxiv.org/abs/2004.12765 ColBERT: Using BERT Sentence Embedding for Humor Detection Created by Issa Annamoradnejad and team at Cornell University

## EDA

### Initial Inspection

Brief summary of what dataset looked like, steps taken to clean, and final dataset I moved forward with

After loading the dataset into Jupyter Notebook using Pandas, I started my analysis by inspecting the data to ensure they were as expected. Below is an example of the first 7 rows of the dataframe.

<p align = 'center'>
    <img src = ''>
</p>

The dataframe has two columns, 'text' and 'humor'. I used .info(), .unique(), and .value_counts() to inspect the dataframe. In all, there were 200k non-null rows. The text column contained short text strings, and the humor column contained either True or False boolean values. There were exactly 100k each of True and False labels.

### Word Clouds

To get an idea of what words were used most frequently in the humorous texts and the serious texts, I created word clouds. The word cloud generator takes a single string of text, so I created a few functions that I could use to get the text from my dataframe and clean it up and generate a word cloud.

```
  import string
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.tokenize import RegexpTokenizer
  from nltk.stem.porter import PorterStemmer as PS

  def clean_corpus(X):
    '''
    Takes in a series or list of strings and returns one string of cleaned text.
    '''
    # lowercase the strings
    corpus = [text.lower() for text in X] 

    # declare regular expression tokenizer, and split strings into words while keeping contractions together
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
```
I also thought it would be fun to play around with the display of the word clouds, so I masked the colors with some images. The code I referenced to do this can be found __[here](https://amueller.github.io/word_cloud/auto_examples/masked.html#sphx-glr-auto-examples-masked-py)__.




## Models 

What features did I use to predict with, how important where those features
How can I go back and change the model to improve performance

## Final Thoughts

Recap important findings
Next steps

Things to look into: n-grams

### References
https://amueller.github.io/word_cloud/auto_examples/masked.html#sphx-glr-auto-examples-masked-py


[Back to Top](#Table-of-Contents)