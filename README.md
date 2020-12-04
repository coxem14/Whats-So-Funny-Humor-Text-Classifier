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

<p align = 'center'>
    <img src = ''>
</p>

<p align = 'center'>
    <img src = ''>
</p>

The dataframe has two columns, 'text' and 'humor'. I used .info(), .unique(), and .value_counts() to inspect the dataframe. In all, there were 200k non-null rows. The text column contained short text strings, and the humor column contained either True or False boolean values. There were exactly 100k each of True and False labels.

### Word Clouds

To get an idea of what words were used most frequently in the humorous texts and the serious texts, I created word clouds. The word cloud generator takes a single string of text, so I created a few functions that I could use to get the text from my dataframe, clean it up, and generate a word cloud.

```
  import string
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.tokenize import RegexpTokenizer
  from nltk.stem.porter import PorterStemmer as PS

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
```
I also thought it would be fun to play around with the display of the word clouds, so I masked the colors with some images. The code I referenced to do this can be found __[here](https://amueller.github.io/word_cloud/auto_examples/masked.html#sphx-glr-auto-examples-masked-py)__.

nltk's word_tokenize
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/old_wc.png'>
</p>

nltk's RegexTokenizer
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/wc_humor.png'>
</p>

Top 10 words in humourous texts: call, say, one, know, go, make, what', joke, peopl, want

Humorous texts patterns seem to be: 
* What did one blank say to the blank? Punchline
* What do you call a blank that blank? Punchline
* What's the difference between blank and blank? Punchline
* A blank walks into a bar
* 'Knock Knock' jokes

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/wc_serious.png'>
</p>

Top 10 words in serious texts: photo, video, new, say, donald trump, trump, make, one, kid, take

Serious texts patterns seem to be:
* News headlines (new, video, watch, photo, report, first, say)
* Article/Blog post titles
* Donald Trump comes up a lot

## Train, Test, Split

```
X = texts['text']
y = texts['humor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                          shuffle=True, 
                                                          stratify=y)
```
Before I started training my models, I used sklearn's train_test_split function to split X and y into training sets and test sets. I stratified the split on y so I could ensure the datasets had a balanced ratio of humorous and serious texts.

The resulting training datasets had 150,000 samples, while the testing datasets had 50,000 samples.

## Model Selection

I wanted to build a variety of models to see which algorithms performed the best classifications. I used sklearn for all the models.

The primary supervised learning models I used are as follows:
* Multinomial Naive Bayes
* Bernoulli Naive Bayes

I also explored:
* Random Forest Classifier
* Multilayer Perceptron
* KMeans Clustering

### Pipelines

```
mnb_pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                         ('model', MultinomialNB())])
```
For each model, I used the TfidfVectorizer to featurize the texts. I found that limiting the number of features and removing stopwords resulted in models with lower accuracy than the default settings.

For the Naive Bayes models, I also used the default settings (alpha=1.0).

### Fitting the Models

I wanted to see how the models performed with cleaned and uncleaned data inputs, so I ran X_train and X_test through the corpus cleaner function prior to fitting and predicting, respectively.

I fit each model with X_train and y_train, got the predictions, and looked at the accuracy, precision, recall scores, and compared confusion matrices.

<p align = 'center'>
    <img src = ''>
</p>


## Final Thoughts

Recap important findings

Next steps:
Further cleaning for word clouds
Further tuning for models - cross validation, test size
Test multiple ks for KMeans
PCA to see which latent features are most important
CNN/RNN


Things to look into: n-grams

### References
https://amueller.github.io/word_cloud/auto_examples/masked.html#sphx-glr-auto-examples-masked-py


[Back to Top](#Table-of-Contents)