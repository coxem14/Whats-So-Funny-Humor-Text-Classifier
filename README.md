**********************************************
# What's So Funny? - Humor Text Classifier
**********************************************

#### Erin Cox
#### https://github.com/coxem14/Capstone-2
*Last update: 12/4/2020*
***

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/fry_meme.jpg'>
</p>

__[quickmeme](https://www.google.com/url?sa=i&url=http%3A%2F%2Fwww.quickmeme.com%2Fmeme%2F3jz8&psig=AOvVaw0pZpkhxvESUZWK5VQ8VAIn&ust=1607139073503000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCPiHzumxs-0CFQAAAAAdAAAAABAP)__

## Table of Contents
- [What's So Funny? - Humor Text Classifier](#whats-so-funny---humor-text-classifier)
      - [Erin Cox](#erin-cox)
      - [https://github.com/coxem14/Capstone-2](#httpsgithubcomcoxem14capstone-2)
  - [Table of Contents](#table-of-contents)
  - [Background - NEEDS UPDATED](#background---needs-updated)
  - [Data - NEEDS UPDATED](#data---needs-updated)
  - [EDA](#eda)
    - [Initial Inspection of Dataset](#initial-inspection-of-dataset)
    - [Word Clouds](#word-clouds)
  - [Train, Test, Split](#train-test-split)
  - [Model Selection](#model-selection)
    - [Pipelines](#pipelines)
    - [Fitting the Models](#fitting-the-models)
    - [Multinomial vs Bernoulli Naive Bayes](#multinomial-vs-bernoulli-naive-bayes)
    - [Random Forest Classifier](#random-forest-classifier)
    - [Multilayer Perceptron](#multilayer-perceptron)
  - [Problem Words](#problem-words)
    - [KMeans](#kmeans)
  - [Results from cohort submissions:](#results-from-cohort-submissions)
    - [MLP](#mlp)
    - [Bernoulli NB](#bernoulli-nb)
  - [Final Thoughts](#final-thoughts)
    - [References](#references)


## Background - NEEDS UPDATED

Intro for why text classification is useful and in particular applications for humor detection

## Data - NEEDS UPDATED

https://www.kaggle.com/moradnejad/200k-short-texts-for-humor-detection

https://arxiv.org/abs/2004.12765 ColBERT: Using BERT Sentence Embedding for Humor Detection Created by Issa Annamoradnejad and team at Cornell University

## EDA

### Initial Inspection of Dataset

After loading the dataset into Jupyter Notebook using Pandas, I started my analysis by inspecting the data to ensure they were as expected.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/texts_df_head.png'>
</p>

The dataframe has two columns, 'text' and 'humor'. I used .info(), .unique(), and .value_counts() to inspect the dataframe. 

In all, there were 200k non-null rows. The text column contained short text strings, and the humor column contained either True or False boolean values. There were exactly 100k each of True and False labels.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/texts_df_info.png'>
</p>

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/texts_humor_value_counts.png'>
</p>

[Back to Top](#Table-of-Contents)

### Word Clouds

To get a visualization of what words were used most frequently in the humorous texts and the serious texts, I created word clouds. The word cloud generator takes a single string of text, so I created a few functions that I could use to get the text from my dataframe, clean it up, and generate a word cloud.

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
[Back to Top](#Table-of-Contents)

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
* What did one blank say to the other blank? Punchline
* What do you call a blank that blank? Punchline
* What's the difference between blank and blank? Punchline
* A blank walks into a bar
* 'Knock Knock' jokes

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/wc_serious.png'>
</p>

Top 10 words in serious texts: photo, video, new, say, donald trump, trump, make, one, kid, take

Serious texts patterns seem to be:
* News headlines (new, video, watch, photo, report, first, say, show)
* Article/Blog post titles
* Donald Trump comes up a lot

[Back to Top](#Table-of-Contents)

## Train, Test, Split

```
X = texts['text']
y = texts['humor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                          shuffle=True, 
                                                          stratify=y)
```
Before I started training my models, I used sklearn's train_test_split function to split X and y into training sets and test sets. I stratified the split on y so I could ensure the datasets had a balanced ratio of humorous and serious texts.

The resulting training datasets had 150,000 texts, while the testing datasets had 50,000 texts.

## Model Selection

I wanted to build a variety of models to see which algorithms performed the best classifications and predictions. I used sklearn for all the models.

The primary models I built are as follows:
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

[Back to Top](#Table-of-Contents)

### Fitting the Models

I wanted to see how the models performed with cleaned and uncleaned data inputs, so I ran X_train and X_test through the corpus cleaner function prior to fitting and predicting, respectively.

### Multinomial vs Bernoulli Naive Bayes

With text classification, the most common Naive Bayes algorithm used is Multinomial, though Bernoulli can also perform well, especially in the case of short texts. In the case of binary classification, the primary difference between the two algorithms is that Multinomial ignores non-occuring features (just doesn't count them) whereas Bernoulli penalizes the non-occurrence of a feature that is an indicator for the given class.

Multinomial Likeihood - The probability of a word given humorous is the probability of a randomly-chosen word in a randomly-chosen humorous text being the current word

Bernoulli Likelihood - The probability of a word given humorous is the probability of a randomly-selected humorous text containing that word

> It seems to make sense to me that looking at the entire text for a word would be a great approach for short texts.

To determine which Naive Bayes model performed the best, I fit each model with X_train and y_train, got the predictions, and compared accuracy, precision, recall scores, confusion matrices, and ROC plots.

[Back to Top](#Table-of-Contents)

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/confusion_matrices.png'>
</p>

Overall, the cost of thinking something is humorous when it is really serious (false positive) is higher than thinking something is serious when is meant to be humorous (false negative). I want to chose the model with the most true positives, and the least false positives.

```
Model: Multinomial Naive Bayes
The accuracy on the test set is 0.916.
The precision on the test set is 0.901.
The recall on the test set is 0.936.

Model: Multinomial Naive Bayes - Cleaned
The accuracy on the test set is 0.893.
The precision on the test set is 0.890.
The recall on the test set is 0.896.

Model: Bernoulli Naive Bayes
The accuracy on the test set is 0.921.
The precision on the test set is 0.907.
The recall on the test set is 0.938.

Model: Bernoulli Naive Bayes - Cleaned
The accuracy on the test set is 0.895.
The precision on the test set is 0.882.
The recall on the test set is 0.912.

The model with the highest accuracy: Bernoulli Naive Bayes
The model with the highest precision: Bernoulli Naive Bayes
The model with the highest recall: Bernoulli Naive Bayes
```

The Bernoulli Naive Bayes model performed better across the board than Multinomial Naive Bayes. The cleaned versions performed worse than their 'unclean' counterparts. 

While accuracy is good, because I want to limit the number of false positives, precision would be the best metric to judge the performance of the model.

To get a better visual of the performance of the models, I plotted the receiver operating characteristic (ROC) curve and calculated the area under the curve (AUC). The AUC score will be close to 0.5 if the classifier isn't much better than random guessing, while it will be 1.0 for perfect classification.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/NB_ROC.png'>
</p>

```
Model: Multinomial Naive Bayes
The ROC AUC score for the model is 0.975.

Model: Bernoulli Naive Bayes
The ROC AUC score for the model is 0.977.

The model with the largest AUC: Bernoulli Naive Bayes
```

[Back to Top](#Table-of-Contents)

### Random Forest Classifier

```
rf_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=1000)),
                        ('model', RandomForestClassifier(n_estimators=1000, max_depth=10)])
```

For the Random Forest Classifier, I began with a relatively restricted model. I limited my features to 1000 in the TfidfVectorizer, and grew 1000 trees with a max depth of 10.

```
The accuracy on the test set is 0.851.
The precision on the test set is 0.888.
The recall on the test set is 0.804.
```

I decided to try testing accuracy over a range of hyperparameters.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/RF_Acc_Max_Feats.png'>
</p>

The model seemed to do better with more features in the Tfidf, but seemed to level off around 10,000. 

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/RF_Acc_Num_Trees.png'>
</p>

The highest accuracy was with less than 2000 trees.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/RF_Acc_Max_Depth.png'>
</p>

Accuracy increased as max depth increased - makes sense given that decision trees are prone to overfitting if left to make as many splits as they please...

[Back to Top](#Table-of-Contents)

Using the boundaries I explored above, I decided to try using RandomizedSearchCV to see if it could find a better set of parameters.

```
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth}

pprint(random_grid)

{'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
 'max_features': ['auto'],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
 
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               n_iter = 3, 
                               cv = 3, 
                               verbose = 2, 
                               n_jobs = -1)

rf_random.fit(X_train_tfid, y_train)
rf_random.best_params_

{'n_estimators': 1600, 'max_features': 'auto', 'max_depth': 90}
```

Running the model again with the new parameters:

```
The accuracy on the test set is 0.907.
The precision on the test set is 0.905.
The recall on the test set is 0.909.

The ROC AUC score for the model is 0.968.
```

Much better, still not quite on par with Naive Bayes. I think there is still room to improve, but RF is so much more computationally expensive for this type of classification, I don't think it is worth it.

[Back to Top](#Table-of-Contents)

### Multilayer Perceptron

With the mlp model, I tried a few different max features (1000, 5000, and 10,000). There was no difference in accuracy between 5000 and 10000, so I decided to limit it to save computation. I limited the batch size and hidden layer size as well (mostly just to make it run faster). My very basic first attempt resulted in my best model yet - I think there is definitely still room for improvement!

```
mlp_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=5000)),
                         ('model', MLPClassifier(batch_size=32,
                                                 hidden_layer_sizes=(32,),
                                                 early_stopping=True))])
```
```
The accuracy on the test set is 0.925.
The precision on the test set is 0.921.
The recall on the test set is 0.930.
```

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/Combined_ROC.png'>
</p>

```
Model: Multinomial Naive Bayes
The ROC AUC score for the model is 0.975.

Model: Bernoulli Naive Bayes
The ROC AUC score for the model is 0.977.

Model: Random Forest
The ROC AUC score for the model is 0.968.

Model: Multilayer Perceptron
The ROC AUC score for the model is 0.979.

The model with the largest AUC: Multilayer Perceptron
```
## Problem Words

I wanted to look into the misclassified texts to see the words with which the model has the most trouble.
I wrote a function that would return the corpus for the misclassifications (overall misclassifications, false positives, and false negatives). I could then use my make_word_cloud function to generate a word cloud easily.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/misclass_wc.png'>
</p>

Top 10 words in misclassified texts: say, make, want, people, go, one, kid, day, know, thing


### KMeans

Initially, I wanted to see what only 2 clusters would look like for the dataset with KMeans.

```
k = 2
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vec = vectorizer.fit_transform(X)
features = vectorizer.get_feature_names()
kmeans = KMeans(n_clusters=k, verbose=2)
kmeans.fit(X_vec) 

# Find the top 10 features for each cluster.
n_features = 10
top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-(n_features+1):-1]
print("top features (words) for each cluster:")
for num, centroid in enumerate(top_centroids):
    print(f"{num}, {', '.join(features[i] for i in centroid)}")
```

```
top features (words) for each cluster:
0, like, trump, new, just, people, does, don, photos, make, know
1, did, say, hear, cross, road, got, man, chicken, know, guy
```

Some sample texts from each cluster:

```
cluster 0:
Quote: What's a monster's favorite bean? a human bean.
 Label: True
Quote: Cheese shop exploded thankfully i was only hit by da brie
 Label: True
Quote: Stephanie gilmore's espy awards fashion is spot on (photos)
 Label: False
Quote: Why the 2-million pound ready-to-eat chicken recall is extra risky
 Label: False
Quote: Worrying is so stupid. it's like carrying an umbrella waiting for it to rain.
 Label: True
Quote: Watertown perspective: the boston marathon suspect manhunt on friday
 Label: False

cluster 1:
Quote: Did you hear about the guy who stole a dictionary from the library? he got away with words.
 Label: True
Quote: What did the triceratops sit on? its tricerabottom.
 Label: True
Quote: What did a bad teacher tell their wisecracking student? don't get smart with me.
 Label: True
Quote: What did mozart tell the terminator i'll be bach
 Label: True
Quote: Why did the melon plan a big elaborate wedding? because he cantaloupe.
 Label: True
Quote: What did iron say to silver after 30 years? you haven't ag-ed a bit.
 Label: True
```

It seems like cluster 1 is a much smaller, more patterned cluster. Cluster 0 is a mix.

```
Cluster 0:
     False (99587 texts)
     True (90012 texts)
Cluster 1:
     True (9988 texts)
     False (413 texts)
```

There are definitely more than just 2 topics (even within humor alone there are lots of different kinds of jokes). I want to find the best k, so I tried making some silhouette plots.

I started with just k=2

Unsuprisingly, it was pretty terrible.

```
For n_clusters = 2 The average silhouette_score is : 0.01396
```

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Capstone-2/blob/main/images/KMeans_silhouette.png'>
</p>

## Results from cohort submissions:

### MLP
```
Text: When the prosecuting attorney asked Gucci Mane if he was guilty he said, 'bitch I might be.'
Prediction: Humorous
Label: Humorous

Text: Camus says that when you wish yourself into the future you commit suicide by wishing yourself closer to your death.
Prediction: Serious
Label: Serious

Text: How did the random variable get into the club? By showing a fake i.i.d.
Prediction: Humorous
Label: Humorous

Text: Laugh it up! Humor is universal across human cultures — and fuels psychological research on everything from social perception to emotion
Prediction: Humorous
Label: Serious

Text: Saddest 6 word story: "Soup of the Day: No Soup."
Prediction: Serious
Label: Humorous

Text: My mom always told me I wouldn’t accomplish anything by lying in bed all day. But look at me now, ma! I’m saving the world!
Prediction: Humorous
Label: Humorous

Text: If I keep stress-eating at this level, the buttons on my shirt will start socially distancing from each other.
Prediction: Humorous
Label: Humorous

Text: To help prevent the spread of COVID-19, everyone should wear a mask in public.
Prediction: Serious
Label: Serious

Text: Avoid close contact with people who are sick.
Prediction: Humorous
Label: Serious

Text: What did one support vector say to another support vector? I feel so marginalized.
Prediction: Humorous
Label: Humorous
```

### Bernoulli NB
```
Text: When the prosecuting attorney asked Gucci Mane if he was guilty he said, 'bitch I might be.'
Prediction: Humorous
Label: Humorous

Text: Camus says that when you wish yourself into the future you commit suicide by wishing yourself closer to your death.
Prediction: Humorous
Label: Serious

Text: How did the random variable get into the club? By showing a fake i.i.d.
Prediction: Humorous
Label: Humorous

Text: Laugh it up! Humor is universal across human cultures — and fuels psychological research on everything from social perception to emotion
Prediction: Serious
Label: Serious

Text: Saddest 6 word story: "Soup of the Day: No Soup."
Prediction: Humorous
Label: Humorous

Text: My mom always told me I wouldn’t accomplish anything by lying in bed all day. But look at me now, ma! I’m saving the world!
Prediction: Humorous
Label: Humorous

Text: If I keep stress-eating at this level, the buttons on my shirt will start socially distancing from each other.
Prediction: Humorous
Label: Humorous

Text: To help prevent the spread of COVID-19, everyone should wear a mask in public.
Prediction: Serious
Label: Serious

Text: Avoid close contact with people who are sick.
Prediction: Humorous
Label: Serious

Text: What did one support vector say to another support vector? I feel so marginalized.
Prediction: Humorous
Label: Humorous
```

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
[amueller.github.io/word_cloud](https://amueller.github.io/word_cloud/auto_examples/masked.html#sphx-glr-auto-examples-masked-py)


[Back to Top](#Table-of-Contents)