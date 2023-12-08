"import required libraries"

import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt



"Fake News Classifier"
"Dataset: https://www.kaggle.com/c/fake-news/data#"



"download stopwords from nltk"
nltk.download('stopwords')



"create a TextClassifier class"
class TextClassifier:
    def __init__(self, data_file):
        # read the input data file
        self.data_file = data_file
        self.df = pd.read_csv(self.data_file)
        # set stopwords and stemmer for text cleaning
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.porter.PorterStemmer()
        # create a TfidfVectorizer object for feature extraction
        self.vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
        # create a MultinomialNB object for classification
        self.classifier = RandomForestClassifier(max_depth=200, random_state=0)  
        # initialize variables for training and testing data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # initialize variable for accuracy score
        self.score = None
        
    def clean_text(self, text):
        # remove all non-alphabetic characters
        text = re.sub('[^a-zA-Z]', ' ', text)
        # convert all text to lowercase
        text = text.lower()
        # tokenize the text
        text = text.split()
        # remove stopwords and stem the remaining words
        text = [self.stemmer.stem(word) for word in text if not word in self.stopwords]
        # join the stemmed words to form cleaned text
        text = ' '.join(text)
        # return the cleaned text
        return text
        
    # method for preprocessing data
    def preprocess_data(self):
        # drop all rows with missing values
        self.df = self.df.dropna()
        # apply text cleaning to the 'text' column and store the cleaned text in a new 'cleaned_text' column
        self.df['cleaned_text'] = self.df['text'].apply(lambda x: self.clean_text(x))
        # extract features from the cleaned text using TfidfVectorizer
        self.X = self.vectorizer.fit_transform(self.df['cleaned_text']).toarray()
        # store the labels in a separate y variable
        self.y = self.df['label']
        # split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=0)
        
    # method for training the classifier
    def train(self):
        # fit the classifier on the training data
        self.classifier.fit(self.X_train, self.y_train)
        
    # method for evaluating the classifier
    def evaluate(self):
        # predict the labels of the test data
        pred = self.classifier.predict(self.X_test)
        # calculate the accuracy score
        self.score = metrics.accuracy_score(self.y_test, pred)
        # print the accuracy score
        print("accuracy", self.score)
        # confusion_matrix_plot = metrics.confusion_matrix(self.y_test, pred)
        # cmap=plt.cm.Blues
        # plt.imshow(confusion_matrix_plot, interpolation='nearest', cmap=cmap)
        


# Usage
if __name__ == "__main__":
    classifier = TextClassifier('train1.csv')
    classifier.preprocess_data()
    classifier.train()
    classifier.evaluate()
