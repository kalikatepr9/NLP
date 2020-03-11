#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:53:44 2019

@author: pranavkalikate
"""

# Natural Language Processing

#Steps:
#1- Clean text
#2- create a bag of words model
#3 - apply ML model(Classification) onto Bag of words

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts to get rid of unneccessary words     #compulsory step in NLP
"""  
#This is for first review..run code line by line to see the results..
import re      #library helps to clean the text efficiently  #alt + tab for white space
import nltk   #library which will help to remove irrelevent words
nltk.download('stopwords')   #all the words in stopwords present in review will be removed
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) #following ^ put things which you dont want to remove
review = review.lower()    #covert to lower case        # ' ' ==removed char will be replaced by space..to avoid sticking of words
review = review.split()  #string into lists
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
"""

import re      
import nltk   
nltk.download('stopwords')   
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  
corpus = []  #corpus is a collection of cleaned text
for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) 
        review = review.lower()    
        review = review.split()  
        ps = PorterStemmer()    
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review) 
        corpus.append(review)  

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer  
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()   #Sparse matrix
y = dataset.iloc[:, 1].values  

# Splitting the dataset into the Training set and Test set       
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
"""

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Accuracy = 72%
#Precision = 85%
#Recall = 54%
#F1 Score = 66%

#Predicting the new results
new_review= "I loved the food"
review = re.sub('[^a-zA-Z]', " ",new_review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
new_review = [new_review]
new_review = cv.transform(new_review).toarray()
new_y_pred = classifier.predict(new_review)
#another review
new_review= " satisfied with the taste of the food"
review = re.sub('[^a-zA-Z]', " ",new_review)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
new_review = [new_review]
new_review = cv.transform(new_review).toarray()
new_y_pred_2 = classifier.predict(new_review)
