import numpy as np
import pandas as pd
from IPython.display import display
import math
import sklearn.preprocessing
import os
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from striprtf.striprtf import rtf_to_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import scipy
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# stems all the words of paragraph
def word_stemmer(words):
    stem_words = [stemmer.stem(o) for o in words]
    return " ".join(stem_words)

# lemmatizes all the words of paragraph
def word_lemmatizer(words):
   lemma_words = [lemmatizer.lemmatize(o) for o in words]
   return " ".join(lemma_words)


# loading the stop words from nltk library 
stop_words = set(stopwords.words('english'))


# the below function processing a given text that is passed to it, The two paragraphs are passed and compared
def text_preprocessing(text):
    text = str(text)
    text = rtf_to_text(text)
    string = ""

    # replace every special char with space
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text)

    # replace html tags with space
    text = re.sub('(<[\w\s]*/?>)', " ", text)

    # Removing all the digits present in the review text
    text = re.sub('\d+', " ", text)

    # replace multiple spaces with single space
    text = re.sub('\s+', ' ', text)

    # converting all the chars into lower case
    text = text.lower()

    for word in text.split():
        # if the word is a not a stop word then retain that word from the data
        if word not in stop_words:
            string += word + " "

    string = word_stemmer(string.split())
    string = word_lemmatizer(string.split())

    return string

def paragraph_comparer(path1, path2, thresh=None):

    with open(path1 , 'r') as infile1:
        content1 = infile1.read()
        processed_1 = text_preprocessing(content1)

    with open(path2 , 'r') as infile2:
        content2 = infile2.read()
        processed_2 = text_preprocessing(content2)

    arr = [processed_1  , processed_2]
    df = pd.DataFrame(arr)
    X = df[0]

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    X_tfidf =X_tfidf.toarray()

    vec1 = X_tfidf[0]
    vec2 = X_tfidf[1]
    cosine = scipy.spatial.distance.cosine(vec1, vec2)

    percent = round((1-cosine)*100,2)

    if thresh is None:
        print('The two paragraphs are similar by', percent,'%')
    elif percent >= thresh:
        print('The two paragraphs are similar')
    else:
        print("The two paragraphs are not similar")

# paragraph_comparer()

def check_test_cases(root, thresh=None):
    for parent in os.listdir(root):
        parent_path = os.path.join(root, parent)
        print(f'For {parent}', end=' ')
        try:
            paragraph_comparer(os.path.join(parent_path, 'para1.txt'), os.path.join(parent_path, 'para2.txt'), thresh)
        except:
            print('Error in file path')

root = './test_cases'

check_test_cases(root, thresh=None)