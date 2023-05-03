import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
texts=pd.read_csv('bbc-text.txt')
texts=texts.rename(columns={'text':'News_Headline'},inplace=False)
texts.head()
texts.category=texts.category.map({'tech':0,'business':1,'sport':2,'entertainment':3,'politics':4})
texts.category.unique()
from sklearn.model_selection import train_test_split
x=texts.News_Headline
y=texts.category
x_train,x_test, y_train, y_test=train_test_split(x,y, train_size=0.6,random_state=1)
stop_words = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stop_words)
vectorizer.fit(x_train)
train_matrix=vectorizer.transform(x_train)
test_matrix = vectorizer.transform(x_test)
classifier = MultinomialNB()
classifier.fit(train_matrix, y_train)
predictions = classifier.predict(test_matrix)
import pickle
import streamlit as slt
saved_model=pickle.dumps(classifier)
s=pickle.loads(saved_model)
slt.header('News text Classifier')
input=slt.text_input('Enter the text')
inp=vectorizer.transform([input]).toarray()
if slt.button("Predict"):
    pre=(str(list(s.predict(inp))[0]).replace('0','TECH').replace('1','BUSINESS').replace('2','SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS'))
    slt.write(pre)
