from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as se
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['GET', 'POST'])
def streambyte():
    speaker = request.form.get("speaker1")
    data=pd.read_csv('mytext.csv')
    cv = CountVectorizer()
    data_fit=cv.fit_transform(data['Chat'])
    data_fit=data_fit.toarray()
    data_out=data['Speaker']
    data_out.value_counts().plot.bar()
    from sklearn.model_selection import train_test_split
    train_x,test_x,train_y,test_y=train_test_split(data_fit,data_out,test_size=0.30,random_state=0)
    model_nb=GaussianNB()
    model_nb.fit(train_x,train_y)
    model_rf=RandomForestClassifier(n_estimators=50,random_state=0)
    model_rf.fit(train_x,train_y)
    model_dt=tree.DecisionTreeClassifier()
    model_dt.fit(train_x,train_y)
    
    j=cv.transform([speaker]).toarray()
    
    preds=model_dt.predict(j)
    
    result= preds[0]
    
    if(result==1):
        prediction_label = "Customer"
    else:
        prediction_label = "Agent"    

    return render_template('index.html', the_speaker = prediction_label)


if __name__ == '__main__':
    app.run(debug=True,port=8000)