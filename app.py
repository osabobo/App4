from flask import Flask, request,render_template,url_for
import numpy as np
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app=Flask(__name__)   #first step
pickle_in = open("logit.pkl","rb")
classifier=pickle.load(pickle_in)#step 2
cv_model = open('vectorizer.pkl', 'rb')
cv = joblib.load(cv_model)
#logit=joblib.load('./logit1.pkl')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features =  request.form['url']
    data=[int_features]
    vect =cv.transform(data)

    prediction = classifier.predict(vect)[0]


    return render_template("home.html", prediction_text='It should be {}'.format(prediction))

if __name__=='__main__':
    app.run(debug=True)    #first step without the items in bracket
