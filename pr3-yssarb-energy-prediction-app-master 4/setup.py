
#Importing our dependencies
from flask import Flask,render_template,url_for,request, jsonify, redirect
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 
import pickle as pkl
import datetime
from flask_pymongo import PyMongo

import os

if os.environ.get('MONGODB_URI'):
	mongo_uri = os.environ.get('MONGODB_URI')
	flask_debug = False
else:
    from dev_config import flask_debug, mongo_uri

#Importing our ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

#Starting our app
app = Flask(__name__)
mongo = PyMongo(app, uri=mongo_uri)

# Use flask_pymongo to set up mongo connection
# app.config["MONGO_URI"] = "mongodb+srv://admin:password123?@cluster0-1mrc0.mongodb.net/predictions?retryWrites=true"
# mongo = PyMongo(app, uri=mongo_uri)

#Loading our ML Model
xg_boost_model = open("models/reg_model.pickle.dat", "rb")
p_enr = joblib.load(xg_boost_model)

#Creating features
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

#Creating return template end-point
@app.route('/')
def index():
	return render_template('index.html')

#Creating Prediction end-point
@app.route('/predict', methods=['POST'])
def predict():
	
	#Receiving the input query from form
	if request.method == 'POST':
    		
		#Striping it down to just the hour (dataset) 
		number_of_hours = int(request.form['number_of_hours']) 
		now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
		
		#Generating a list of datetimes for x hours 
		time_df = pd.date_range(start=now, periods=number_of_hours, freq="H").to_frame()
		user_input = create_features(time_df)

		#Adding the predicition column
		user_input['prediction'] = p_enr.predict(user_input)
		user_input = user_input[["prediction"]]
		
		#Using string format to overwrite the index for the date we want 
		user_input.index = user_input.index.strftime('%Y-%m-%d %H:%M')
		prediction_dictionary = user_input.to_dict()['prediction']
		x = list(prediction_dictionary.keys())
		y = list(prediction_dictionary.values())
		prep_dict = {'x':x, 'y':y}
		
		#Adding our predicted data to our database
		print(prediction_dictionary)
		predictions = mongo.db.predictions
		predictions.update({}, prep_dict, upsert=True)
	return redirect('/#predict', code=302)

#Creating plot end-point
@app.route('/plots')
def plots():
	pred = mongo.db.predictions.find_one({}, {"_id":False})
	print(pred)
	return jsonify(dict(pred))

if __name__ == '__main__':
	app.run(debug=flask_debug)
