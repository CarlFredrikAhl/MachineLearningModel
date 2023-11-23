import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv('data/music.csv')
X = data.drop(columns=['genre'])

y = data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

model.dump(model, 'music_recommender_model.joblib')

# Testing so that the trained model works, and it works
'''
trained_model = joblib.load('music_recommender_model.joblib') 
predictions = trained_model.predict([[31, 1]])
print(predictions)
'''
