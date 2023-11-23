import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

data = pd.read_csv('data/music.csv')
X = data.drop(columns=['genre'])

y = data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, 'music_recommender_model.joblib')
