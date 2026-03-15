import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv("Crop_recommendation.csv")
df = pd.DataFrame(data)
x = df.drop(columns=['label'])
y = df['label']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = RandomForestClassifier()
model.fit(x_train,y_train)
pickle.dump(model, open('model.pkl', 'wb'))
