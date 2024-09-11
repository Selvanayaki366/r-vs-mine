import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

sonar_data=pd.read_csv('/kaggle/input/rock-vs-mine-prediction/Copy of sonar data.csv', header=None)

sonar_data

sonar_data.head(5)

# number of rows and clumns 
sonar_data.shape

sonar_data.describe() # statstical measures of the data

sonar_data[60].value_counts()
# to fine rock and mine 

sonar_data.groupby(60).mean()

# separating data and labels
x= sonar_data.drop(columns=60,axis=1)
y= sonar_data[60]

print(x)
print(y)

x_train, x_test, y_train, y_test=train_test_split(x , y, test_size=0.1, stratify=y, random_state=1)

print(x.shape, x_train.shape, x_test.shape)

print(x_train)
print(y_train)

model = LogisticRegression()

model.fit(x_train, y_train)

# accuracy on trainig data 

x_train_prediction = model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction, y_train)

print('Accuracy on training data :',training_data_accuracy )

print('Accuracy on test data :',test_data_accuracy )

input_data =(0.0335,0.0134,0.0696,0.1180,0.0348,0.1180,0.1948,0.1607,0.3036,0.4372,0.5533,0.5771,0.7022,0.7067,0.7367,0.7391,0.8622,0.9458,0.8782,0.7913,0.5760,0.3061,0.0563,0.0239,0.2554,0.4862,0.5027,0.4402,0.2847,0.1797,0.3560,0.3522,0.3321,0.3112,0.3638,0.0754,0.1834,0.1820,0.1815,0.1593,0.0576,0.0954,0.1086,0.0812,0.0784,0.0487,0.0439,0.0586,0.0370,0.0185,0.0302,0.0244,0.0232,0.0093,0.0159,0.0193,0.0032,0.0377,0.0126,0.0156)

# changing the input_data to a numpy array

input_data_as_numpy_array=np.asarray(input_data)

# reshape the  np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a mine')
