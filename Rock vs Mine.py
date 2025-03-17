
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#data collection and processing
sonar_data=pd.read_csv('rock vs mine\Copy of sonar data.csv',header=None)
sonar_data.head()
#no of rows and  columns
sonar_data.shape 
sonar_data.describe() #describegives statistics of data
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)
X_train,X_test,Y_train,Ytest = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
print(X.shape,X_train.shape,X_test.shape)
model = LogisticRegression()
model.fit(X_train,Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('Accuracy on training data:',training_data_accuracy)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Ytest)
print('Accuracy on test data:',test_data_accuracy)
input_data=(0.0158,0.0239,0.0150,0.0494,0.0988,0.1425,0.1463,0.1219,0.1697,0.1923,0.2361,0.2719,0.3049,0.2986,0.2226,0.1745,0.2459,0.3100,0.3572,0.4283,0.4268,0.3735,0.4585,0.6094,0.7221,0.7595,0.8706,1.0000,0.9815,0.7187,0.5848,0.4192,0.3756,0.3263,0.1944,0.1394,0.1670,0.1275,0.1666,0.2574,0.2258,0.2777,0.1613,0.1335,0.1976,0.1234,0.1554,0.1057,0.0490,0.0097,0.0223,0.0121,0.0108,0.0057,0.0028,0.0079,0.0034,0.0046,0.0022,0.0021)
#changeing the input data into numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the numpy Array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')