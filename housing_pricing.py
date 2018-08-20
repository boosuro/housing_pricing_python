#importing  libraries from sklearn,numpy,pandas

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston

#load the Boston housing datasets
boston=load_boston()

#print boston to see data
#print(boston)

#creating dataframe from datasets
data_frame_x=pd.DataFrame(boston.data,columns=boston.feature_names)
data_frame_y=pd.DataFrame(boston.target)

#describing the dataset
data_frame_x.describe()

#print result
print(data_frame_x)

#training the regression model
reg_model=linear_model.LinearRegression()

#split dataset into testing and taining datasets
data_frame_x_train,data_frame_x_test,data_frame_y_train,data_frame_y_test=train_test_split(data_frame_x,data_frame_y,test_size=0.2,random_state=4)

#fit the training datasets into model
reg_model.fit(data_frame_x_train,data_frame_y_train)

#calculate the coefficients
reg_model.coef_

#print output
print(reg_model.coef_)

#perform predictions
prediction=reg_model.predict(data_frame_x_test)

print(prediction)
#using index to see data like: prediction[2]
#print data_frame_y_test to see corresponding results

print(data_frame_y_test)

#determine mean square error
print(np.mean((prediction-data_frame_y_test)**2))