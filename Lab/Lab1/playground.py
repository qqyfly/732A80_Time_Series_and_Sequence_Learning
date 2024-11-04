# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas  # Loading data / handling data frames
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm  # Used for solving linear regression problems
from sklearn.neural_network import MLPRegressor # Used for NAR model

from tssltools_lab1 import acf, acfplot # Module available in LISAM - Used for plotting ACF

# load the csv file
data = pandas.read_csv('sealevel.csv')

# print the shape of data
print(np.shape(data))

# get the plot data (GMSL and Year)
plot_data = data[['GMSL', 'Year']]

# plot the data(can not specify x and y here)
plt.plot("Year","GMSL",data=plot_data)

# prep the data to train linear regression model
X = data[['Year']]
Y = data[['GMSL']]

# train the model
lr_model = lm.LinearRegression().fit(X, Y)

# predict the data
lr_pred = lr_model.predict(X)

# plot the original data
plt.plot("Year","GMSL",data=plot_data)

# plot the prediction value over the original data
plt.plot(X, lr_pred, label='Linear Regression Fitted Values')
plt.xlabel('Year')
plt.ylabel('GMSL')
plt.legend()
plt.show()

# substract the prediction value from the original data
detrended_y = Y - lr_pred

# make a data frame with the detrended GMSL and the Year values
detrended_data = pandas.DataFrame({'Year': X['Year'], 'GMSL': detrended_y['GMSL']})

# plot the detrended data
plt.plot("Year","GMSL",data=detrended_data)

# we will use the pre processed detrended data here.

# get the training data(1-700)
train_data = detrended_data[0:700] 

# get the validation data(701-end)
validation_data = detrended_data[700:] #

# plot the training data and validation data
plt.plot("Year","GMSL",data=train_data, label='Training Data')
plt.plot("Year","GMSL",data=validation_data,label='Validation Data')
plt.xlabel('Year')
plt.ylabel('GMSL')
plt.legend()
plt.show()