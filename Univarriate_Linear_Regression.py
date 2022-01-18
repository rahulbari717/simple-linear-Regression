'''
Created on Sat Jan  15 09:14:46 2022
@author: RAHUL KAILAS BARI

'''
# %%
#  Import all necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
print("Library are imported")
# %%
# Read Data
df = pd.read_csv('test.csv')
# print(df.columns)
x = df['x'].values
y = df['y'].values
N = len(df['x'])
Nr = int(0.8 * N)
print("Total records = {}, Records for Training = {}".format(N, Nr))

#%%
# df.describe()

# %%
# Defining Function of mean variance covariance m(slope), c(intercept)
# mean
def get_mean(arr):
    return np.sum(arr)/len(arr)

# variance
def get_variance(arr, mean):
    return np.sum((arr-mean)**2)

# covariance
def get_covariance(arr_x, mean_x, arr_y, mean_y):
    final_arr = (arr_x - mean_x)*(arr_y - mean_y)
    return np.sum(final_arr)

def get_coefficients(x, y):
    x_mean = get_mean(x)
    y_mean = get_mean(y)
    m = get_covariance(x, x_mean, y, y_mean)/get_variance(x, x_mean)
    c = y_mean - x_mean*m
    return m, c
# %%
# Regression Function
def linear_regression(x_train, y_train, x_test, y_test):
    prediction = []
    m, c = get_coefficients(x_train, y_train)
    print(" \n Without Library ==> ")
    print("Slope : {}, Intercept: {}".format(m, c))
    for x in x_test:
        y = m*x + c
        prediction.append(y)
    
    r2 = r2_score(y_test, prediction)
    mse = mean_squared_error(prediction, y_test)
    print("The R2 score of the model is: ", r2)
    print("The MSE score of the model is: ", mse)
    return prediction
# %%
# There are 300 samples out of which 240 are for training and 60 are for testing

linear_regression(x[:Nr], y[:Nr], x[Nr:], y[Nr:])
X_train = np.expand_dims(x[:Nr], axis=1)
Y_train = y[:Nr]
X_test = np.expand_dims(x[Nr:], axis=1)
Y_test = y[Nr:]
#%%
print(" By using Sklearn Library ==> ")
reg = LinearRegression().fit(X_train, Y_train)
print("Slope: {}, Intercept: {}".format(reg.coef_, reg.intercept_))
print("Regression Score {}".format(reg.score(X_test, Y_test)))

# %%
#visualize 
sns.jointplot(x='x', y='y', data=df)
# %%
# line of linear regression
sns.jointplot(x='x', y='y', data=df, kind = 'reg')
# %%
# help(reg)
# %%
def plot_reg_line(x, y):
    prediction = []
    m, c = get_coefficients(x, y)
    xMin, xMax = min(x), max(x)
    xa = np.linspace(xMin, xMax, 100)
    for x0 in xa:
        yhat = m*x0 + c
        prediction.append(yhat)
    
    fig = plt.figure(figsize=(20,7))
    plt.subplot(1,2,1)
    sns.scatterplot(x=x, y=y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot between X and Y')
    
    plt.subplot(1,2,2)
    sns.scatterplot(x=x, y=y, color = 'blue')
    plt.plot(xa, prediction, 'ro')
    sns.lineplot(x = xa, y = prediction, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Regression Plot')
    plt.show()
    
plot_reg_line(x, y)
# %%
