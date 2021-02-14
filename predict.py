from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import time
from numpy.core import ravel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
# *** 1. Download the data *** #
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# This method is for the test class
def enter_stock(ticker):
    data = yf.download(ticker, datetime(1994, 1, 1), datetime(2020, 1, 1), interval='1d')

data = yf.download("AZN", datetime(1994, 1, 1), datetime(2020, 1, 1), interval='1d')

# *** 2. Prepare the data *** #
# Grab only 'Open' data - for our test model purposes
prices = data[data.columns[0:1]]
prices.reset_index(level=0, inplace=True)

timestamps = []
tempPrices = [] # Just in case

# Created my timestamps manually - yfinance ones were messed up
dateArray = pd.to_datetime(prices['Date'])
for x in range(len(dateArray)):
    stringTime = (str(prices['Date'][x]))
    stringTime = stringTime.split(' ')
    timestamp = time.mktime(datetime.strptime(stringTime[0], "%Y-%m-%d").timetuple())
    timestamps.append(timestamp)

for price in prices['Open']:
    tempPrices.append(price)

# Take peek
# print(tempPrices)
# print(timestamps)

# Drop date from our dataframe
prices = prices.drop(['Date'], axis=1)

# Peek our data

prices.append(timestamps)
prices['Time'] = timestamps
print(prices['Time'][1])
print(prices)
# *** 2b. Prepare our model *** #
dataset = prices.values
X = dataset[:,1].reshape(-1,1)
Y = ravel(dataset[:,0:1])  # Ravel changes it from vector to 1D array


validation_size = 0.20
seed = 7
plot = plt.plot(X,Y,'r')
plt.setp(plot, 'color', 'r', 'linewidth', 0.5)
plt.figure(1)
plt.show()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

num_folds = 10
seed = 7
scoring = "r2"
scoring2 = 'accuracy'

# Check Algorithms
models = []
models.append((' LR ', LinearRegression()))
models.append((' LASSO ', Lasso()))
models.append((' EN ', ElasticNet()))
models.append((' KNN ', KNeighborsRegressor()))
models.append((' CART ', DecisionTreeRegressor()))
models.append((' SVR ', SVR()))

# Step 3: Evaluate each model - which one performs the best?
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    # print(cv_results)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


dates = ["2020-12-23", "2020-12-24", "2020-12-25", "2020-12-26", "2020-12-27",]
# convert to time stamp
for dt in dates:
    stringTime = (str(dt))
    timestamp = time.mktime(datetime.strptime(stringTime, "%Y-%m-%d").timetuple())
    # to array X
    np.append(X, int(timestamp))



# Define model
model = DecisionTreeRegressor()
# Fit to model
model.fit(X_train, Y_train)
# predict
predictions = model.predict(X)
print(mean_squared_error(Y, predictions))

# %matplotlib inline
fig= plt.figure(figsize=(24,12))
plt.plot(X,Y)
plt.plot(X,predictions)
plt.show()