from datetime import datetime, timedelta, date
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
from matplotlib.ticker import MaxNLocator


# Properties
timestamps = []


# *** Step 1. Pull the data *** #
def enter_stock(ticker, startDate, endDate):
    data = yf.download(ticker, startDate, endDate, interval='1d')
    return data

def create_timestamps(stockData, dateArray):
    for x in range(len(dateArray)):
        stringTime = (str(stockData['Date'][x]))
        stringTime = stringTime.split(' ')
        timestamp = time.mktime(datetime.strptime(stringTime[0], "%Y-%m-%d").timetuple())
        timestamps.append(timestamp)


# *** 2. Prepare the data *** # (We are grabbing 'Open' price)
def prep_data(rawData):
    stockData = rawData[rawData.columns[0:1]]
    stockData.reset_index(level=0, inplace=True)
    dateArray = pd.to_datetime(stockData['Date'])
    create_timestamps(stockData, dateArray)
    stockData['Time'] = timestamps  # Add timestamps to our data
    stockData = stockData.drop(['Date'], axis=1)  # We drop date now that we have timestamp
    return stockData


# *** 2b. Prepare our model *** #
def prep_model(prices):
    dataset = prices.values
    X = dataset[:,1].reshape(-1,1)
    Y = ravel(dataset[:,0:1])  # Ravel changes it from vector to 1D array
    validation_size = 0.15
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    return X,Y,X_train,X_validation,Y_train,Y_validation


def test_models(X_train,Y_train):
    num_folds = 10
    seed = 7
    scoring = "r2"

    # Check Algorithms
    models = []
    models.append((' LR ', LinearRegression()))
    models.append((' LASSO ', Lasso()))
    models.append((' EN ', ElasticNet()))
    models.append((' KNN ', KNeighborsRegressor()))
    models.append((' DTR ', DecisionTreeRegressor()))
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


def predict_dtr(X, Y, X_train, Y_train, X_validation, Y_validation, daysToPredict):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(daysToPredict)]
    predictTimestampList = []  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        stringTime = (str(dt))
        predictTimestampList.append(stringTime)
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

    print(len(predictions))
    leng = len(predictions)
    count = [0,0]
    for predict in predictions:
        count[0]+=1
        if count[0] > leng - daysToPredict:
            count[1]+=1
            print(f'Prediction ({predictTimestampList[count[1]-1]}) = ' + str(predict))

    # Final step - create and show the graph.
    plt.figure(figsize=(50, 25))
    tempLeng = len(predictTimestampList)
    temp = []
    count = 0
    for leng in range(tempLeng):
        count+=1
        temp.append(count)

    # plt.yticks(temp)
    # plt.plot(X, Y)
    # plt.plot(predictTimestampList, predictions[(5313-60):5313])
    plt.ylabel('Price')
    plt.xlabel('Time (Days)')
    plt.yscale('linear')
    plt.xlabel(predictTimestampList)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(predictTimestampList, predictions[(5313-daysToPredict):5313]) # Improvement

    # format x-axis (time)
    plt.xticks(predictTimestampList, temp)
    plt.rcParams.update({'font.size': 1})

    plt.show()

    return predictions


# Extra methods:
# ** Plot a graph of two variables - no predictions ** #
def plot_raw_hist(X,Y):
    plot = plt.plot(X,Y,'r')
    plt.setp(plot, 'color', 'r', 'linewidth', 0.5)
    plt.figure(1)
    plt.show()


# ** Peek prices ** #
def peek(prices):
    print(prices)

