import os
import time
from datetime import datetime, timedelta, date
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import ravel
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import plotly.express as px

# *** 1. Download the data *** #
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Turn off chained assignment warning - we add our timestamps using chained assignments. No harm in this scenario.
pd.options.mode.chained_assignment = None  # default='warn'


# *** Step 1b. Pull Stock Data from yfinance *** #
def enter_stock(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date, interval='1d')
    return data



# *** Step 1c. Convert our dates to timestamps *** #
def create_timestamps(stock_data, date_array):
    timestamps = []
    for x in range(len(date_array)):
        str_time = (str(stock_data['Date'][x]))
        str_time = str_time.split(' ')
        timestamp = time.mktime(datetime.strptime(str_time[0], "%Y-%m-%d").timetuple())
        timestamps.append(timestamp)
    return timestamps


# *** 2. Prepare the data *** # (We are grabbing 'Open' price)
def prep_data(raw_data):
    stock_df = raw_data[raw_data.columns[0:1]]              # Convert Raw Data to Stock dataframe
    stock_df.reset_index(level=0, inplace=True)             # Reset index to align our dataframe
    date_array = pd.to_datetime(stock_df['Date'])           # Create Timestamps
    timestamps = create_timestamps(stock_df, date_array)
    stock_df['Timestamp'] = timestamps                      # Add timestamps to our dataframe - this
    stock_df = stock_df[['Open', 'Timestamp', 'Date']]      # Reorder columns

    return stock_df


# *** 2b. Prepare our model *** #
def prep_model(prices):
    dataset = prices.values
    x = dataset[:, 1].reshape(-1, 1)
    y = ravel(dataset[:, 0:1])                              # Ravel changes it from vector to 1D array
    validation_size = 0.20  # 0.15
    seed = 7
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)
    return x, y, x_train, x_validation, y_train, y_validation


def test_models(ticker, x_train, y_train):
    num_folds = 100
    seed = 7
    scoring = "r2"

    # Check Algorithms
    models = [(' LR ', LinearRegression()), (' LASSO ', Lasso()), (' EN ', ElasticNet()),
              (' KNN ', KNeighborsRegressor()), (' DTR ', DecisionTreeRegressor())] #, (' SVR ', SVR())]

    # Step 3: Evaluate each model - which one performs the best?
    results = []
    names = []
    print(f"Model results for {ticker}")
    for name, model in models:
        k_fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scoring)
        # print(cv_results)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def predict_dtr(x, y, x_train, y_train, days_predict):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]
    predict_timestamp_list = []  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        string_time = (str(dt))
        predict_timestamp_list.append(string_time)
        timestamp = time.mktime(datetime.strptime(string_time, "%Y-%m-%d").timetuple())
        # to array X
        np.append(x, int(timestamp))

    # Define model
    model = DecisionTreeRegressor()
    # Fit to model
    model.fit(x_train, y_train)
    # predict
    predictions = model.predict(x)
    print(mean_squared_error(y, predictions))

    print(len(predictions))
    length = len(predictions)
    count = [0, 0]
    for predict in predictions:
        count[0] += 1
        if count[0] > length - days_predict:
            count[1] += 1
            print(f'Prediction ({predict_timestamp_list[count[1] - 1]}) = ' + str(predict))


def predict_dtr_csv(x, y, x_train, y_train, days_predict):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]
    predict_timestamp_list = []  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        string_time = (str(dt))
        predict_timestamp_list.append(string_time)
        timestamp = time.mktime(datetime.strptime(string_time, "%Y-%m-%d").timetuple())
        # to array X
        np.append(x, int(timestamp))

    # Define model
    model = DecisionTreeRegressor()
    # Fit to model
    model.fit(x_train, y_train)
    # predict
    predictions = model.predict(x)
    mean_sq_error = mean_squared_error(y, predictions)

    # We return the same amount of days in the past as the user wants to predict - e.g. 90 daysToPredict returns 180
    return predictions[(len(predictions) - (days_predict * 2)):len(predictions)], mean_sq_error


def predict_dtr_plot(ticker, x, y, x_train, y_train, days_predict, filePath):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]
    predict_timestamp_list = []  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        predict_timestamp_list.append((str(dt)))
        timestamp = time.mktime(datetime.strptime((str(dt)), "%Y-%m-%d").timetuple())
        np.append(x, int(timestamp))

    model = DecisionTreeRegressor()                     # Define model - DTR worked best for most stocks.
    model.fit(x_train, y_train)                         # Fit to model
    predictions = model.predict(x)                      # predict

    print(len(predictions))
    length = len(predictions)
    count = [0, 0]
    # prediction_timestamp_2plot = []
    for predict in predictions:
        count[0] += 1
        if count[0] > length - days_predict:
            count[1] += 1
            print(f'Prediction ({predict_timestamp_list[count[1] - 1]}) = ' + str(predict))
            # prediction_timestamp_2plot.append(predict_timestamp_list[count[1] - 1])

    # Final step - create and show the graph
    pred_length = len(predict_timestamp_list)
    temp = []
    count = 0
    for length in range(length):
        count += 1
        temp.append(count)

    plt.cla() # Clear old plot
    plt.clf()
    # predictions = predictions[(pred_length - days_predict):pred_length]
    predictions=predictions[-90:]
    plt.figure(figsize=(20, 5))
    # prediction_dates = np.array(prediction_timestamp_2plot)
    plt.plot(predict_timestamp_list, predictions)
    plt.title(str(ticker))
    plt.ylabel('Price', fontsize=12)

    # I use slice notation for the ticks - ex: a[start_index:end_index:step]
    plt.yticks(predictions[::10])
    plt.xticks(predict_timestamp_list[::10])

    # plt.xlabel('Time (Days)', fontsize=12)
    # plt.yscale('linear')
    # plt.xlabel(predict_timestamp_list)
    # ax = plt.figure().gca()
    # plt.suptitle(ticker, fontsize=20)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Improvement
    plt.grid(True)
    # ax.set_xticklabels(predict_timestamp_list, rotation=80)
    # plt.xticks(predict_timestamp_list[1::3], temp[1::3])  # This is numpy's slicing
    if filePath: # Make directory to store our export data
        try:
            plt.savefig(f"{filePath}/{ticker}.png")
            print(f"Plot image is located at: {filePath}/{ticker}/{ticker}.png")
        except:
            print(f"There was an exporting the plot image for {ticker}.")
    plt.show()

    # print("Mean sq. error:" + str(mean_squared_error(y, predictions)))

    return predictions, predict_timestamp_list


def predict_plotly(ticker, x, y, x_train, y_train, days_predict):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]
    predict_timestamp_list=[]  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        predict_timestamp_list.append((str(dt)))
        timestamp = time.mktime(datetime.strptime((str(dt)), "%Y-%m-%d").timetuple())
        # np.append(x, int(timestamp))

    model = DecisionTreeRegressor()  # Define model - DTR worked best for most stocks.
    model.fit(x_train, y_train)  # Fit to model
    predictions = model.predict(x)
    for x in predictions:
        print(x)

    model=DecisionTreeRegressor()  # Define model - DTR worked best for most stocks.
    model.fit(x_train, y_train)  # Fit to model
    predictions=model.predict(x)  # predict


# Extra methods:
# ** Plot a graph of two variables - no predictions ** #
def plot_raw_hist(x, y):
    plot = plt.plot(x, y, 'r')
    plt.setp(plot, 'color', 'r', 'linewidth', 0.5)
    plt.figure(1)
    plt.show()


# ** Peek prices ** #
def peek(prices):
    print(prices)
