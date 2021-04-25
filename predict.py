import time
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import ravel
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score


# *** 1. Download the data *** #
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# *** Step 1b. Pull the data *** #
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
    stock_data = raw_data[raw_data.columns[0:1]] # Convert Raw Data to Stock Data
    stock_data.reset_index(level=0, inplace=True)
    date_array = pd.to_datetime(stock_data['Date'])
    timestamps = create_timestamps(stock_data, date_array)
    stock_data['Time'] = timestamps  # Add timestamps to our data
    stock_data = stock_data.drop(['Date'], axis=1)  # We drop date now that we have timestamp
    return stock_data


# *** 2b. Prepare our model *** #
def prep_model(prices):
    dataset = prices.values
    x = dataset[:, 1].reshape(-1, 1)
    y = ravel(dataset[:, 0:1])  # Ravel changes it from vector to 1D array
    validation_size = 0.20  # 0.15
    seed = 7
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)
    return x, y, x_train, x_validation, y_train, y_validation


def test_models(x_train, y_train):
    num_folds = 100
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
        cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        # print(cv_results)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


def predict_dtr(x, y, x_train, y_train, days_predict):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]
    predictTimestampList = []  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        stringTime = (str(dt))
        predictTimestampList.append(stringTime)
        timestamp = time.mktime(datetime.strptime(stringTime, "%Y-%m-%d").timetuple())
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
            print(f'Prediction ({predictTimestampList[count[1]-1]}) = ' + str(predict))


def predict_dtr_csv(x, y, x_train, y_train, days_predict):
    # get prediction dates
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]
    predictTimestampList = []  # Used to display the date of prediction to user

    # convert to time stamp
    for dt in dates:
        stringTime = (str(dt))
        predictTimestampList.append(stringTime)
        timestamp = time.mktime(datetime.strptime(stringTime, "%Y-%m-%d").timetuple())
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
    return predictions[(len(predictions)-(days_predict*2)):len(predictions)], mean_sq_error


def predict_dtr_plot(ticker, x, y, x_train, y_train, days_predict):
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
    # predict_length = len(predictions)

    print(len(predictions))
    length = len(predictions)
    count = [0, 0]
    for predict in predictions:
        count[0] += 1
        if count[0] > length - days_predict:
            count[1] += 1
            print(f'Prediction ({predict_timestamp_list[count[1]-1]}) = ' + str(predict))

    # Final step - create and show the graph.

    pred_length = len(predict_timestamp_list)
    temp = []
    count = 0
    for length in range(length):
        count += 1
        temp.append(count)

    # Clear old plot - for the for loop
    plt.clf()
    plt.cla()
    fig = plt.figure(figsize=(20, 5))
    # plt.yticks(temp)
    # plt.plot(X, Y)
    # plt.plot(predict_timestamp_list, predictions[(5313-60):5313])
    plt.ion()
    plt.title(str(ticker))
    plt.ylabel('Price')
    plt.xlabel('Time (Days)')
    plt.yscale('linear')
    plt.xlabel(predict_timestamp_list)
    ax = plt.figure().gca()
    plt.suptitle(ticker, fontsize=20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(predict_timestamp_list, predictions[(pred_length-days_predict):pred_length])  # Improvement
    plt.grid()
    ax.set_xticklabels(predict_timestamp_list, rotation=80)
    # format x-axis (time)
    plt.xticks(predict_timestamp_list[1::3], temp[1::3])  # This is numpy's slicing
    # plt.xticks(predict_timestamp_list, temp, fontsize=5)
    # ax.xaxis.rcParams.update({'font.size': 5})

    plt.show()

    print("Mean sq. error:" + str(mean_squared_error(y, predictions)))

    return predictions


# Extras
# # Creates an excel graph as well as a csv - not done - using matlab .jpg for now
# def predict_dtr_csv_linechart(ticker,X, Y, X_train, Y_train, daysToPredict, filePath):
#     # get prediction dates
#     base = date.today()
#     dates = [base + timedelta(days=x) for x in range(daysToPredict)]
#     predictTimestampList = []  # Used to display the date of prediction to user
#
#     # convert to time stamp
#     for dt in dates:
#         stringTime = (str(dt))
#         predictTimestampList.append(stringTime)
#         timestamp = time.mktime(datetime.strptime(stringTime, "%Y-%m-%d").timetuple())
#         # to array X
#         np.append(X, int(timestamp))
#
#     # Define model
#     model = DecisionTreeRegressor()
#     # Fit to model
#     model.fit(X_train, Y_train)
#     # predict
#     predictions = model.predict(X)
#     meanSqError = mean_squared_error(Y, predictions)
#
#     # excel line graph
#     wb = Workbook()
#     ws = wb.active
#     sheet_name = ticker
#     writer = pd.ExcelWriter(f'{filePath}\\{ticker}.xlsx', engine='xlsxwriter')
#
#     # create dataframe/merge dates and predictions
#     predictDates_df = DataFrame(dates)
#     predictions_df = DataFrame(predictions)
#     predictDates_df.columns = ['dates']
#     predictDates_df.insert(0, "predictions", predictions_df)
#     df = predictions_df
#     print(df)
#     df.to_excel(writer,sheet_name=sheet_name)
#
#     workbook = writer.book
#     worksheet = writer.sheets[sheet_name]
#
#     chart = workbook.add_chart({'type': 'line'})
#
#     worksheet.insert_chart('D2',chart)
#
#     # z1 = LineChart()
#     # z1.title = ticker
#     #
#     # data = Reference(writer, min_col=1, min_row=1, max_col=daysToPredict, max_row=daysToPredict)
#     # z1.add_data(data)
#
#     # Need to configure series ##########################
#     chart.add_series({'values': f'={sheet_name}!$A$2:$A${daysToPredict}'})
#     # writer.add_chart(z1, "A10")
#     writer.save()
#
#     # We return the same amount of days in the past as the user wants to predict - e.g. 90 daysToPredict returns 180
#     # return predictions[(len(predictions) - daysToPredict:len(predictions)], meanSqError

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

