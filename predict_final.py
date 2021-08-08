import time
from datetime import datetime, timedelta, date
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ravel
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import warnings
# *** 1. Download the data *** #
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import UndefinedMetricWarning


# Used to avoid unwanted warnings. All warnings have been accounted for.
def warn(*args, **kwargs):
    pass


# Turn off chained assignment warning - we add our timestamps using chained assignments. No harm in this scenario.
pd.options.mode.chained_assignment = None  # default='warn'
warnings.warn = warn


# *** Step 1a. Pull Stock Data from yfinance *** #
def enter_stock(ticker, start_date, end_date):
    data = yf.download(ticker, start_date, end_date, interval='1d')
    return data


# *** Step 1b. Prepare the data *** # (We are grabbing 'Open' price)
def prep_data(raw_data):
    stock_df = raw_data[raw_data.columns[0:1]]              # Convert Raw Data to Stock dataframe
    stock_df.reset_index(level=0, inplace=True)             # Reset index to align our dataframe
    date_array = pd.to_datetime(stock_df['Date'])           # Create Timestamps
    timestamps = create_timestamps(stock_df, date_array)
    stock_df['Timestamp'] = timestamps                      # Add timestamps to our dataframe - this
    stock_df = stock_df[['Open', 'Timestamp', 'Date']]      # Reorder columns
    return stock_df


# *** Step 1c. Convert our dates to timestamps *** #
def create_timestamps(stock_data, date_array):
    timestamps = []
    for x in range(len(date_array)):
        str_time = (str(stock_data['Date'][x]))
        str_time = str_time.split(' ')
        timestamp = time.mktime(datetime.strptime(str_time[0], "%Y-%m-%d").timetuple())
        timestamps.append(timestamp)
    return timestamps


# *** 2. Prepare our model *** #
def prep_model(stock_df):
    dataset = stock_df.values
    x = dataset[:, 1].reshape(-1, 1)            # Ravel is used to change multi-dimensional array
    y = ravel(dataset[:, 0:1]).reshape(-1, 1)   # into a contiguous flattened array
    validation_size = 0.20
    seed = 7
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)
    return x, y, x_train, x_validation, y_train, y_validation


# *** 3. Test Models *** #
def test_models(ticker, x_train, y_train):
    num_folds = 100
    seed = 7
    scoring = "r2"
    models = [(' LR ', LinearRegression()), (' LASSO ', Lasso()), (' EN ', ElasticNet()),
              (' KNN ', KNeighborsRegressor()), (' DTR ', DecisionTreeRegressor())]
    results = []
    names = []
    print(f"Model results for {ticker}")
    for name, model in models:  # ex: name = 'LR', model = LinearRegression()
        k_fold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())  # Retrieve Mean/SD for each of our results
        print(msg)


def predict_dtr_plot(ticker, x, y, x_train, y_train, days_predict, file_path):
    base = date.today()
    dates = [base + timedelta(days=x) for x in range(days_predict)]     # Get Prediction Dates
    predict_date_list = []                      # Used to plot timestamps - need to be real legible dates.
    # Convert Dates to Timestamp
    for dt in dates:
        predict_date_list.append((str(dt)))
        timestamp = time.mktime(datetime.strptime((str(dt)), "%Y-%m-%d").timetuple())
        np.append(x, int(timestamp))
    model = DecisionTreeRegressor()             # Define model - DTR worked best for most stocks.
    # model = KNeighborsRegressor
    model.fit(x_train, y_train)                 # Fit to model
    predictions = model.predict(x)              # predict
    print(len(predictions))                     # Print length of dataset
    count = 0
    for predict in predict_date_list[-days_predict:]:  # Grab the last {day_predict} number of prices
        print(f'{ticker.upper()} Prediction - {predict_date_list[count]} = ' + str(predict))

    metrics = input("Would you like to see the metrics behind this prediction? (y/n) ")
    if metrics == 'y':
        print(f"{ticker.upper()} metrics:")
        print(f"Mean Sq. Error: {str(mean_squared_error(y, predictions))}")
        r2 = r2_score(y, predictions)
        print(f"R-squared value: {str(r2)}")
        # Scikit learn does not have adjusted r2 built in - must do it manually
        n = y.shape[0]
        k = x_train.shape[1]
        adj_r_sq = 1 - (1 - r2) * (n - 1) / (n - 1 - k)
        print(f"Adjusted R-squared {adj_r_sq}")
        input("Enter any key to continue...")
    else:
        print("Ok, not showing metrics. ")

    # Final step - create and show the graph
    plt.cla()                           # Clear old plot
    plt.clf()                           # Clear old figure
    predictions=predictions[-days_predict:]
    plt.figure(figsize=(20, 17))
    plt.plot(predict_date_list, predictions)
    plt.title(str(ticker))
    plt.ylabel('Price', fontsize=12)

    # I use slice notation for the ticks - ex: a[start_index:end_index:step]
    plt.yticks(predictions[::10])
    plt.xticks(predict_date_list[::7])  # Label counting each week
    plt.grid(True)                      # Show grid on plot

    if file_path:                       # Save plot image
        try:
            plt.savefig(f"{file_path}/{ticker}.png")
            print(f"\nPlot image is located at: {file_path}/{ticker}.png")
        except Exception as e:
            print(f"There was an error exporting the plot image for {ticker}. (Error: {e})")

    plt.show()                          # Show plot
    return predictions, predict_date_list
