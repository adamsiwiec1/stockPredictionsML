import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from predict import enter_stock,prep_data, prep_model, peek, plot_raw_hist,test_models,\
    predict_dtr, predict_dtr_plot, predict_dtr_csv
# from mypackages import stock2csv

dt = datetime.today()

def user_input():
    tickerList = []
    ticker = input("Hello please enter a ticker or -1 to exit: ")
    while ticker != "-1":
        ticker = input("Enter a ticker: ")
        if(ticker != "-1"): tickerList.append(ticker)
    days = input("How many days would you like to predict? (ex: 90)")
    print(f"\n\nOK, predicting {days} days into the future. \n**We start our predictions a week prior from today's date in case the stock market is closed**")
    return tickerList, days

def predict(tickers, days):
    dt = datetime.today() - timedelta(days=int(7)) # week prior
    dt_predict = datetime.today() + timedelta(days=int(90))
    for ticker in tickers:
        data = enter_stock(ticker, datetime(2000, 1, 1), datetime(dt_predict.year, dt_predict.month, dt_predict.day))
        prices = prep_data(data)
        x, y, x_train, x_validation, y_train, y_validation = prep_model(prices)
        predict_dtr_plot(ticker, x, y, x_train, y_train, int(days))


if __name__ == "__main__":
    tickers, days = user_input()
    start = input("Please enter 'y' to continue or 'n' to exit: (y/n)")
    # while start.lower() :
    #     start = input("Error. Please enter 'y' to start and 'n' to exit: (y/n)")
    if start.lower() == 'y':
        predict(tickers, days)
    if start.lower() == 'n':
        print("Ok, exiting the program.")

