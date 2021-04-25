import os

import pandas as pd
from datetime import datetime, timedelta
from predict2 import enter_stock, prep_data, prep_model, peek, plot_raw_hist, test_models, \
    predict_dtr, predict_dtr_plot, predict_dtr_csv, predict_plotly
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory


def user_input():
    ticker_list = []
    ticker = ""
    while ticker != "-1":
        ticker = input("Enter a ticker: ")
        if ticker != "-1":
            ticker_list.append(ticker)
    days_input = input("How many days would you like to predict? (ex: 90) ")
    print(f"\n\nOK, predicting {days_input} days into the future.")
    return ticker_list, days_input


def predict(list_of_tickers, days_to_predict, csv_bool, png_bool, filepath):
    dt_predict = datetime.today() + timedelta(days=int(days_to_predict))
    for ticker in list_of_tickers:  # We predict using data going back to January 1st, 2000
        data = enter_stock(ticker, datetime(2000, 1, 1), datetime(dt_predict.year, dt_predict.month, dt_predict.day))
        prices = prep_data(data)
        try:

            # Export PART 2 BEGIN#
            x, y, x_train, x_validation, y_train, y_validation = prep_model(prices)
            test_models(ticker, x_train, y_train)
            if png_bool:
                predictions, dates = predict_dtr_plot(ticker, x, y, x_train, y_train, int(days_to_predict), filePath=filepath)
            elif not png_bool:
                predictions, dates = predict_dtr_plot(ticker, x, y, x_train, y_train, int(days_to_predict), filePath=filepath)
            else:
                predictions = None
                dates = None
            if csv_bool and predictions.any() and dates:
                try:
                    # Convert to Numpy Array
                    prices = np.array(predictions)
                    # dates = np.array(dates)
                    data = pd.DataFrame({"Price": prices})
                    data.to_csv(str(filepath) + f"/{ticker}/{ticker}.csv")
                    print(f"Successfully created .csv located at: {filepath}/{ticker}/{ticker}.csv")
                except Exception as e:
                    print(e)
                    print(f"There was an error creating .csv for {ticker}")
            # Export PART 2 END#

        except ValueError as e:
            print(e)
            print(f"Sorry, failed to pull data for {ticker}.")
        except Exception as e:
            print(e)
            print(f"An unknown error occurred trying to pull data for {ticker}.")
        # Test Models - DTR was best for this data.


def run(tickers, days):
    start = input("Please enter 'y' to continue or 'n' to exit: ")
    if start.lower() == 'y':

        # Export PART 1 BEGIN #
        export = input("Would you like to export your predictions to .csv format? (y/n) ")
        if export.lower() == 'y':
            root = Tk()
            root.withdraw()
            path = askdirectory()
            # Export plot image #
            png = input("Would you like to export your plot images? (y/n) ")
            if png.lower() == 'y':
                predict(tickers, days, csv_bool=True, png_bool=True, filepath=path)
            else:       # If they cant enter 'y' correctly they dont need images
                print("OK, we will not export your plot images.")
                predict(tickers, days, csv_bool=True, png_bool=True, filepath=None)
        if export.lower() == 'n':
            print("OK, we will print your predictions in the terminal.")
            predict(tickers, days, csv_bool=False, png_bool=False, filepath=None)
        # Export PART 1 END #

    if start.lower() == 'n':
        print("Ok, exiting the program.")
    elif start.lower() != 'n' and start.lower() != 'y':                 # Recursion to handle incorrect input
        print("Error, incorrect input. Please choose y/n.\n\n")


if __name__ == "__main__":
    tick, d = user_input()
    count = 0
    print("These are the tickers we will predict:\n")
    for t in tick:
        count = count+1
        print(f"{count}. {t}\n")
    run(tickers=tick, days=d)
