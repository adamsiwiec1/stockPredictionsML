from datetime import datetime
from pathlib import Path
import pandas as pd
from predict import enter_stock,prep_data, prep_model, peek, plot_raw_hist,test_models,\
    predict_dtr, predict_dtr_plot, predict_dtr_csv, predict_dtr_csv_linechart  # predict_knn, predict_cart_and_knn,
from mypackages import stock2csv
import os

ticker = "nok"
tickerList = ['aapl','gme','nok', 'plyz','tsla','nakd']

days = 90

filePath = "Z:\\test"


# Single prediction
def predict(ticker):
        data = enter_stock(ticker, datetime(2000, 1, 1), datetime(2021, 2, 14))
        prices = prep_data(data)
        X, Y, X_train, X_validation, Y_train, Y_validation = prep_model(prices)
        # Show models & probability - currently using CART
        test_models(X_train, Y_train)
        predict_dtr_plot(ticker, X, Y, X_train, Y_train, 90)


# Prediction loop
def predict_list(tickers):
    for ticker in tickers:
        data = enter_stock(ticker, datetime(2000, 1, 1), datetime(2021, 2, 14))
        prices = prep_data(data)
        X, Y, X_train, X_validation, Y_train, Y_validation = prep_model(prices)
        predict_dtr_plot(ticker, X, Y, X_train, Y_train, days)


# Number predictions to csv
def predict_to_csv(tickers, filePath):
    for ticker in tickers:
        data = enter_stock(ticker, datetime(2000, 1, 1), datetime(2021, 2, 14))
        prices = prep_data(data)
        X, Y, X_train, X_validation, Y_train, Y_validation = prep_model(prices)
        predictions, meansqError = predict_dtr_csv(X,Y,X_train,Y_train, days)


        error = [meansqError,'']
        error = pd.DataFrame(error)

        # Convert/add to csv
        predictions = pd.DataFrame(predictions)
        # predictions.append(meansqError)
        predictions.columns = [f'{ticker}']

        # Add error column to our dataframe
        predictions.insert(1, 'Mean sq. Error', error)

        # Make directory if it doesnt exist
        if not os.path.exists(Path(filePath)):
            os.makedirs(Path(filePath))
            
        # Create a folder for each ticker - i comment this out if i dont need it - but u need to refactor
        os.makedirs(Path(filePath+f"\\{ticker}"))
        tempPath = filePath+f"\\{ticker}"

        print(predictions)
        length = len(tempPath)
        if tempPath[length-1] == '\\':
            predictions.to_csv(tempPath+ticker+".csv")
        elif tempPath[length-1] != '\\':
            predictions.to_csv(tempPath+"\\"+ticker+".csv")


# def predict_to_excel_wlinechart(tickers, filePath):
#     for ticker in tickers:
#         data = enter_stock(ticker, datetime(2000, 1, 1), datetime(2021, 2, 14))
#         prices = prep_data(data)
#         X, Y, X_train, X_validation, Y_train, Y_validation = prep_model(prices)
#         predict_dtr_csv_linechart(ticker,X,Y,X_train,Y_train, days, filePath)
#
#
#
# for ticker in tickerList:
#     predict_to_excel_wlinechart(ticker,filePath)
# predict_to_csv(tickerList,filePath)
# predict_list(tickerList)

# predict('azn')
# predict('nok')

# Extras
# predict_knn(X,Y,X_train,Y_train,X_validation,Y_validation,60)
# predict_cart_and_knn(X,Y,X_train,Y_train,X_validation,Y_validation,60)
# plot_raw_hist(X,Y)
# test_models(X_train, Y_train)

#     with open(filePath+ticker+".csv") as row:
#         for d in dfrange:
#             row.write(d)