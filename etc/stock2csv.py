from pip._vendor.colorama import Fore
try:
    import yfinance as yf
except Exception as e:
    print(e)
    print(Fore.RED + 'Failed to import yfinance.')
    exit(0)
try:

    from pathlib import Path
    import os.path
    from os import path
except Exception as e:
    print(Fore.RED + 'Failed to import operating system.')
try:
    from pip._vendor.distlib.compat import raw_input
    from datetime import datetime, date
except Exception as e:
    print(Fore.RED + 'Failed to import packages.')


class SearchList:
    def __init__(self, startDate=None, endDate=None, path=None):
        self.stocks = []
        self.startDate = startDate
        self.endDate = endDate
        self.path = path

    def append_stock(self, stock):
        self.stocks.append(stock)


def create_csv(ticker, start_date, end_date, filePath):
    data = yf.download(ticker, start_date, end_date)
    filePath = str(filePath)

    # Check if user gave \ at the end of filepath or not
    length = len(filePath)

    if filePath[length-1] == '\\':
        data.to_csv(str(filePath) + f'{ticker}.csv')
    elif filePath[length-1] != '\\':
        data.to_csv(filePath+f'\\{str(ticker)}.csv')
    elif filePath[length-1] and filePath[length-2] and filePath[length-3] == ' ':
        # I can parse them out later too lazy to handle all those excepts rn
        print(Fore.RED + "Please do not add spaces after your file path.")
        print("Exiting program....")
        exit(0)


def run():

    # Create an object of search list class
    searchList = SearchList()

    print("Enter the stocks you would like to pull data from and type -1 to exit")
    userInput = input("Enter stock:")

    while userInput != "-1":
        searchList.append_stock(userInput)
        userInput = input("Enter stock:")

    print("Enter the time range to pull data for.")
    escape = '0'
    while escape != "-1":
        startYear = raw_input("Start year:")
        while len(startYear) != 4 or int(startYear) > 2021:
            print("Please enter a valid start year.")
            startYear = raw_input("Start year:")
        startMonth = raw_input("Start month:")
        while len(startMonth) > 2 or int(startMonth) > 12:
            print("Please enter a valid start month.")
            startMonth = raw_input("Start month:")
        startDay = raw_input("Start day:")
        while len(startDay) > 2 or int(startDay) > 31:
            print("Please enter a valid start day.")
            startDay = raw_input("Start day:")
        escape = '-1'
        try:
            startYear = int(startYear)
            startMonth = int(startMonth)
            startDay = int(startDay)
            startDate = datetime(startYear, startMonth, startDay)
            searchList.startDate = startDate
        except:
            print("Please enter a correct start date.")
            pass

    escape = '0'
    while escape != "-1":
        endYear = raw_input("End year:")
        while len(endYear) != 4 or int(endYear) < 1900:
            print("Please enter a valid end year.")
            endYear = raw_input("End year:")
        endMonth = raw_input("End month:")
        while len(endMonth) > 2 or int(endMonth) > 12:
            print("Please enter a valid end month.")
            endMonth = raw_input("End month:")
        endDay = raw_input("End day:")
        while len(endDay) > 2 or int(endDay) > 31:
            print("Please enter a valid end day.")
            endDay = raw_input("End day:")
        escape = '-1'
        try:
            endYear = int(endYear)
            endMonth = int(endMonth)
            endDay = int(endDay)
            endDate = datetime(endYear, endMonth, endDay)
            searchList.endDate = endDate
        except:
            print("Please enter a correct end date.")
            pass

    p = Path(str(input("Enter a filepath to an empty folder:")))
    while not path.exists(p):
        print("Please enter a correct file path")
        p = Path(str(input("Enter a filepath to an empty folder:")))

    searchList.path = p

    for stock in searchList.stocks:
        try:
            create_csv(stock, searchList.startDate, searchList.endDate, searchList.path)
        except PermissionError:
           print(Fore.RED + "Error: Permission Denied. Run as Administrator.")