from datetime import datetime
import requests
import pandas as pd
import io

# CSV with all stocks
url="https://pkgstore.datahub.io/core/nasdaq-listings/nasdaq-listed_csv/data/7665719fb51081ba0bd834fde71ce822/nasdaq-listed_csv.csv"
s = requests.get(url).content
companies = pd.read_csv(io.StringIO(s.decode('utf-8')))

# Set the location here for your csv download - mine is ~/project/csv/allstocks.csv
companies.to_csv('csv/allstocks')