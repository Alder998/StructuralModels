# Prepariamo il dataset da cui scaricare

import math
import random
import pandas as pd
import yfinance as yf
import random

dfSP500 = pd.read_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\S&P 500 Components - 4 July 2022.xlsx').sort_values(by='Ticker')

# Dobbiamo lanciare uno scarico per ogni ticker e controllare se le colonne di cui abbiamo bisogno siano presenti nel dataset

colonne = ['Total Stockholder Equity','Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']

presenzaColonne = list()
for ticker in dfSP500['Ticker']:
     stock = yf.Ticker(ticker).get_balancesheet().transpose()

     presenzaColonne.append(pd.Series(colonne).isin(pd.Series(stock.columns)).unique())

print(pd.Series(presenzaColonne))