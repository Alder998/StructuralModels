# Here we will set up the final calibration file, that would allow us to have an as big as possible dataset
# with all the starting data from the debt and the equity value of each company of S&P 500 (at least, the companies
# that have a public income statament)

import pandas as pd
import numpy as np
import yfinance as yf
import math
import MertonModel as mm

# import the general Dataset with all the tickers from S&P 500

dfSP500 = pd.read_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\S&P 500 Components - 4 July 2022.xlsx')

# Get the market Value of Equity

# Extract the Total Equity Value from the companies Balance Sheet
equityPerCompany = list()
for ticker in dfSP500['Ticker'][100:101]:

   if yf.Ticker(ticker).get_balancesheet().empty == False:
       stock = yf.Ticker(ticker).get_balancesheet().transpose()['Total Stockholder Equity'][0]
       equityPerCompany.append(stock)

   else:
       equityPerCompany.append(math.nan)

baseEquity = pd.concat([dfSP500['Ticker'], pd.Series(equityPerCompany)], axis = 1)

# Extract the total debt from the companies Balance Sheet

debtPerCompany = list()
for ticker in dfSP500['Ticker'][100:101]:

   if yf.Ticker(ticker).get_balancesheet().empty == False:
       stock = yf.Ticker(ticker).get_balancesheet().transpose()['Total Liab'][0]
       debtPerCompany.append(stock)

   else:
       debtPerCompany.append(math.nan)

assetValue = pd.Series(equityPerCompany) + pd.Series(debtPerCompany)

base = pd.concat([dfSP500['Ticker'], pd.Series(equityPerCompany), pd.Series(debtPerCompany), pd.Series(assetValue)], axis = 1).set_axis(['Ticker', 'Total Market Equity Value', 'Total Market Debt Value', 'Total Assets Value'], axis = 1)

print(base)
#base.to_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Big Dataset Calibration\Big_dataset_calibration_File_100-200.xlsx')