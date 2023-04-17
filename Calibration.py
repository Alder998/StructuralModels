# Here we try to calibrate our model to the american Market Stock prices, on the S&P 500, taking all the (aggregate)
# equity value of the index companies
import math
import random
import pandas as pd
import yfinance as yf

print('\n')
print('DOWNLOADING STOCK DEBT FROM YAHOO FINANCE API...')
print('\n')

# Import the dataset with all the constituents of S&P 500
dfSP500 = pd.read_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\S&P 500 Components - 4 July 2022.xlsx').sort_values(by='Ticker')

# Extract the Total Equity Value from the companies Balance Sheet
TotalDebtPerCompany = list()

for ticker in dfSP500['Ticker'][24:40]:

   if yf.Ticker(ticker).get_balancesheet().empty == False:

       stock = yf.Ticker(ticker).get_balancesheet()
       stockDebtBase = stock.transpose()[
           ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
       stockDebt = (stockDebtBase['Total Current Liabilities'] - stockDebtBase['Accounts Payable'] - stockDebtBase[
           'Other Current Liab']) + (stockDebtBase['Long Term Debt'])
       TotalDebtPerCompany.append(stockDebt[3])

       # Roungh loading bar to check everything will be ok
       print('Ticker', ticker, 'Scaricato')
       print('Valore:', stockDebt[3])
       print('\n')

   else:
       TotalDebtPerCompany.append(math.nan)
       # same loading bar
       print('Ticker', ticker, 'Inesistente')

base = pd.concat([dfSP500['Ticker'], pd.Series(TotalDebtPerCompany)], axis = 1)

base.to_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Calibration_Base2440.xlsx')