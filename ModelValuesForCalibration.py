import math

import MertonModel as mm
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

dfSP500 = pd.read_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\S&P 500 Components - 4 July 2022.xlsx')

modelValues = list()
for ticker in dfSP500['Ticker'][0:50]:
    if yf.Ticker(ticker).get_balancesheet().empty == False:
       b = mm.MertonModelEquityPricingRealStocks(ticker, 1)
       modelValues.append(b['Value Of Total Equity'][3])
    else:
        modelValues.append(math.nan)

modelBase = pd.concat([dfSP500['Ticker'], pd.Series(modelValues)], axis = 1)

modelBase.to_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Model_Calibration_Base.xlsx')