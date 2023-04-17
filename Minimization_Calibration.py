# Minimization Problem

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import MertonModel as mm
import matplotlib.pyplot as plt
import DownAndOutCall as dao

# Start from the vector of the "First Guess" Parameters
df = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Big Dataset Calibration\CDS Calibration.xlsx")
df = df[df['Asset to Debt Ratio'] > 1]
drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

startingParameters = [0.05, 0.3]

def optimal_params(x, valueOfDebt, valueOfAssets):

    candidate_prices = dao.downAndOutCDSSpread(valueOfAssets, valueOfDebt, riskFreeRate = x[0], sigmaAssets =x[1], maturity = T)

    return np.linalg.norm((mktValues - candidate_prices)/candidate_prices, 2)

T = 5
valueOfDebt = df['Total Market Debt Value']
valueOfAssets = df['Total Assets Value']
x0 = startingParameters  # initial guess for algorithm
bounds = ((-0.10, 0.10), (0, np.inf)) #bounds for minimization
mktValues = df['5Y CDS Spread']

res = minimize(optimal_params, method='trust-constr',  x0=x0, args=(valueOfDebt, valueOfAssets),
                  tol=1e-20, bounds=bounds,
                  options={"maxiter":1000})

#print('\n')
#print(res)
print('Calibrated Drift:', res.x[0], '(Starting drift: ', drift, ')')
print ('Calibrated Volatility:', res.x[1], '(Starting volatility: ', volatility, ')')

# CALIBRATION FINISHED - See how the values compare with the real ones, and plot them

# Values with the starting parameters

jnk1 = mm.MertonModelEquityPricing(9040648000, 1489223000, res.x[0], res.x[1], 3)

print('Predicted Equity in 2021:', jnk1)
print('Actual Equity in 2021:', 63090000000)
print('Difference:', round(((63090000000-jnk1)/63090000000)*100, 2), '%')






