import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import MertonModel as mm
from scipy import stats
import DownAndOutCall as dao
import matplotlib.pyplot as plt
import MarketDataScraper as yfs
import JumpDiffusionModelClosedForm as jp

yahooFinanceTicker = 'BMPS.MI'

stockTimeSeries = (((yf.Ticker(yahooFinanceTicker).history('5Y', '1d'))['Close']).pct_change().dropna())

zScore = pd.DataFrame(np.abs(stats.zscore(stockTimeSeries))).set_index(stockTimeSeries.index)
z = pd.concat([zScore, stockTimeSeries], axis=1).set_axis(['Z-Score', 'rendimenti'], axis=1)
correctedStockTimeSeries = z[z['Z-Score'] < 0.5]

del (correctedStockTimeSeries['Z-Score'])
minBound = correctedStockTimeSeries.quantile(0.25)
maxBound = correctedStockTimeSeries.quantile(0.75)

# Start from the vector of the "First Guess" Parameters

df = pd.read_excel(
    r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\DBPerAnnoNoNaN.xlsx")

drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

startingParameters = [0.02, 0.01, 0.20, 0.3, 0.30]

def optimal_params(x, mktValues2021, valueOfDebt):
    candidate_prices = jp.jumpDiffusionEquityPricing(valueOfAssets, valueOfDebt, riskFreeRate=x[0], meanOfJump=x[1],
                                                     volOfJump=x[2], lambdaJump=x[3], sigmaAssets=x[4], maturity=T)
    return np.linalg.norm((mktValues2021 - candidate_prices), 2)

T = 1
valueOfDebt = df['Total Liabilities 2019']
valueOfAssets = df['Total Assets 2019']
x0 = startingParameters  # initial guess for algorithm

if (yfs.getValueDebtAverage(yahooFinanceTicker) < 1.5) & (stockTimeSeries.std() >= 0.04):
     bounds = ((minBound, maxBound), (minBound, maxBound), (0, np.inf), (0, 5), (0, np.inf))  # bounds for minimization
else:
     bounds = ((minBound, maxBound), (minBound, maxBound), (0, np.inf), (0, 5), (0,  np.inf))  # bounds for minimization

mktValues2021 = df['Total Equity 2020'].dropna()
res = minimize(optimal_params, method='trust-constr', x0=x0, args=(mktValues2021, valueOfDebt),
               tol=1e-20, bounds=bounds,
               options={"maxiter": 1000})

print('Calibrated drift:', res.x[0])
print('Calibrated mean of Jump Component:', res.x[1])
print('Calibrated volatility Of Jump Component:', res.x[2])
print('Calibrated Lambda:', res.x[3])
print('Calibrated volatility of Assets:', res.x[4])

print('\n')
print('Test on:', yahooFinanceTicker)

debt2018 = yfs.getDebtValue(yahooFinanceTicker)
assets2018 = yfs.getAssetValue(yahooFinanceTicker)
foundametals2018 = yfs.getValueToDebt(yahooFinanceTicker)
estimatedEquity = jp.jumpDiffusionEquityPricing(assets2018[2], debt2018[2], res.x[0], res.x[1], res.x[2], (res.x[3]), res.x[4], 1)
PDestimatedNow = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(foundametals2018[3], res.x[0], res.x[1], res.x[2], (res.x[3]), res.x[4], 1)
value2021 = yfs.getEquityValue(yahooFinanceTicker)[3]

print('\n')
print('Estimated 2021:', estimatedEquity, 'Associated 1Y PD now:', PDestimatedNow * 100, '%')
print('Real Equity 2021:', value2021)
print('Difference:', ((value2021 - estimatedEquity)/value2021) * 100, '%')
