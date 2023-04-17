import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import MertonModel as mm
import matplotlib.pyplot as plt
from scipy import stats
import JumpDiffusionModelClosedForm as jp

yahooFinanceTicker = 'AAPL'

stockTimeSeries = (((yf.Ticker(yahooFinanceTicker).history('max', '1d'))['Close']).pct_change().dropna())
zScore = pd.DataFrame(np.abs(stats.zscore(stockTimeSeries))).set_index(stockTimeSeries.index)
z = pd.concat([zScore, stockTimeSeries], axis=1).set_axis(['Z-Score', 'rendimenti'], axis=1)
correctedStockTimeSeries = z[z['Z-Score'] < 0.5]
del (correctedStockTimeSeries['Z-Score'])
minBound = correctedStockTimeSeries.quantile(0.25)
maxBound = correctedStockTimeSeries.quantile(0.75)
maxBoundLambda = correctedStockTimeSeries.max()

# Start from the vector of the "First Guess" Parameters
df = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\BaseDataset\BaseDataset.xlsx")
drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

startingParameters = [0.05, 0.03, 0.3, 0.03, 0.3]

def optimal_params(x, mktValues2021, valueOfDebt, valueOfAssets):

    candidate_prices = jp.jumpDiffusionEquityPricing(valueOfAssets, valueOfDebt, riskFreeRate = x[0], meanOfJump= x[1], volOfJump=x[2], lambdaJump= x[3], sigmaAssets =x[4], maturity = T)

    return np.linalg.norm((mktValues2021 - candidate_prices), 2)

T = 3
valueOfDebt = df['Total Debt 29/09/2018']
valueOfAssets = df['Total Assets 29/09/2018']
x0 = startingParameters  # initial guess for algorithm
bounds = ((minBound, maxBound), (minBound, maxBound), (0, np.inf), (0, 5), (0, np.inf)) #bounds for minimization
mktValues2021 = df['Total Equity 29/09/2021']

res = minimize(optimal_params, method='trust-constr',  x0=x0, args=(mktValues2021, valueOfDebt, valueOfAssets),
                  tol=1e-20, bounds = bounds,
                  options={"maxiter":1000})

#print('\n')
#print(res)
print('Calibrated drift:', res.x[0], '(Starting drift:', startingParameters[0], ')')
print ('Calibrated mean of Jump Component:', res.x[1], '(Starting volatility:', startingParameters[1], ')')
print ('Calibrated volatility Of Jump Component:', res.x[2], '(Starting volatility:', startingParameters[2], ')')
print ('Calibrated Lambda:', res.x[3], '(Starting volatility:', startingParameters[3], ')')
print ('Calibrated volatility of Assets:', res.x[4], '(Starting volatility:', startingParameters[4], ')')

# See how the values compare with the real ones, and plot them

# TEST ON A TICKER

print('\n')
print('Test on:', yahooFinanceTicker)

for valueA in df['Total Assets 29/09/2018'][df['Ticker'] == yahooFinanceTicker]:
    for valueD in df['Total Debt 29/09/2018'][df['Ticker'] == yahooFinanceTicker]:
        a = jp.jumpDiffusionEquityPricing(valueA, valueD, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 3)
        PD = jp.JumpDiffusionProbabilityOfDefaultBis(valueA, valueD, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4], 1)

for value in df['Total Equity 29/09/2021'][df['Ticker'] == yahooFinanceTicker]:
    print('2021 Predicted Equity Value:', a)
    print('2021 Real Equity Value:', value)
    print('Difference:', round(((a - value)/a)*100, 2),'%')

print('1Y % PD of', yahooFinanceTicker, ':', PD*100)



