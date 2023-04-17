# Here we are implementing the Merton Model, one of the simplest credit risk modelling frameworks, that starts from the option
# valuation Theory.

# The base Assumption of this model is to treat the equity as a call option on the company's asset
# Therefore, it is convenient to excercise the option when the Assets of the company (== seen as the price of the underlying)
# is higher than the debt of the company (== seen as the strike price of an option)

# Therefore, the company patyoff is:
# Max(At - k ; 0)

# Having the same payoff of a call option, we can easily price the equity of the company with a Black-Scholes Analytic formula,
# given all the paramters.

# Following the reasoning of B-S, we are gonna have the base formula, defined as:

# EquityValue = AssetsValue * normCDF(d1) - exp(r*T) * DebtValue * normCDF(d2)

# Where:
# d1 = (ln V/D + (r + 0,5 * sigmaAsset^2) * T) / (sigmaAssets * sqrt(T))
# d2 = d1 - sigmaAssets * sqrt(t)

# Let's now concentrate on inferring the firm Value with the formula above

def assetValueAndVarianceOfAssetsEstimator (startingLevelOfDebt, riskFreeRate, firstGuessSigma, maturity, startingLevelOfEquity):

    import pandas as pd
    import MertonModel as mm
    from sympy.solvers import nsolve
    from sympy import exp, sqrt, ln
    from sympy.stats import Normal
    from sympy.stats import cdf
    from sympy import S
    from sympy import simplify
    import numpy as np

    startValue = startingLevelOfDebt
    obs = 10
    f = list()
    for i in range(obs):
        start = startValue
        inc = np.random.normal(loc=0, scale=0.5, size=1)
        w = (start + inc).cumsum()
        f.append(w.reshape(1, 1))
    a = np.array(f)
    levelOfDebt = (pd.DataFrame(a.reshape(obs, 1)).T).transpose()

    startValueEq = startingLevelOfEquity
    obs = 10
    fEq = list()
    for i in range(obs):
        startEq = startValueEq
        incEq = np.random.normal(loc=0, scale=5, size=1)
        wEq = (startEq + incEq).cumsum()
        fEq.append(wEq.reshape(1, 1))
    aEq = np.array(fEq)
    levelOfEquity = (pd.DataFrame(aEq.reshape(obs, 1)).T).transpose()

    timeSeriesA = list()

    for time in range(len(levelOfDebt[0])):

        A = Normal("A", 0, S(1))

        d = (simplify(cdf((ln(A / levelOfDebt[0][time]) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (
                    firstGuessSigma * sqrt(maturity)))(0)))
        d1 = (simplify(
            cdf(((ln(A / levelOfDebt[0][time]) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (
                        firstGuessSigma * sqrt(maturity))) - (firstGuessSigma * sqrt(maturity)))(0)))

        finalEq = nsolve(A * (
            (cdf((ln(A / levelOfDebt[0][time]) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (
                        firstGuessSigma * sqrt(maturity)))(0))) - levelOfDebt[0][time] * exp(
            -riskFreeRate * maturity) * ((
            cdf(((ln(A / levelOfDebt[0][time]) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (
                        firstGuessSigma * sqrt(maturity))) - (firstGuessSigma * sqrt(maturity)))(0))) - levelOfEquity[0][time],
                         A, (1.2,))
        timeSeriesA.append(finalEq)

    finalSTD = (((pd.Series(timeSeriesA).pct_change())))
    finalSTD = finalSTD[1:len(finalSTD)].std()

    return pd.Series(timeSeriesA)


def KMVParamtersEstimatorOnRealStocks (YahooFinanceTicker, fistGuessVar, riskFreeRate ,maturity, returnType):

    import MertonModel as mm
    import pandas as pd

    GuessVarL = [fistGuessVar]
    obs = 10

    while len(GuessVarL)-1 < obs:
        GuessVar = mm.assetValueAndVarianceOfAssetsEstimatorOnRealStocks(YahooFinanceTicker, riskFreeRate, GuessVarL[len(GuessVarL) - 1], maturity)
        GuessVarL.append(GuessVar)

        if len(GuessVarL) >= 50:
            break

    estimatedVariance = pd.Series(GuessVarL[1:len(GuessVarL)]).mean()

    if returnType == 'volatilty':
       return estimatedVariance

    if returnType == 'Assets':
        return mm.assetValueAndVarianceOfAssetsEstimatorOnRealStocks(YahooFinanceTicker, riskFreeRate, estimatedVariance, maturity, returnType='Assets')



def assetValueAndVarianceOfAssetsEstimatorOnRealStocks (yahooFinanceTicker, riskFreeRate, firstGuessSigma, maturity, returnType = 'volatility'):

    import pandas as pd
    import MertonModel as mm
    from sympy.solvers import nsolve
    from sympy import exp, sqrt, ln
    from sympy.stats import Normal
    from sympy.stats import cdf
    from sympy import S
    from sympy import simplify

    levelOfDebt = mm.getDebtValue(yahooFinanceTicker, TimeSeries=True)

    timeSeriesA = list()
    for levelOfDebtSingle in levelOfDebt:

        A = Normal("A", 0, S(1))

        d = (simplify(cdf((ln(A / levelOfDebtSingle) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (firstGuessSigma * sqrt(maturity)))(0)))
        d1 = (simplify(
            cdf(((ln(A / levelOfDebtSingle) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (firstGuessSigma * sqrt(maturity))) - (firstGuessSigma * sqrt(maturity)))(0)))

        finalEq = nsolve(A * (
        (cdf((ln(A / levelOfDebtSingle) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (firstGuessSigma * sqrt(maturity)))(0))) - levelOfDebtSingle * exp(
            -riskFreeRate * maturity) * ((
            cdf(((ln(A / levelOfDebtSingle) - (riskFreeRate - 0.5 * firstGuessSigma) * maturity) / (firstGuessSigma * sqrt(maturity))) - (firstGuessSigma * sqrt(maturity)))(0))),
                         A, (1.2,))
        timeSeriesA.append(finalEq)

    finalSTD = (((pd.Series(timeSeriesA).pct_change())))
    finalSTD = finalSTD[1:len(finalSTD)].std()


    if returnType == 'volatility':
        return finalSTD

    if returnType == 'Assets':
        return (finalEq[len(finalEq)])

# This method uses the base Merton Formula for the option pricing to compute the value of equity at maturity T
# - valueOfAssets: level of asset (as St in the option framework)
# - valueOfDebt: level of debt of the company (K, i.e. the strike in the option framework)
# - Risk-Free rate: the drift of the dynamics, in Black-Scholes framework
# - sigmaAssets: volatility of assets

def MertonModelEquityPricing (valueOfAssets, valueOfDebt, riskFreeRate, sigmaAssets, maturity):

    # Implementation of Merton Closed form Equation

    import numpy as np
    from scipy.stats import norm

    # definition of d1 and d2

    d1 = (np.log(valueOfAssets/valueOfDebt) + ((riskFreeRate + 0.5 * sigmaAssets**2) * maturity)) / (sigmaAssets * np.sqrt(maturity))

    d2 = d1 - sigmaAssets * np.sqrt(maturity)

    # Create the closed form equation

    closedFormEquation = valueOfAssets * norm.cdf(d1) - np.exp(- riskFreeRate*maturity) * valueOfDebt * norm.cdf(d2)

    return closedFormEquation


def MertonModelEquityPricingRealStocks (yahooFinanceTicker, maturity, plot = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import yfinance as yf

    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    fundamentals = stock.transpose()[['Total Assets', 'Total Liab']]

    companyDefaultTimeSeries = list()
    for i in range(0, len(fundamentals.index)):
        jnk = mm.MertonModelEquityPricing(fundamentals['Total Assets'][i], fundamentals['Total Liab'][i], drift, volatility, maturity)
        companyDefaultTimeSeries.append(jnk)

    observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']
    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Value Of Total Equity'], axis=1)
    defaultDataset = defaultDataset.sort_values(by='Date', ascending=True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    if plot == True:
        plt.figure(figsize=(12, 5))
        plt.scatter(x=defaultDataset.index, y=defaultDataset)
        plt.plot(defaultDataset)
        plt.show()

    return defaultDataset


    return defaultDataset


# In Merton framework, the Probability of Default is defined as the probability of the option to expire out-of the money,
# therefore we define it as the normal CDF of the d1 term

def MertonProbabilityOfDefault (valueOfAssets, valueOfDebt, riskFreeRate, sigmaAssets, maturity, showDD = False):

    import numpy as np
    from scipy.stats import norm
    import pandas as pd

    distanceToDefault = (np.log(valueOfAssets/valueOfDebt) + ((riskFreeRate + 0.5 * sigmaAssets**2) * maturity)) \
                        / (sigmaAssets * np.sqrt(maturity))

    probabilityOfDefault = norm.cdf( - distanceToDefault)

    if showDD == False:
        return probabilityOfDefault

    if showDD == True:
        return (pd.concat([pd.Series(distanceToDefault), pd.Series(probabilityOfDefault)], axis = 1)).set_axis(['Distance to Default',
                                                                                                'Probability of Default'], axis = 1)


def MertonProbabilityOfDefaultWithValueToDebt (valueToDebtRatio, riskFreeRate, sigmaAssets, maturity, showDD = False):

    # To have one single parameter could be way more practical in implementation. Therefore, when it is possible,
    # The PD formulas has been implemented as a function of just the Value/Debt Ratio. This is one of the cases.

    import numpy as np
    from scipy.stats import norm
    import pandas as pd

    # The "distance to default" parameter is nothing but the d1 parameter in the closed-form Merton relation

    distanceToDefault = (np.log(valueToDebtRatio) + ((riskFreeRate + 0.5 * sigmaAssets**2) * maturity)) \
                        / (sigmaAssets * np.sqrt(maturity))

    # The Probability of Default is defined as the normal CDF of (- distance to default)

    probabilityOfDefault = norm.cdf( - distanceToDefault)

    # In the fist implementation, it was included a function to show the value of the distance to default together with the PD

    if showDD == False:
        return probabilityOfDefault

    if showDD == True:
        return (pd.concat([pd.Series(distanceToDefault), pd.Series(probabilityOfDefault)], axis=1)).set_axis(
            ['Distance to Default', 'Probability of Default'], axis=1)



# let's select a real stock from the Market and compute its probability of Default

def MertonProbabilityOfDefaultRealStocks (yahooFinanceTicker, maturity, parameters = 'MM', plot = False, calibrate = False, freq = 'yearly'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import MarketDataScraper as mkt
    import yfinance as yf

    # if it is decided to calibrate the model, the function mertonAlternativeCalibration is called. It is called Alternative because
    # in the first time I thought it was better to calibrate Merton on CDS, and I actually did it. However, the scarcity and the realiability
    # of the available data led to this change.

    if calibrate == True:
        params = mm.mertonAlternativeCalibration(yahooFinanceTicker)
        drift = params[0]
        sigmaAsset = params[1]

        print('Calibrated Drift:', params[0])
        print('Calibrated Volatility:', params[1])

    else:
        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        sigmaAsset = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()


    # "MM" stands for Modigliani-Miller. However, Modigliani and Miller have nothing to do with this notation, since
    # it is just a personal reminder to use Total Asset and total Liabilities

    if parameters == 'MM':


        if freq == 'yearly':
            stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
        if freq == 'quarterly':
            stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

        stockM = stock.transpose()[['Total Assets', 'Total Liab']]

        stockValueToDebt = stockM['Total Assets'] / stockM['Total Liab']


    # This is the implementation of the BhSh parameters explained in Chapter 2

    if parameters == 'BhSh':

        if freq == 'yearly':
            stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
        if freq == 'quarterly':
            stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = drift
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stockEquity + stockDebt
        sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

        stockValueToDebt = stockAssets / stockDebt

    # For each value of the available Debt and Asset Series, aggregated in Value to Debt Ratio, the relative PD is computed,
    # stored, organized, showed in the final plot

    companyDefaultTimeSeries = list()
    for value in range(len(stockValueToDebt)):

        if parameters == 'BhSh':

            if calibrate == True:
                jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt[value], drift, sigmaAsset, maturity)
            else:
                jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt[value], drift, sigmaAssetBhSh[value], maturity)

        if parameters == 'MM':

            if calibrate == True:
                jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt[value], drift, sigmaAsset, maturity)
            else:
                driftMM = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
                sigmaAssetMM = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()
                jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt[value], driftMM, sigmaAssetMM, maturity)

        companyDefaultTimeSeries.append(jnk * 100)

    # The following lines of code are just for Output Organization and Plotting

    if freq == 'yearly':
        observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']

    if freq == 'quarterly':
        observations = ['2022-09-25', '2022-06-26', '2022-03-25', '2021-12-25']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by = 'Date', ascending = True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    if plot == True:
        plt.figure(figsize=(12, 5))
        plt.scatter(x = defaultDataset.index, y = defaultDataset)
        plt.plot(defaultDataset)
        plt.show()

    return defaultDataset


def MertonProbabilityOfDefaultRealStocks2021 (yahooFinanceTicker, maturity, parameters = 'MM', calibrate = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import MarketDataScraper as mkt
    import yfinance as yf

    if calibrate == True:
        params = mm.mertonAlternativeCalibration(yahooFinanceTicker)
        drift = params[0]
        sigmaAsset = params[1]

        print('Calibrated Drift:', params[0])
        print('Calibrated Volatility:', params[1])

    else:
        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        sigmaAsset = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    if parameters == 'MM':

        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()

        stockM = stock.transpose()[['Total Assets', 'Total Liab']]

        stockValueToDebt = stockM['Total Assets'] / stockM['Total Liab']


    if parameters == 'BhSh':

        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet().transpose().reset_index()
        stock = stock.transpose()
        stock = stock[0]

        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]

        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = drift#((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stockEquity + stockDebt
        sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

        stockValueToDebt = stockAssets / stockDebt


    if parameters == 'BhSh':

        if calibrate == True:
            companyDefaultTimeSeries = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt, drift, sigmaAsset, maturity)*100
        else:
            companyDefaultTimeSeries = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt, drift, sigmaAssetBhSh, maturity)*100

    if parameters == 'MM':

        if calibrate == True:
            companyDefaultTimeSeries = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt, drift, sigmaAsset, maturity)*100
        else:
            driftMM = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
            sigmaAssetMM = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()
            companyDefaultTimeSeries = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt, driftMM, sigmaAssetMM, maturity)*100


    observations = ['2021', '2020', '2019', '2018']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by = 'Date', ascending = True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    return defaultDataset.dropna()


def getValueDebtRatio(yahooFinanceTicker):

    import pandas as pd
    import yfinance as yf

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    stock = stock.transpose()[['Total Assets', 'Total Liab']]

    stockValueToDebt = stock['Total Assets'] / stock['Total Liab']

    return stockValueToDebt[3]


def getDebtValue(yahooFinanceTicker, TimeSeries = False):

    import pandas as pd
    import yfinance as yf

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    stockDebt = stock.transpose()['Total Liab']

    if TimeSeries == False:
        return stockDebt[3]

    if TimeSeries == True:
        return stockDebt

def getEquityValue (yahooFinanceTicker, TimeSeries = False):
    import pandas as pd
    import yfinance as yf

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    stockEquity = stock.transpose()['Total Stockholder Equity']

    if TimeSeries == False:
        return stockEquity[3]

    if TimeSeries == True:
        return stockEquity


def calibrateMerton (yahooFinanceTicker, maturity, return_series = False):

    import pandas as pd
    import numpy as np
    from scipy.optimize import minimize
    import yfinance as yf
    import MertonModel as mm
    from scipy import stats
    import matplotlib.pyplot as plt

    # Start from the vector of the "First Guess" Parameters
    df = pd.read_excel(
        r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Big Dataset Calibration\CDS Calibration.xlsx")
    df = df[df['Asset to Debt Ratio'] > 1]
    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    startingParameters = [0.05, 0.3]

    # Try to set the mean bounds from the stock time Series
    stockTimeSeries = (((yf.Ticker(yahooFinanceTicker).history('max', '1d'))['Close']).pct_change().dropna())
    zScore = pd.DataFrame(np.abs(stats.zscore(stockTimeSeries))).set_index(stockTimeSeries.index)
    z = pd.concat([zScore, stockTimeSeries], axis=1).set_axis(['Z-Score', 'rendimenti'], axis=1)
    correctedStockTimeSeries = z[z['Z-Score'] < 0.5]
    del (correctedStockTimeSeries['Z-Score'])
    minBound = correctedStockTimeSeries.quantile(0.25)
    maxBound = correctedStockTimeSeries.quantile(0.75)

    def optimal_params(x, mktValues, valueOfDebt, valueOfAssets):

        candidate_prices = mm.MertonCDSSpread(valueOfAssets, valueOfDebt, riskFreeRate=x[0], sigmaAssets=x[1],
                                              maturity=T)

        return np.linalg.norm(mktValues - candidate_prices, 2)

    T = maturity
    valueOfDebt = df['Total Market Debt Value']
    valueOfAssets = df['Total Assets Value']
    x0 = startingParameters  # initial guess for algorithm
    bounds = ((minBound, maxBound), (0, np.inf))  # bounds for minimization
    mktValues = df['5Y CDS Spread']

    res = minimize(optimal_params, method='trust-constr', x0=x0, args=(mktValues, valueOfDebt, valueOfAssets),
                   bounds=bounds, tol=1e-20,
                   options={"maxiter": 1000})

    if return_series == True:
        return pd.DataFrame([res.x[0], res.x[1]]).transpose().set_axis(['Drift', 'Volatility'], axis = 1).set_index(pd.Series(yahooFinanceTicker))
    else:
       return res.x[0], res.x[1]




# Here we implement the CDS spreads from Merton Model with Hull Equation, that is:
# CDSSpread = [ln(1-PD)] * [(RecoveryRate - 1) / Maturity]
# Here we assume that the recovery rate (in %) is equal to the level of Debt on the assets
# of the firm we want to analyse

def MertonCDSSpread (valueOfAssets, valueOfDebt, riskFreeRate, sigmaAssets, maturity):

    import numpy as np
    from scipy.stats import norm
    import pandas as pd

    distanceToDefault = (np.log(valueOfAssets/valueOfDebt) + ((riskFreeRate + 0.5 * sigmaAssets**2) * maturity)) \
                        / (sigmaAssets * np.sqrt(maturity))

    probabilityOfDefault = norm.cdf( - distanceToDefault)

    # Now we use the PD to feed the Hull formula to get the CDS Spreads

    CDSSpread = (np.log(1 - probabilityOfDefault)) * (((valueOfDebt/valueOfAssets) - 1) / maturity)

    # We convert them in bp, because we Hull gives them in percentage points

    return CDSSpread



def MertonCDSSpreadOnRealStocks (yahooFinanceTicker, maturity, plot = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import yfinance as yf

    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    stock = stock.transpose()[['Total Assets', 'Total Liab']]

    stockValueToDebt = stock['Total Assets'] / stock['Total Liab']

    companyDefaultTimeSeries = list()
    for ratioYear in stockValueToDebt:
        jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(ratioYear, drift, volatility, maturity)
        companyDefaultTimeSeries.append(jnk)


    observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']
    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default'], axis=1)
    defaultDataset = defaultDataset.sort_values(by='Date', ascending=True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    recovery = pd.DataFrame(stock['Total Liab'] / stock['Total Assets']).set_index(pd.Series(observations))

    baseCDSPricing = pd.concat([recovery, defaultDataset], axis = 1).set_axis(['Recovery Rate', 'Probability of Default'], axis = 1)

    CDSSpread = (np.log(1 - baseCDSPricing['Probability of Default'])) * ((baseCDSPricing['Recovery Rate'] - 1) / maturity)
    CDSSpread = pd.DataFrame(CDSSpread).set_axis(['CDS Spread'], axis = 1)

    # We convert them in bp, because we Hull gives them in percentage points

    CDSSpread['CDS Spread'] = CDSSpread['CDS Spread'] * 100

    return CDSSpread


def mertonAlternativeCalibration (yahooFinanceTicker, return_series = False):

    import pandas as pd
    import numpy as np
    from scipy.optimize import minimize
    import yfinance as yf
    import MertonModel as mm
    from scipy import stats
    import matplotlib.pyplot as plt

    # Alternative Calibration Method: recall that we can price the Equity with our simple Merton Model, so we can calibrate it
    # with the historical data we already have from Yahoo Finance

    # We can test if our model at period start could have predicted the present equity value. therefore, we can test if our model
    # in 2019 could have predicted the price of Equity in 2020. We ca assume to 'Freeze' the date on the
    # balance Sheets date we have in our download, and minimize the difference between the estimate parameters and actual ones

    # Define the different bounds for each stock, as explained in Chapter 3

    yahooFinanceTicker = yahooFinanceTicker

    stockTimeSeries = (((yf.Ticker(yahooFinanceTicker).history('5Y', '1d'))['Close']).pct_change().dropna())
    zScore = pd.DataFrame(np.abs(stats.zscore(stockTimeSeries))).set_index(stockTimeSeries.index)
    z = pd.concat([zScore, stockTimeSeries], axis=1).set_axis(['Z-Score', 'rendimenti'], axis=1)
    correctedStockTimeSeries = z[z['Z-Score'] < 0.5]
    del (correctedStockTimeSeries['Z-Score'])
    minBound = correctedStockTimeSeries.quantile(0.25)
    maxBound = correctedStockTimeSeries.quantile(0.75)

    # Load the Dataset

    df = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\DBPerAnnoNoNaN.xlsx")

    # Initialize the "First Guess" parameters

    startingParameters = [0.05, 0.3]

    def optimal_params(x, realValueEquity2021, valueOfDebt2018, valueOfAssets2018):

        PredictionEquityIn2021 = mm.MertonModelEquityPricing(valueOfAssets2018, valueOfDebt2018, riskFreeRate=x[0],
                                                             sigmaAssets=x[1], maturity=T)

        return np.linalg.norm((realValueEquity2021 - PredictionEquityIn2021), 2)

    # Selection of Parameters, Minimization Problem setting

    T = 1
    valueOfDebt2018 = df['Total Liabilities 2020']
    valueOfAssets2018 = df['Total Assets 2020']
    x0 = startingParameters  # initial guess for algorithm
    bounds = ((minBound, maxBound), (0, 5))  # bounds for minimization
    valueEquity2021 = df['Total Equity 2021']

    res = minimize(optimal_params, method='trust-constr', x0=x0,
                   args=(valueEquity2021, valueOfAssets2018, valueOfDebt2018),
                   tol=1e-20, bounds = bounds,
                   options={"maxiter": 1000})

    #print('Calibrated Drift:', res.x[0], '(Starting drift: ', drift, ')')
    #print('Calibrated Volatility:', res.x[1], '(Starting volatility: ', volatility, ')')

    # The Vector of results is returned

    if return_series == True:
        return pd.DataFrame([res.x[0], res.x[1]]).transpose().set_axis(['Drift',
                            'Volatility' ], axis = 1).set_index(pd.Series(yahooFinanceTicker))
    else:
       return res.x[0], res.x[1]



def compareMertonImplementationOnRealStocks (yahooFinanceTicker, frequence, maturity):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    from scipy import stats
    from scipy.stats import norm
    import yfinance as yf
    from datetime import datetime

    realStock = yahooFinanceTicker
    maturity = maturity
    freq = frequence

    b = mm.MertonProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    c = mm.MertonProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=False, freq=freq)
    d = mm.MertonProbabilityOfDefaultRealStocks(realStock, maturity, parameters='MM', calibrate=True, freq=freq)
    e = mm.MertonProbabilityOfDefaultRealStocks(realStock, maturity, parameters='MM', calibrate=False, freq=freq)

    if freq == 'yearly':
        stockTS = (((yf.Ticker(realStock).history(start='2018-12-31', end='2021-12-31'))[
            'Close']).pct_change() * 100).cumsum()
    if freq == 'quarterly':
        stockTS = (((yf.Ticker(realStock).history(start='2021-09-25', end='2022-06-25'))[
            'Close']).pct_change() * 100).cumsum()

    print('BhSh Parameters - Calibration', b)
    print('BhSh Parameters - NO Calibration', c)
    print('MM Parameters - Calibration', d)
    print('MM Parameters - NO Calibration', e)

    dateIndex = list()
    for date in stockTS.index:
        dateIndex.append(datetime.date(date))
    stockTS.index = pd.Series(dateIndex)

    PDs = pd.concat([b, c, d, e], axis=1).set_axis(['BhSh Parameters - Calibration', 'BhSh Parameters - NO Calibration',
                                                    'MM Parameters - Calibration', 'MM Parameters - NO Calibration'],
                                                   axis=1)

    PDs.index = pd.to_datetime(PDs.index)
    dateIndexPD = list()
    for value in PDs.index:
        dateIndexPD.append(datetime.date(value))
    PDs.index = pd.Series(dateIndexPD)

    finalRep = pd.concat([stockTS, PDs], axis=1)
    finalRep = finalRep.sort_index()

    fig, ax1 = plt.subplots(figsize = (12, 5))

    ax1.scatter(x=finalRep.index, y=finalRep['BhSh Parameters - Calibration'], color='blue')
    ax1.plot(finalRep['BhSh Parameters - Calibration'].dropna(), color='blue', label='BhSh Parameters - Calibration')
    ax1.scatter(x=finalRep.index, y=finalRep['BhSh Parameters - NO Calibration'], color='red')
    ax1.plot(finalRep['BhSh Parameters - NO Calibration'].dropna(), color='red',
             label='BhSh Parameters - NO Calibration')
    ax1.scatter(x=finalRep.index, y=finalRep['MM Parameters - Calibration'], color='green')
    ax1.plot(finalRep['MM Parameters - Calibration'].dropna(), color='green', label='MM Parameters - Calibration')
    ax1.scatter(x=finalRep.index, y=finalRep['MM Parameters - NO Calibration'], color='orange')
    ax1.plot(finalRep['MM Parameters - NO Calibration'].dropna(), color='orange',
             label='MM Parameters - NO Calibration')
    ax1.set_ylabel('PD (%)')
    ax1.set_xlabel('Date')
    ax1.set_ylim(0, 50)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(finalRep['Close'], color='black', label='Stock Time Series')
    ax2.set_ylabel('Stock Returns')
    ax2.tick_params(axis='y')
    ax2.legend(['Stock Returns'], loc='upper right')

    plt.title(realStock + '  PD Analysis: Different Approaches of Merton Model')
    plt.show()

    # plt.savefig(r"C:\Users\39328\OneDrive\Desktop\MertonApproches", dpi = 500)

    return finalRep












