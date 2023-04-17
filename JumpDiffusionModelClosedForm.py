# Here we implement a Jump-Diffusion Model to compute Default probabilities and price defaultable securities

def jumpDiffusionEquityPricing (valueOfAssets, valueOfDebt, riskFreeRate, meanOfJump, volOfJump, lambdaJump, sigmaAssets, maturity):
    import numpy as np
    import MertonModel as mm

    # Implementation of the closed-form Merton relation for

    p = 0
    for k in range(40):
        r_k = riskFreeRate - lambdaJump * (meanOfJump - 1) + (k * np.log(meanOfJump)) / maturity
        sigma_k = np.sqrt(sigmaAssets ** 2 + (k * volOfJump ** 2) / maturity)
        k_fact = np.math.factorial(k)
        p += (np.exp(-meanOfJump * lambdaJump * maturity) * (meanOfJump * lambdaJump * maturity) ** k / (
            k_fact)) * mm.MertonModelEquityPricing(valueOfAssets,valueOfDebt, r_k, sigma_k, maturity)

    return p


def jumpDiffusionEquityPricingRealStocks (yahooFinanceTicker, maturity, plot = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf

    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    fundamentals = stock.transpose()[['Total Assets', 'Total Liab']]

    companyDefaultTimeSeries = list()
    for i in range(0, len(fundamentals.index)):
        jnk = jumpDiffusionEquityPricing(fundamentals['Total Assets'][i], fundamentals['Total Liab'][i], 0.0822, 0.058, 0.046, -0.042,
                                          0.317735, maturity)
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


def JumpDiffusionProbabilityOfDefault (valueToDebtRatio, riskFreeRate, meanOfJump, volOfJump, lambdaJump, sigmaFirm, maturity):

    import numpy as np
    import MertonModel as mm

    p = 0
    for k in range(40):
        r_k = riskFreeRate - lambdaJump * (meanOfJump - 1) + (k * np.log(meanOfJump)) / maturity
        sigma_k = np.sqrt(sigmaFirm ** 2 + (k * volOfJump ** 2) / maturity)
        k_fact = np.math.factorial(k)
        p += (np.exp(-meanOfJump * lambdaJump * maturity) * (meanOfJump * lambdaJump * maturity) ** k / (k_fact)) * \
             mm.MertonProbabilityOfDefaultWithValueToDebt(valueToDebtRatio, r_k, sigma_k, maturity)

    return p


def JumpDiffusionProbabilityOfDefaultBis (valueOfAssets, valueOfDebt, riskFreeRate, meanOfJump, volOfJump, lambdaJump, sigmaAssets, maturity):

    import numpy as np
    import MertonModel as mm
    from scipy.stats import norm

    vJump = np.exp(meanOfJump + 0.5*(volOfJump**2)) - 1
    p = 0
    for k in range(40):

        jumpDistanceToDefault = (np.log(valueOfAssets/valueOfDebt) + ((riskFreeRate + 0.5 * sigmaAssets ** 2 - (lambdaJump * vJump)) * maturity) + k*meanOfJump) \
                            / np.sqrt(sigmaAssets**2 * maturity + k * volOfJump**2)

        # create the factorial part
        k_fact = np.math.factorial(k)

        # final Closed-Form Formula
        p += ( (np.exp(- lambdaJump * maturity) * (lambdaJump * maturity) ** k) / (k_fact) ) * norm.cdf( - jumpDistanceToDefault)

    return p


def JumpDiffusionProbabilityOfDefaultWithValueDebtRatio (ValueDebtRatio, riskFreeRate, meanOfJump, volOfJump, lambdaJump, sigmaAssets, maturity):

    import numpy as np
    import MertonModel as mm
    from scipy.stats import norm


    vJump = np.exp(meanOfJump + 0.5*(volOfJump**2)) - 1
    p = 0
    for k in range(40):

        jumpDistanceToDefault = (np.log(ValueDebtRatio) + ((riskFreeRate + 0.5 * sigmaAssets ** 2 - (lambdaJump * vJump)) * maturity) + k*meanOfJump) \
                            / np.sqrt(sigmaAssets**2 * maturity + k * volOfJump**2)


        # create the factorial part
        k_fact = np.math.factorial(k)

        # final Closed-Form Formula
        p += ( (np.exp(- lambdaJump * maturity) * (lambdaJump * maturity) ** k) / (k_fact) ) * norm.cdf( - jumpDistanceToDefault)

    return p


def jumpDiffusionProbabilityOfDefaultRealStocks (yahooFinanceTicker, maturity, parameters = 'MM', freq ='yearly', calibrate = False, plot = False, ):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf

    if calibrate == True:
        calBase = calibrateJumpDiffusion(yahooFinanceTicker, maturity)

        drift = calBase[0]
        volatility = calBase[4]
        meanOfJump = calBase[1]
        volOfJump = calBase[2]
        lamdaJump = calBase[3]

    if calibrate == False:
        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()
        meanOfJump = 0.005
        volOfJump = 0.02
        lamdaJump = 0.5

   # Take the parameters from Yahoo Finance

    if freq == 'yearly':
        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    if freq == 'quarterly':
        stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

    if parameters == 'BhSh':
        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stockEquity + stockDebt
        sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

        stockValueToDebt = stockAssets / stockDebt

    if parameters == 'MM':
        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stock.transpose()['Total Assets']

        stockValueToDebt = stockAssets / stockDebt

    companyDefaultTimeSeries = list()
    for ratioYear in range(len(stockValueToDebt)):

        if parameters == 'MM':
            jnk = JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(stockValueToDebt[ratioYear], drift, meanOfJump, volOfJump,
                                                                      lamdaJump, volatility, maturity)

        if parameters == 'BhSh':
            jnk = JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(stockValueToDebt[ratioYear], drift, meanOfJump, volOfJump,
                                                                      lamdaJump, sigmaAssetBhSh[ratioYear], maturity)
        companyDefaultTimeSeries.append(jnk * 100)

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

def jumpDiffusionProbabilityOfDefaultRealStocks2021 (yahooFinanceTicker, maturity, parameters = 'MM', freq ='yearly', calibrate = False, plot = False, ):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf

    if calibrate == True:
        calBase = calibrateJumpDiffusion(yahooFinanceTicker, maturity)

        drift = calBase[0]
        volatility = calBase[4]
        meanOfJump = calBase[1]
        volOfJump = calBase[2]
        lamdaJump = calBase[3]

    if calibrate == False:
        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()
        meanOfJump = 0.005
        volOfJump = 0.02
        lamdaJump = 0.5

   # Take the parameters from Yahoo Finance


    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()

    if parameters == 'BhSh':

        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet().transpose().reset_index()
        stock = stock.transpose()
        stock = stock[0]

        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]

        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = drift  # ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stockEquity + stockDebt
        sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

        stockValueToDebt = stockAssets / stockDebt

    if parameters == 'MM':
        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stock.transpose()['Total Assets']

        stockValueToDebt = stockAssets / stockDebt

    if parameters == 'MM':
        companyDefaultTimeSeries = JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(stockValueToDebt, drift, meanOfJump, volOfJump,
                                                                  lamdaJump, volatility, maturity)*100
    if parameters == 'BhSh':
        companyDefaultTimeSeries = JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(stockValueToDebt, drift, meanOfJump, volOfJump,
                                                                  lamdaJump, sigmaAssetBhSh, maturity)*100

    observations = ['2021', '2020', '2019', '2018']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by = 'Date', ascending = True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    return defaultDataset.dropna()


def CompareProbabilityOfDefaultRealStocks (yahooFinanceTicker, maturity, plot = False, subplots = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf
    import MertonModel as mm

    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    stock = stock.transpose()[['Total Assets', 'Total Liab']]

    stockValueToDebt = stock['Total Assets'] / stock['Total Liab']

    # Jump Diffusion Setting
    companyDefaultTimeSeries = list()
    for ratioYear in stockValueToDebt:
        jnk = JumpDiffusionProbabilityOfDefault(ratioYear, 0.005, 0.02, 1, drift, volatility, maturity)
        companyDefaultTimeSeries.append(jnk * 100)

    observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']
    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by = 'Date', ascending = True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    # Merton Model Setting
    companyDefaultTimeSeriesM = list()
    for ratioYear in stockValueToDebt:
        jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(ratioYear, drift, volatility, maturity)
        companyDefaultTimeSeriesM.append(jnk * 100)

    observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']
    defaultDatasetM = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeriesM)], axis=1)
    defaultDatasetM = defaultDatasetM.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDatasetM = defaultDatasetM.sort_values(by='Date', ascending=True)
    defaultDatasetM = defaultDatasetM.set_index(defaultDatasetM['Date'])
    del (defaultDatasetM['Date'])

    if plot == True:
        plt.figure(figsize=(12, 5))
        plt.plot(defaultDataset, color = 'blue')
        plt.plot(defaultDatasetM, color = 'red')
        plt.scatter(x=defaultDatasetM.index, y=defaultDatasetM, color='red')
        plt.scatter(x=defaultDataset.index, y=defaultDataset, color = 'blue')
        plt.xlabel('Balance Sheet level for year')
        plt.ylabel('Probability of Default (%)')
        plt.legend(['Merton Distance to Default Model', 'Jump Diffusion Model (closed Form)'])
        plt.show()

    if subplots == True:

        plt.figure(figsize=(12, 7))

        plt.title(yf.Ticker(yahooFinanceTicker).info['longName'])
        plt.subplot(2, 1, 1)

        plt.plot(defaultDataset, color='blue')
        plt.scatter(x=defaultDataset.index, y=defaultDataset, color = 'blue')
        plt.xlabel('Balance Sheet level for year')
        plt.ylabel('Probability of Default (%)')
        plt.legend(['Jump Diffusion Model (closed Form)'])

        plt.subplot(2, 1, 2)

        plt.plot(defaultDatasetM, color='red')
        plt.scatter(x=defaultDatasetM.index, y=defaultDatasetM, color='red')
        plt.xlabel('Balance Sheet level for year')
        plt.ylabel('Probability of Default (%)')
        plt.legend(['Merton Distance to Default Model'])

        plt.show()

    return defaultDataset


def calibrateJumpDiffusion (yahooFinanceTicker, maturity, return_series = False):

    import pandas as pd
    import numpy as np
    from scipy.optimize import minimize
    import yfinance as yf
    import MarketDataScraper as yfs
    import MertonModel as mm
    import matplotlib.pyplot as plt
    from scipy import stats
    import JumpDiffusionModelClosedForm as jp

    yahooFinanceTicker = yahooFinanceTicker

    stockTimeSeries = (((yf.Ticker(yahooFinanceTicker).history('max', '1d'))['Close']).pct_change().dropna())

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

    bounds = ((minBound, maxBound), (minBound, maxBound), (0, np.inf), (0, 5), (0, np.inf))

    mktValues2021 = df['Total Equity 2020'].dropna()
    res = minimize(optimal_params, method='trust-constr', x0=x0, args=(mktValues2021, valueOfDebt),
                   tol=1e-20, bounds=bounds,
                   options={"maxiter": 1000})

    print('Calibrated drift:', res.x[0])
    print('Calibrated mean of Jump Component:', res.x[1])
    print('Calibrated volatility Of Jump Component:', res.x[2])
    print('Calibrated Lambda:', res.x[3])
    print('Calibrated volatility of Assets:', res.x[4])

    if return_series == True:
        return pd.DataFrame([res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]]).transpose().set_axis(['Drift', 'Mean_Jump', 'Vol_Jump', 'Lambda',
                                                                                                      'Volatility' ], axis = 1).set_index(pd.Series(yahooFinanceTicker))
    else:
       return res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]


def compareJDImplementations (yahooFinanceTicker, frequence, maturity):
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

    b = jumpDiffusionProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    c = jumpDiffusionProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=False, freq=freq)
    d = jumpDiffusionProbabilityOfDefaultRealStocks(realStock, maturity, parameters='MM', calibrate=True, freq=freq)
    e = jumpDiffusionProbabilityOfDefaultRealStocks(realStock, maturity, parameters='MM', calibrate=False, freq=freq)

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

    fig, ax1 = plt.subplots(figsize=(12, 5))

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
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(finalRep['Close'], color='black', label='Stock Time Series')
    ax2.set_ylabel('Stock Returns')
    ax2.tick_params(axis='y')
    ax2.legend(['Stock Returns'], loc='upper right')

    plt.title(realStock + '  PD Analysis: Different Approaches of Jump-Diffusion Merton Model')
    plt.show()

    # plt.savefig(r"C:\Users\39328\OneDrive\Desktop\MertonApproches", dpi = 500)

    return finalRep
