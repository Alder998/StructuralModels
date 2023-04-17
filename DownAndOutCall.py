# here we are implementing the Down-And-Out Model for valuating credit risk. It uses the barrier Option framework to
# compute the probability of Default and teh price of teh equity of a defaultable Security.

# I think it is a good alternative to the standard Merton, because it involves that the default could happen at ANY TIME
# of the process path (being a method to valuate originally PATH DEPENDENT), but at the same time it has an analytic
# (Black-Scholes) valuation formula.

# It may require a Montecarlo approach through a binomial (or a Feyman-Kac) simulation, and it could be interesting to
# implement in this way, if there would be time. It could be even nice to apply the Boyle-Lau functional in this
# setting.

def downAndOutEquityPricing (valueOfAssets, valueOfDebt, barrier, riskFreeRate, sigmaAssets, maturity):

    import numpy as np
    from scipy.stats import norm

    eta = (riskFreeRate / (sigmaAssets ** 2)) + 0.5
    eta1 = riskFreeRate + (0.5 * (sigmaAssets**2))

    if valueOfDebt < barrier:
        a = (np.log(valueOfAssets / barrier) / (sigmaAssets * np.sqrt(maturity))) + (
                    eta * sigmaAssets * np.sqrt(maturity))

    if valueOfDebt >= barrier:
        a = (np.log(valueOfAssets / valueOfDebt) / (sigmaAssets * np.sqrt(maturity))) + (
                eta * sigmaAssets * np.sqrt(maturity) )

    if valueOfDebt >= barrier:
        b = (np.log(barrier ** 2 / (valueOfAssets * valueOfDebt)) / (sigmaAssets * np.sqrt(maturity))) + (
                eta * sigmaAssets * np.sqrt(maturity))

    if valueOfDebt < barrier:
        b = (np.log(barrier / valueOfAssets) / (sigmaAssets * np.sqrt(maturity))) + (
                eta * sigmaAssets * np.sqrt(maturity))

    closedFormEquityPrice = valueOfAssets * norm.cdf(a) - valueOfDebt * np.exp(-riskFreeRate * maturity) * norm.cdf(
        a - (sigmaAssets * np.sqrt(maturity))) \
                            - valueOfAssets * (barrier / valueOfAssets) ** (2 * eta) * norm.cdf(b) + valueOfDebt * np.exp(
        -riskFreeRate * maturity) * \
                            (barrier / valueOfAssets) ** (2 * eta - 2) * norm.cdf(b - (sigmaAssets * np.sqrt(maturity)))

    return closedFormEquityPrice



def downAndOutProbabilityOfDefault (valueOfAssets, barrier, riskFreeRate, sigmaAssets, maturity):
    import numpy as np
    from scipy.stats import norm

    eta = (riskFreeRate / sigmaAssets ** 2) + 0.5

    probabilityOfDefault = norm.cdf((- np.log(valueOfAssets / barrier) - (riskFreeRate - 0.5 * sigmaAssets ** 2) * maturity) / (
                sigmaAssets * np.sqrt(maturity))) \
                           + np.exp(((- 2 * riskFreeRate) * np.log(valueOfAssets / barrier)) / sigmaAssets ** 2) \
                           * norm.cdf((- np.log(valueOfAssets / barrier) + (riskFreeRate - 0.5 * sigmaAssets ** 2) * maturity) / (
                sigmaAssets * np.sqrt(maturity)))

    return probabilityOfDefault

def downAndOutProbabilityOfDefaultAT1P (valueOfAssets, barrier, betaBarrier, sigmaAssets, maturity):

    import numpy as np
    from scipy.stats import norm

    timeSteps = 250
    dt = maturity/timeSteps

    probabilitySurvival = norm.cdf((np.log(valueOfAssets / barrier) + betaBarrier*(sigmaAssets ** 2) * dt) / np.sqrt(
                sigmaAssets**2 * dt)) - ((barrier/valueOfAssets)**(2*betaBarrier)) \
                           * norm.cdf((np.log(barrier / valueOfAssets) + betaBarrier*(sigmaAssets ** 2) * dt) / np.sqrt(
                sigmaAssets**2 * dt))

    return (1-probabilitySurvival)


def downAndOutEquityPricingRealStocks (yahooFinanceTicker, maturity, plot = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import DownAndOutCall as dao
    import yfinance as yf

    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    fundamentals = stock.transpose()[['Total Assets', 'Total Liab']]

    companyDefaultTimeSeries = list()
    for i in range(0, len(fundamentals.index)):
        jnk = dao.downAndOutEquityPricing(fundamentals['Total Assets'][i], fundamentals['Total Liab'][i], drift, volatility, maturity)
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


def downAndOutProbabilityOfDefaultWithValueToDebt (valueToDebtRatio, riskFreeRate, sigmaAssets, maturity):
    import numpy as np
    from scipy.stats import norm


    eta = (riskFreeRate / sigmaAssets ** 2) + 0.5


    probabilityOfDefault = norm.cdf(
        (- np.log(valueToDebtRatio) - (riskFreeRate - 0.5 * sigmaAssets ** 2) * maturity) / (
                sigmaAssets * np.sqrt(maturity))) \
                           + np.exp(((- 2 * riskFreeRate) * np.log(valueToDebtRatio)) / (sigmaAssets ** 2)) \
                           * norm.cdf(
        (- np.log(valueToDebtRatio) + (riskFreeRate - 0.5 * sigmaAssets ** 2) * maturity) / (
                sigmaAssets * np.sqrt(maturity)))

    return probabilityOfDefault


def downAndOutProbabilityOfDefaultRealStocks (yahooFinanceTicker, maturity, calibrate = False, freq = 'yearly', parameters = 'MM', plot = False):

    import pandas as pd
    import numpy as np
    import DownAndOutCall as dao
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import yfinance as yf
    import MarketDataScraper as mkt

    if calibrate == True:
        params = mm.mertonAlternativeCalibration(yahooFinanceTicker)
        drift = params[0]
        volatility = params[1]

    if calibrate == False:
        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    if freq == 'yearly':
        if parameters == 'MM':
            debtValue = mkt.getDebtValue(yahooFinanceTicker,BhSh=False, freq = 'yearly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker,BhSh=False, freq = 'yearly').transpose()

        if parameters == 'BhSh':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').transpose()
            volatility = mkt.getBhShVolatility(yahooFinanceTicker, volatility, freq = 'yearly')


    if freq == 'quarterly':
        if parameters == 'MM':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=False, freq = 'quarterly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker,BhSh=False, freq = 'quarterly').transpose()

        if parameters == 'BhSh':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'quarterly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'quarterly').transpose()
            volatility = mkt.getBhShVolatility(yahooFinanceTicker, volatility, freq = 'yearly')

    companyDefaultTimeSeries = list()
    for timeIndex in range(len(debtValue)):

        if parameters == 'MM':
            jnk = downAndOutProbabilityOfDefault(assetValue[timeIndex], debtValue[timeIndex], drift, volatility, maturity)

        if parameters == 'BhSh':
            jnk = downAndOutProbabilityOfDefault(assetValue[timeIndex], debtValue[timeIndex], drift, volatility[timeIndex],
                                                 maturity)
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

def downAndOutProbabilityOfDefaultRealStocks2021 (yahooFinanceTicker, maturity, calibrate = False, parameters = 'MM'):

    import pandas as pd
    import numpy as np
    import DownAndOutCall as dao
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import yfinance as yf
    import MarketDataScraper as mkt

    if calibrate == True:
        params = mm.mertonAlternativeCalibration(yahooFinanceTicker)
        drift = params[0]
        volatility = params[1]

    if calibrate == False:
        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    if parameters == 'MM':
        debtValue = mkt.getDebtValue(yahooFinanceTicker,BhSh=False, freq = 'yearly').transpose().reset_index()[0]
        assetValue = mkt.getAssetValue(yahooFinanceTicker,BhSh=False, freq = 'yearly').transpose().reset_index()[0]

    if parameters == 'BhSh':
        debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').reset_index().transpose()[0]
        assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').reset_index().transpose()[0]
        volatility = mkt.getBhShVolatility(yahooFinanceTicker, volatility, freq = 'yearly').reset_index().transpose()[0]

    for value in range(len(volatility)-1):

        if parameters == 'MM':
            companyDefaultTimeSeries = downAndOutProbabilityOfDefault(assetValue[value], debtValue[value], drift, volatility[value],
                                                                      maturity)
        if parameters == 'BhSh':
            companyDefaultTimeSeries = downAndOutProbabilityOfDefault(assetValue[value], debtValue[value], drift, volatility[value],
                                                                      maturity)

    observations = ['2021', '2020', '2019', '2018']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries)], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by = 'Date', ascending = True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    return defaultDataset.dropna()


# Now we compute the relative CDS Spreads with the Hull formula

def downAndOutCDSSpread (valueOfAssets, barrier, riskFreeRate, sigmaAssets, maturity):
    import numpy as np
    from scipy.stats import norm

    probabilityOfDefault = norm.cdf(
        (- np.log(valueOfAssets / barrier) - (riskFreeRate - 0.5 * sigmaAssets ** 2) * maturity) / (
                sigmaAssets * np.sqrt(maturity))) \
                           + np.exp(((- 2 * riskFreeRate) * np.log(valueOfAssets / barrier)) / sigmaAssets ** 2) \
                           * norm.cdf(
        (- np.log(valueOfAssets / barrier) + (riskFreeRate - 0.5 * sigmaAssets ** 2) * maturity) / (
                sigmaAssets * np.sqrt(maturity)))

    # Hull formula is always the same used for Merton Model

    CDSSpread = (np.log(1 - probabilityOfDefault)) * (((barrier / valueOfAssets) - 1) / maturity)

    # We convert them in bp, because we Hull gives them in percentage points

    return CDSSpread


def DownAndOutCDSSpreadOnRealStocks (yahooFinanceTicker, maturity, plot = False):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import DownAndOutCall as dao
    import yfinance as yf

    drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
    volatility = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    stock = stock.transpose()[['Total Assets', 'Total Liab']]

    stockValueToDebt = stock['Total Assets'] / stock['Total Liab']


    companyDefaultTimeSeries = list()
    for ratioYear in stockValueToDebt:
        jnk = dao.downAndOutProbabilityOfDefaultWithValueToDebt(ratioYear, drift, volatility, maturity)
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


