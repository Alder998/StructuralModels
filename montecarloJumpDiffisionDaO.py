# Here we will implement a Montecarlo method to compute default Probabilities with a Jump-Diffusion
# Framework, and we will build cool stuff on it

def montecarloJumpDiffusionDownAndOutPD (assetValue, debtValue, barrier, drift, sigmaAssets, lambdaJump, meanOfJump,
                                       volatilityOfJump, numberOfTimeSteps, numberOfPaths, maturity, plot = False):
    import pandas as pd
    import processes as pr
    import math
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt

    H = list()
    for i in range(numberOfTimeSteps):
       functionalH = barrier #+ 2e1 * (np.cos(i))
       H.append(functionalH)

    # set the Dynamics of the process

    j = pr.JumpDiffusionMerton(assetValue, maturity, drift, sigmaAssets, lambdaJump, meanOfJump, volatilityOfJump, numberOfTimeSteps, numberOfPaths, False)

    # In the format we created before, the number of path is espressed with the columns of the dataframe (== every column is
    # a specific process)

    # Now we have to play with the barrier. We have to create a framework for which, once the level of the barrier is touched,
    # the process is ending forever, as the option that becomes worthless once the barrier is touched.

    prova = list()
    for valueBarrier in range(len(H)):
        onlyIntheMoney = j.transpose()[valueBarrier].where(
            (j.transpose()[valueBarrier] > H[valueBarrier]) & (j.transpose()[len(H) - 1] > debtValue), math.nan)
        prova.append(onlyIntheMoney)

    prova = pd.concat([row for row in prova], axis=1)

    prova = prova.transpose()

    # Set a way to make all the values NaN after the first NaN

    St = prova.where(prova.notna().cumprod(axis=0).eq(1))

    # initialize the Payoff function

    indice = pd.Series(St.isna().nunique())

    PDstart = list()
    for value in indice:
        PDstart.append((indice[indice == 2].count()))

    PDstart = pd.Series(PDstart).unique()
    PD = PDstart / numberOfPaths

    if plot == True:

        plt.figure(figsize=(11, 5))
        plt.plot(St)
        plt.plot(H, color='black', linestyle='dashed', label = 'Level Of Debt (= Barrier)')
        plt.title('Figure 1.23')
        plt.legend()
        #plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\Montecarlo Jump 100k.png", dpi = 1500)
        plt.show()

    for i in PD:
        return i * 100

def montecarloJumpDiffusionDownAndOutPDStochasticBarrier (yahooFinanceTicker, assetValue, debtValue, drift, sigmaAssets, lambdaJump, meanOfJump,
                                       volatilityOfJump, numberOfTimeSteps, numberOfPaths, maturity, plot = False):
    import pandas as pd
    import processes as pr
    import math
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt

    H = pr.barrierProcess(yahooFinanceTicker, numberOfTimeSteps, 1)

    # set the Dynamics of the process

    j = pr.JumpDiffusionMerton(assetValue, maturity, drift, sigmaAssets, lambdaJump, meanOfJump, volatilityOfJump,
                               numberOfTimeSteps, numberOfPaths, False)

    # In the format we created before, the number of path is espressed with the columns of the dataframe (== every column is
    # a specific process)

    # Now we have to play with the barrier. We have to create a framework for which, once the level of the barrier is touched,
    # the process is ending forever, as the option that becomes worthless once the barrier is touched.

    prova = list()
    for valueBarrier in range(len(H)):
        onlyIntheMoney = j.transpose()[valueBarrier].where(
            (j.transpose()[valueBarrier] > H[valueBarrier]) & (j.transpose()[len(H) - 1] > debtValue), math.nan)
        prova.append(onlyIntheMoney)

    prova = pd.concat([row for row in prova], axis=1)

    prova = prova.transpose()

    # Set a way to make all the values NaN after the first NaN

    St = prova.where(prova.notna().cumprod(axis=0).eq(1))

    # initialize the Payoff function

    indice = pd.Series(St.isna().nunique())

    PDstart = list()
    for value in indice:
        PDstart.append((indice[indice == 2].count()))

    PDstart = pd.Series(PDstart).unique()
    PD = PDstart / numberOfPaths

    if plot == True:
        plt.figure(figsize=(11, 7))

        plt.subplots_adjust(hspace=0.300)

        plt.subplot(2, 1, 1)
        plt.plot(St)
        plt.plot(H, color='black', linestyle='dashed')
        plt.title('Montecarlo Down-And-Out Paths')

        plt.subplot(2, 1, 2)
        plt.plot(H, color='red', label='Barrier Calibrated Process')
        plt.title('Barrier Calibrated Process')

        plt.legend()
        #plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\Montecarlo Stochastic Barrier 10k.png", dpi = 1500)
        plt.show()

    for i in PD:
        return i * 100


def montecarloJumpDiffusionDownAndOutPayOff (assetValue, debtValue, barrier, drift, sigmaAssets, lambdaJump, meanOfJump,
                                       volatilityOfJump, numberOfTimeSteps, numberOfPaths, maturity):
    import pandas as pd
    import processes as pr
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    H = list()
    for i in range(numberOfTimeSteps):
        functionalH = barrier + 2e4 * (i**2)
        H.append(functionalH)

    # set the Dynamics of the process

    j = pr.JumpDiffusionMerton(assetValue, maturity, drift, sigmaAssets, lambdaJump, meanOfJump, volatilityOfJump, numberOfTimeSteps, numberOfPaths, False)

    # In the format we created before, the number of path is espressed with the columns of the dataframe (== every column is
    # a specific process)

    # Now we have to play with the barrier. We have to create a framework for which, once the level of the barrier is touched,
    # the process is ending forever, as the option that becomes worthless once the barrier is touched.

    prova = list()
    for valueBarrier in range(len(H)):
        onlyIntheMoney = j.transpose()[valueBarrier].where(
            (j.transpose()[valueBarrier] > H[valueBarrier]) & (j.transpose()[len(H) - 1] > debtValue), math.nan)
        prova.append(onlyIntheMoney)

    prova = pd.concat([row for row in prova], axis=1)

    prova = prova.transpose()

    # Set a way to make all the values NaN after the first NaN

    St = prova.where(prova.notna().cumprod(axis=0).eq(1))

    # initialize the Payoff function

    payoff = St - debtValue

    return payoff


def montecarloJumpDiffusionDownAndOutOnRealStocksSB (yahooFinanceTicker, maturity, calibrate = False, parameters = 'MM', freq = 'yearly'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import JumpDiffusionModelClosedForm as jp
    import yfinance as yf
    import MertonModel as mm
    import MarketDataScraper as mkt

    if calibrate == False:

        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        sigmaAssets = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

        lambdaJump = 0.5
        meanOfJump = 0.05
        volatilityOfJump = 0.2

    if calibrate == True:

        params = jp.calibrateJumpDiffusion(yahooFinanceTicker, maturity)

        drift = params[0]
        sigmaAssets = params[4]
        lambdaJump = params[3]
        meanOfJump = params[1]
        volatilityOfJump = params[2]

    if freq == 'yearly':
        if parameters == 'MM':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, freq = 'yearly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, freq = 'yearly').transpose()

        if parameters == 'BhSh':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').transpose()
            sigmaAssets = mkt.getBhShVolatility(yahooFinanceTicker, sigmaAssets, freq = 'yearly')

    if freq == 'quarterly':
        if parameters == 'MM':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, freq = 'quarterly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, freq = 'quarterly').transpose()

        if parameters == 'BhSh':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'quarterly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'quarterly').transpose()
            sigmaAssets = mkt.getBhShVolatility(yahooFinanceTicker, sigmaAssets, freq = 'quarterly')

    # Jump Diffusion Setting
    companyDefaultTimeSeries = list()

    for timeIndex in range(len(debtValue)):

        if parameters == 'BhSh':
             test = montecarloJumpDiffusionDownAndOutPDStochasticBarrier(yahooFinanceTicker, assetValue[timeIndex], debtValue[timeIndex],
                                                        drift, sigmaAssets[timeIndex],lambdaJump, meanOfJump, volatilityOfJump,
                                                        250*maturity, 10000, maturity)


        if parameters == 'MM':
             test = montecarloJumpDiffusionDownAndOutPDStochasticBarrier(yahooFinanceTicker, assetValue[timeIndex], debtValue[timeIndex],
                                                        drift, sigmaAssets,lambdaJump, meanOfJump, volatilityOfJump,
                                                        250*maturity, 10000, maturity)

        companyDefaultTimeSeries.append(test)

    if freq == 'yearly':
        observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']
    if freq == 'quarterly':
        observations = ['2022-06-25', '2022-03-26', '2021-12-25', '2021-09-25']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries).dropna()], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by='Date', ascending=True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    return defaultDataset

def montecarloJumpDiffusionDownAndOutOnRealStocks (yahooFinanceTicker, maturity, calibrate = False, parameters = 'MM', freq = 'yearly'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import JumpDiffusionModelClosedForm as jp
    import yfinance as yf
    import MertonModel as mm
    import MarketDataScraper as mkt

    if calibrate == False:

        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        sigmaAssets = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

        lambdaJump = 0.5
        meanOfJump = 0.05
        volatilityOfJump = 0.2

    if calibrate == True:

        params = jp.calibrateJumpDiffusion(yahooFinanceTicker, maturity)

        drift = params[0]
        sigmaAssets = params[4]
        lambdaJump = params[3]
        meanOfJump = params[1]
        volatilityOfJump = params[2]

    if freq == 'yearly':
        if parameters == 'MM':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, freq = 'yearly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, freq = 'yearly').transpose()

        if parameters == 'BhSh':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'yearly').transpose()
            sigmaAssets = mkt.getBhShVolatility(yahooFinanceTicker, sigmaAssets, freq = 'yearly')

    if freq == 'quarterly':
        if parameters == 'MM':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, freq = 'quarterly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, freq = 'quarterly').transpose()

        if parameters == 'BhSh':
            debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq = 'quarterly').transpose()
            assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq = 'quarterly').transpose()
            sigmaAssets = mkt.getBhShVolatility(yahooFinanceTicker, sigmaAssets, freq = 'quarterly')

    # Jump Diffusion Setting
    companyDefaultTimeSeries = list()

    for timeIndex in range(len(debtValue)):

        if parameters == 'BhSh':

             test = montecarloJumpDiffusionDownAndOutPD(assetValue[timeIndex], debtValue[timeIndex], debtValue[timeIndex],
                                                 drift, sigmaAssets[timeIndex],lambdaJump, meanOfJump, volatilityOfJump,
                                                 250*maturity, 10000, maturity)

        if parameters == 'MM':

             test = montecarloJumpDiffusionDownAndOutPD(assetValue[timeIndex], debtValue[timeIndex],
                                                        debtValue[timeIndex],
                                                        drift, sigmaAssets, lambdaJump, meanOfJump,
                                                        volatilityOfJump,
                                                        250 * maturity, 10000, maturity)
        companyDefaultTimeSeries.append(test)

    if freq == 'yearly':
        observations = ['2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31']
    if freq == 'quarterly':
        observations = ['2022-09-25', '2022-06-26', '2022-03-25', '2021-12-25']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries).dropna()], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by='Date', ascending=True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    return defaultDataset


def montecarloJumpDiffusionDownAndOutOnRealStocks2021 (yahooFinanceTicker, maturity, calibrate = False, parameters = 'MM'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import JumpDiffusionModelClosedForm as jp
    import yfinance as yf
    import MertonModel as mm
    import MarketDataScraper as mkt

    if calibrate == False:

        drift = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).mean()
        sigmaAssets = (((yf.Ticker('^GSPC').history('max', '1d'))['Close']).pct_change()).std()

        lambdaJump = 0.5
        meanOfJump = 0.05
        volatilityOfJump = 0.2

    if calibrate == True:

        params = jp.calibrateJumpDiffusion(yahooFinanceTicker, maturity)

        drift = params[0]
        sigmaAssets = params[4]
        lambdaJump = params[3]
        meanOfJump = params[1]
        volatilityOfJump = params[2]

    if parameters == 'MM':
        debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=False, freq='yearly').transpose().reset_index()[0]
        assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=False, freq='yearly').transpose().reset_index()[0]

    if parameters == 'BhSh':
        debtValue = mkt.getDebtValue(yahooFinanceTicker, BhSh=True, freq='yearly').reset_index().transpose()[0]
        assetValue = mkt.getAssetValue(yahooFinanceTicker, BhSh=True, freq='yearly').reset_index().transpose()[0]
        sigmaAssets = mkt.getBhShVolatility(yahooFinanceTicker, sigmaAssets, freq='yearly').reset_index().transpose()[0]

    # Jump Diffusion Setting
    companyDefaultTimeSeries = list()

    for timeIndex in range(len(sigmaAssets) - 1):

        if parameters == 'BhSh':

             test = montecarloJumpDiffusionDownAndOutPD(assetValue[timeIndex], debtValue[timeIndex], debtValue[timeIndex],
                                                 drift, sigmaAssets[timeIndex],lambdaJump, meanOfJump, volatilityOfJump,
                                                 250*maturity, 10000, maturity)

        if parameters == 'MM':

             test = montecarloJumpDiffusionDownAndOutPD(assetValue[timeIndex], debtValue[timeIndex],
                                                        debtValue[timeIndex],
                                                        drift, sigmaAssets, lambdaJump, meanOfJump,
                                                        volatilityOfJump,
                                                        250 * maturity, 10000, maturity)
        companyDefaultTimeSeries.append(test)

    observations = ['2021', '2020', '2019', '2018']

    defaultDataset = pd.concat([pd.Series(observations), pd.Series(companyDefaultTimeSeries).dropna()], axis=1)
    defaultDataset = defaultDataset.set_axis(['Date', 'Probability of Default (%)'], axis=1)
    defaultDataset = defaultDataset.sort_values(by='Date', ascending=True)
    defaultDataset = defaultDataset.set_index(defaultDataset['Date'])
    del (defaultDataset['Date'])

    return defaultDataset.dropna()


def MontecarloDownAndOutEquityPricing (AssetValue, valueOfDebt, valueOfBarrier, drift, sigmaAssets, lambdaJump, meanOfJump, volOfJump, numberOfTimeSteps, numberOfPaths, maturity):

    import math
    import pandas as pd
    import numpy as np
    import processes as pr
    import montecarloJumpDiffisionDaO as mdao
    import statsmodels.api as sm
    from sklearn.preprocessing import PolynomialFeatures

    # Create the process Dynamics
    St = pr.JumpDiffusionMerton(AssetValue, maturity, drift, sigmaAssets, lambdaJump, meanOfJump, volOfJump, numberOfTimeSteps, numberOfPaths, False)

    # Import the payoff paths
    payoff = mdao.montecarloJumpDiffusionDownAndOutPayOff(AssetValue, valueOfDebt, valueOfBarrier, drift, sigmaAssets, lambdaJump, meanOfJump, volOfJump, numberOfTimeSteps,
                                                          numberOfPaths, maturity)

    # Take the paths that are in-the-money
    payoffOnlyInTheMoney = payoff.dropna(axis=1).T.reset_index(drop=True).T

    # Invert the indices
    payoffI = payoffOnlyInTheMoney.transpose().sort_index(ascending=False).mean()

    # define the first guess of the algorithm

    final = list()
    for timeStepBackward in payoffI.index:

        x = St.transpose().reset_index()[timeStepBackward]

        y = payoffOnlyInTheMoney.transpose().reset_index()[numberOfTimeSteps - 1] * np.exp(-drift * maturity)

        itmPath = pd.concat([x, y], axis=1).set_axis(['x', 'y'], axis=1)
        itmPath = itmPath.dropna()

        # We have created a Montecarlo payoff function: we have the payoff value for any time step of our model

        # Now, however, we have to establish the continuation value of our processes, and we have to do it with a
        # POLYNOMIAL REGRESSION on any time steps of our processes

        x = np.array(itmPath['x']).reshape(-1, 1)
        y = np.array(itmPath['y']).reshape(-1, 1)

        # polynomial_features = PolynomialFeatures(degree=2)
        # xp = polynomial_features.fit_transform(x)
        #
        xp = sm.add_constant(x)
        coeff = sm.OLS(y, xp).fit()

        # print(coeff.params)

        Cs = (St[timeStepBackward] * coeff.params[1]).sum()

        if (Cs[timeStepBackward] < payoffI[timeStepBackward]):
            final.append(payoffI[timeStepBackward])
        if (Cs[timeStepBackward] > payoffI[timeStepBackward]):
            final.append(math.nan)

        for EquityValue in (pd.Series(final).dropna().head(1)):
            return EquityValue


def compareMontecarloImplementations (yahooFinanceTicker, maturity, freq = 'yearly'):
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
    #freq = frequence

    b = montecarloJumpDiffusionDownAndOutOnRealStocks(yahooFinanceTicker, maturity, calibrate = True, parameters='BhSh', freq = freq)
    c = montecarloJumpDiffusionDownAndOutOnRealStocks(yahooFinanceTicker, maturity, calibrate = False, parameters='BhSh', freq = freq)
    d = montecarloJumpDiffusionDownAndOutOnRealStocks(yahooFinanceTicker, maturity, calibrate = True, parameters='MM', freq = freq)
    e = montecarloJumpDiffusionDownAndOutOnRealStocks(yahooFinanceTicker, maturity, calibrate = False, parameters='MM', freq = freq)

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
    #ax1.set_ylim(0, 50)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(finalRep['Close'], color='black', label='Stock Time Series')
    ax2.set_ylabel('Stock Returns')
    ax2.tick_params(axis='y')
    ax2.legend(['Stock Returns'], loc='upper right')

    plt.title(realStock + '  PD Analysis: Different Approaches of Montecarlo Down-And-Out')
    plt.show()

    # plt.savefig(r"C:\Users\39328\OneDrive\Desktop\MertonApproches", dpi = 500)

    return finalRep


def calibrateStochasticBarrier (yahooFinanceTicker, timeSteps):

    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import yfinance as yf
    import MarketDataScraper as mkt

    ticker = yahooFinanceTicker

    stock = (((yf.Ticker(ticker).history('5Y', '1d'))['Close']).pct_change()).dropna()

    # Regressors
    vix = (((yf.Ticker('^VIX').history('5Y', '1d'))['Close']).pct_change()).dropna()
    SPX = (((yf.Ticker('^GSPC').history('5Y', '1d'))['Close']).pct_change()).dropna()
    ir = (((yf.Ticker('^IRX').history('5Y', '1d'))['Close']).pct_change()).dropna()
    oil = (((yf.Ticker('CL=F').history('5Y', '1d'))['Close']).pct_change()).dropna()
    gas = (((yf.Ticker('NG=F').history('5Y', '1d'))['Close']).pct_change()).dropna()

    base = pd.concat([stock, vix, SPX, ir, oil, gas], axis=1).set_axis([ticker, '^VIX', 'SPX', 'IR', 'CL=F', 'NG=F'],
                                                                       axis=1).dropna()
    proxy = list()
    for index in ['^VIX', 'SPX', 'IR', 'CL=F', 'NG=F']:
        x = base[ticker]
        x = sm.add_constant(x)
        y = base[index]

        coeff = sm.OLS(x, y).fit()

        m = coeff.params[1]
        proxy.append(m)

    magnitude = list()
    for p in proxy:
        for value in p:
            magnitude.append((value))

    magnitude = pd.Series(magnitude).sum()

    financialSituation = mkt.getValueDebtAverage(ticker)

    #print(ticker, 'Average Asset/Debt:', mkt.getValueDebtAverage(ticker))
    #print(ticker, 'Market Index:', magnitude)
    print(ticker, 'Calibrated Default Barrier Intensity:', (5-financialSituation) )
    print(ticker, 'Calibrated Default Barrier Jump Volatility:', magnitude)

    # print(ticker, 'News Index:', mkt.getNewsIndex(ticker))

    dt = 1 / timeSteps

    for m in m:
        poi_rv = np.multiply(np.random.poisson((3 - financialSituation) * dt, size=timeSteps),
                             np.random.normal(0, magnitude, size=timeSteps)).cumsum(axis=0)

    return poi_rv