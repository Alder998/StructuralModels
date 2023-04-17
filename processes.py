# Here I will drop some methods to implement, study, and visualize some stochastic processes

def geometricBrownianMotion (initialPrice, drift, diffusion, timeSteps, numberOfPaths, visualization = False):

    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt

    f = list()

    for i in range(numberOfPaths):
        # Brownian Motion

        start = np.zeros(timeSteps)
        inc = np.random.normal(0, 1, size=timeSteps)
        Wt = pd.Series((start + inc).cumsum())
        Wt = np.array(Wt)

        # Solution of the Stochastic Exponential

        St = np.exp(((drift - diffusion ** 2 / 2)) * (1 / timeSteps) + diffusion * inc)
        St = initialPrice * (St * St.cumprod(axis=0))
        f.append(St.reshape(timeSteps, 1))

    a = np.array(f)

    final = pd.DataFrame(a.reshape(numberOfPaths, timeSteps)).T
    paths = final.transpose()

    if visualization == True:
        plt.figure(figsize=(12, 5))
        plt.plot(final)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Geometric Brownian Motion')
        plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\GBM",
                    dpi=1500)
        plt.show()

    return paths.transpose()


def JumpDiffusionMerton (initialPrice, maturity, drift, volatility, lambdaJump, meanOfJump, volatilityOfJump,
                        numberOfTimeSteps, numberOfPaths, visualization=False):
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt

    size = (numberOfTimeSteps, numberOfPaths)
    dt = maturity / numberOfTimeSteps
    poi_rv = np.multiply(np.random.poisson(lambdaJump * dt, size=size),
                         np.random.normal(meanOfJump, volatilityOfJump, size=size)).cumsum(axis=0)

    geo = np.cumsum(((drift - volatility ** 2 / 2 - lambdaJump * (meanOfJump + volatilityOfJump ** 2 * 0.5)) * dt + \
                     volatility * np.sqrt(dt) * \
                     np.random.normal(size=size)), axis=0)

    final = np.exp(geo + poi_rv) * initialPrice

    finalDf = pd.DataFrame(final, columns=[np.arange(0, numberOfPaths)])

    if visualization == True:
        plt.figure(figsize=(12, 5))
        plt.plot(final)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Jump Diffusion Process')
        #plt.savefig(
        #    r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\JumpDIff", dpi=1500)
        plt.show()

    return finalDf


def barrierProcess(yahooFinanceTicker, numberOfTimeSteps, numberOfPaths, visualization = False):

    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    import yfinance as yf
    import statsmodels.api as sm
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
    print(ticker, 'Calibrated Default Barrier Intensity:', max(financialSituation-1, 0) )
    print(ticker, 'Calibrated Default Barrier Jump Volatility:', max(magnitude-1, 0.1))

    # print(ticker, 'News Index:', mkt.getNewsIndex(ticker))

    dt = 1 / numberOfTimeSteps

    size = (numberOfTimeSteps, 1)

    poi_rv = np.multiply(np.random.poisson(max(financialSituation-1, 1e-20) * dt, size=size),
                         np.random.normal(0, max(magnitude-1, 0.1), size=size)).cumsum(axis=0)

    initialValue = mkt.getDebtValue(yahooFinanceTicker)[3]
    final = np.exp(poi_rv) * initialValue

    finalDf = pd.DataFrame(final, columns=[np.arange(0, numberOfPaths)])

    H = ((finalDf.transpose()).mean()).transpose()

    #H = H.rolling(150, min_periods=1).mean()

    if visualization == True:

        plt.figure(figsize=(12, 5))

        plt.plot(H)
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title('Jump Diffusion Process')
        plt.show()

    return H

