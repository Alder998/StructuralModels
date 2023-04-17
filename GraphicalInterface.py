# proviamo a costruire una dashboard con Plotly per fare quello che vogliamo

import time
import numpy as np
from dash import dcc
from dash import html
from dash import Input
from dash import Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import MertonModel as mm
import matplotlib.pyplot as plt
import processes as pr
import JumpDiffusionModelClosedForm as jp
import montecarloJumpDiffisionDaO as mdao
import statsmodels.api as sm
import yfinance as yf
import modelComparison as model
import DownAndOutCall as dao
import MarketDataScraper as mkt
from sklearn.preprocessing import PolynomialFeatures
import time

# Creare Framework per arrivare ai risultati

def getCalibrationParametersDB(yahooFinanceTicker):

    import pandas as pd
    import MertonModel as mm
    import JumpDiffusionModelClosedForm as jp

    ticker = yahooFinanceTicker

    baseMerton = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Quant\Storage_Merton.xlsx").set_index('Unnamed: 0')
    baseJD = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Quant\Storage_JD.xlsx").set_index('Unnamed: 0')


    if (pd.Series(ticker).isin(baseMerton.index)).unique() == True:
        print('Already in Database')
    else:
        mmCal = mm.mertonAlternativeCalibration(ticker, return_series=True)
        baseMerton = pd.concat([baseMerton, mmCal], axis = 0)
        baseMerton.to_excel(r"C:\Users\39328\OneDrive\Desktop\Quant\Storage_Merton.xlsx")

    if (pd.Series(ticker).isin(baseJD.index)).unique() == True:
        print('Already in Database')
    else:
        JDCal = jp.calibrateJumpDiffusion(ticker, 1, return_series=True)
        baseJD = pd.concat([baseJD, JDCal], axis = 0)
        baseJD.to_excel(r"C:\Users\39328\OneDrive\Desktop\Quant\Storage_JD.xlsx")

    return baseMerton, baseJD


# Mettiamo su i 4 modelli che conosciamo, ma con predisposizione ad avere la serie di calibrazione per i parametri calibrati
# Inseriamo anche solo BhSh come parameter selection


def MertonProbabilityOfDefaultRealStocksForGraphicalInterface (yahooFinanceTicker, maturity, freq = 'yearly'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import MarketDataScraper as mkt
    import yfinance as yf

    stockValueToDebt = (mkt.getAssetValue(yahooFinanceTicker, freq=freq, BhSh=True) / mkt.getDebtValue(yahooFinanceTicker, freq=freq,
                                                                                           BhSh=True))
    calibratedParameters = getCalibrationParametersDB(yahooFinanceTicker)[0]
    calibratedParameters = calibratedParameters[calibratedParameters.index == yahooFinanceTicker]

    companyDefaultTimeSeries = list()
    for value in range(len(stockValueToDebt)):

        jnk = mm.MertonProbabilityOfDefaultWithValueToDebt(stockValueToDebt[value], calibratedParameters['Drift'][yahooFinanceTicker],
                                                           calibratedParameters['Volatility'][yahooFinanceTicker], maturity)

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

    return defaultDataset


def JDProbabilityOfDefaultRealStocksForGraphicalInterface (yahooFinanceTicker, maturity, freq = 'yearly'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import MarketDataScraper as mkt
    import JumpDiffusionModelClosedForm as jd
    import yfinance as yf

    stockValueToDebt = (mkt.getAssetValue(yahooFinanceTicker, freq=freq, BhSh=True) / mkt.getDebtValue(yahooFinanceTicker, freq=freq,
                                                                                           BhSh=True))
    calibratedParameters = getCalibrationParametersDB(yahooFinanceTicker)[1]
    calibratedParameters = calibratedParameters[calibratedParameters.index == yahooFinanceTicker]

    companyDefaultTimeSeries = list()
    for value in range(len(stockValueToDebt)):

        jnk = jd.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(stockValueToDebt[value], calibratedParameters['Drift'][yahooFinanceTicker], calibratedParameters['Mean_Jump'][yahooFinanceTicker],
                                                        calibratedParameters['Vol_Jump'][yahooFinanceTicker], calibratedParameters['Lambda'][yahooFinanceTicker],
                                                        calibratedParameters['Volatility'][yahooFinanceTicker], maturity)

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

    return defaultDataset



def MertonDownAndOutRealStocksForGraphicalInterface (yahooFinanceTicker, maturity, freq = 'yearly'):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import MertonModel as mm
    import MarketDataScraper as mkt
    import DownAndOutCall as dao
    import yfinance as yf

    stockValueToDebt = (mkt.getAssetValue(yahooFinanceTicker, freq=freq, BhSh=True) / mkt.getDebtValue(yahooFinanceTicker, freq=freq,
                                                                                           BhSh=True))
    calibratedParameters = getCalibrationParametersDB(yahooFinanceTicker)[0]
    calibratedParameters = calibratedParameters[calibratedParameters.index == yahooFinanceTicker]

    companyDefaultTimeSeries = list()
    for value in range(len(stockValueToDebt)):

        jnk = dao.downAndOutProbabilityOfDefaultWithValueToDebt(stockValueToDebt[value], calibratedParameters['Drift'][yahooFinanceTicker],
                                                           calibratedParameters['Volatility'][yahooFinanceTicker], maturity)

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

    return defaultDataset
