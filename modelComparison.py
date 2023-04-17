def allModelsComparisonOnRealStock (yahooFinanceTicker, maturity, freq = 'yearly'):

    import MertonModel as mm
    import JumpDiffusionModelClosedForm as jp
    import montecarloJumpDiffisionDaO as mdao
    import DownAndOutCall as dao
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

    b = mm.MertonProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    c = jp.jumpDiffusionProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    d = dao.downAndOutProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    e = mdao.montecarloJumpDiffusionDownAndOutOnRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)

    if freq == 'yearly':
        stockTS = (((yf.Ticker(realStock).history(start='2018-12-31', end='2021-12-31'))[
            'Close']).pct_change() * 100).cumsum()
    if freq == 'quarterly':
        stockTS = (((yf.Ticker(realStock).history(start='2021-12-25', end='2022-09-29'))[
            'Close']).pct_change() * 100).cumsum()

    print('Standard Merton - BhSh Parameters', b)
    print('Jump-Diffusion Merton - BhSh Parameters', c)
    print('Down-And-Out Merton - Fixed Barrier', d)
    print('Montecarlo Jump-Diffusion Down-And-Out', e)

    dateIndex = list()
    for date in stockTS.index:
        dateIndex.append(datetime.date(date))
    stockTS.index = pd.Series(dateIndex)

    PDs = pd.concat([b, c, d, e], axis=1).set_axis(['Standard Merton - BhSh Parameters', 'Jump-Diffusion Merton - BhSh Parameters',
                                                    'Down-And-Out Merton - Fixed Barrier', 'Montecarlo Jump-Diffusion Down-And-Out'],
                                                    axis=1)

    PDs.index = pd.to_datetime(PDs.index)
    dateIndexPD = list()
    for value in PDs.index:
        dateIndexPD.append(datetime.date(value))
    PDs.index = pd.Series(dateIndexPD)

    finalRep = pd.concat([stockTS, PDs], axis=1)
    finalRep = finalRep.sort_index()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.scatter(x=finalRep.index, y=finalRep['Standard Merton - BhSh Parameters'], color='blue')
    ax1.plot(finalRep['Standard Merton - BhSh Parameters'].dropna(), color='blue', label='Merton with BhSh Parameters')

    ax1.scatter(x=finalRep.index, y=finalRep['Jump-Diffusion Merton - BhSh Parameters'], color='red')
    ax1.plot(finalRep['Jump-Diffusion Merton - BhSh Parameters'].dropna(), color='red',
             label='Jump-Diffusion Merton - BhSh Parameters')

    ax1.scatter(x=finalRep.index, y=finalRep['Down-And-Out Merton - Fixed Barrier'], color='green')
    ax1.plot(finalRep['Down-And-Out Merton - Fixed Barrier'].dropna(), color='green', linestyle = 'dashed',
             label='Down-and-Out with BhSh Parameters')

    ax1.scatter(x=finalRep.index, y=finalRep['Montecarlo Jump-Diffusion Down-And-Out'], color='orange')
    ax1.plot(finalRep['Montecarlo Jump-Diffusion Down-And-Out'].dropna(), color='orange', linestyle = 'dashed',
             label='Montecarlo Jump-Diffusion Down-And-Out')

    ax1.set_ylabel('PD (%)')
    ax1.set_xlabel('Date')
    #ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.plot(finalRep['Close'], color='black', label='Stock Time Series')
    ax2.set_ylabel('Stock Returns')
    ax2.tick_params(axis='y')
    ax2.legend(['Stock Returns'], loc='upper right')

    plt.title(realStock + '  PD Analysis: Different Models Comparison')

    #plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\IBEq", dpi = 1500)

    plt.show()

    return finalRep


def StochasticBarrierClassification(yahooFinanceTicker, maturity, freq = 'yearly'):

    import MertonModel as mm
    import JumpDiffusionModelClosedForm as jp
    import montecarloJumpDiffisionDaO as mdao
    import DownAndOutCall as dao
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

    e = mdao.montecarloJumpDiffusionDownAndOutOnRealStocksSB(realStock, maturity, parameters='BhSh', calibrate=True,
                                                           freq=freq)

    if freq == 'yearly':
        stockTS = (((yf.Ticker(realStock).history(start='2018-12-31', end='2021-12-31'))[
            'Close']).pct_change() * 100).cumsum()
    if freq == 'quarterly':
        stockTS = (((yf.Ticker(realStock).history(start='2021-09-25', end='2022-06-25'))[
            'Close']).pct_change() * 100).cumsum()

    # print('Standard Merton - BhSh Parameters', b)
    # print('Jump-Diffusion Merton - BhSh Parameters', c)
    # print('Down-And-Out Merton - Fixed Barrier', d)
    # print('Montecarlo Down-And-Out Model', e)

    dateIndex = list()
    for date in stockTS.index:
        dateIndex.append(datetime.date(date))
    stockTS.index = pd.Series(dateIndex)

    PDs = e.set_axis(['Montecarlo DaO Stochastic Barrier'],axis=1)

    PDs.index = pd.to_datetime(PDs.index)
    dateIndexPD = list()
    for value in PDs.index:
        dateIndexPD.append(datetime.date(value))
    PDs.index = pd.Series(dateIndexPD)
    PDs = PDs.sort_index()

    # let's set the classification Framework

    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 0) & (
                PDs['Montecarlo DaO Stochastic Barrier'] < 0.05), 'MdaoSB Rating'] = 'AAA'
    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 0.05) & (
                PDs['Montecarlo DaO Stochastic Barrier'] < 0.09), 'MdaoSB Rating'] = 'AA'
    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 0.09) & (
                PDs['Montecarlo DaO Stochastic Barrier'] < 0.23), 'MdaoSB Rating'] = 'A'
    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 0.23) & (
                PDs['Montecarlo DaO Stochastic Barrier'] < 1.16), 'MdaoSB Rating'] = 'BBB'
    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 1.16) & (
                PDs['Montecarlo DaO Stochastic Barrier'] < 5.44), 'MdaoSB Rating'] = 'BB'
    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 5.44) & (
                PDs['Montecarlo DaO Stochastic Barrier'] < 14.21), 'MdaoSB Rating'] = 'B'
    PDs.loc[(PDs['Montecarlo DaO Stochastic Barrier'] >= 14.21), 'MdaoSB Rating'] = 'CCC'

    PDs.loc[(PDs['MdaoSB Rating'] == 'AAA') | (PDs['MdaoSB Rating'] == 'AA') | (PDs['MdaoSB Rating'] == 'A'), 'MdaoSBSimpl Rating'] = 'A'
    PDs.loc[(PDs['MdaoSB Rating'] == 'BBB'), 'MdaoSBSimpl Rating'] = 'B'
    PDs.loc[(PDs['MdaoSB Rating'] == 'BB') | (PDs['MdaoSB Rating'] == 'B'), 'MdaoSBSimpl Rating'] = 'HY'
    PDs.loc[(PDs['MdaoSB Rating'] == 'CCC'), 'MdaoSBSimpl Rating'] = 'D'

    tickerCol = pd.DataFrame(np.full(len(PDs['Montecarlo DaO Stochastic Barrier']), yahooFinanceTicker)).set_index(
        PDs.index).set_axis(['Ticker'], axis=1)

    PDs = pd.concat([tickerCol, PDs], axis=1)

    newIndex = ['2018', '2019', '2020', '2021']

    PDs = PDs.set_index(pd.Series(newIndex))

    # PDs.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Test Dataset\Classification Results\AMZNProva.xlsx")

    return PDs


def allModelClassification (yahooFinanceTicker, maturity, freq = 'yearly'):

    import MertonModel as mm
    import JumpDiffusionModelClosedForm as jp
    import montecarloJumpDiffisionDaO as mdao
    import DownAndOutCall as dao
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

    b = mm.MertonProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    c = jp.jumpDiffusionProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    d = dao.downAndOutProbabilityOfDefaultRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)
    e = mdao.montecarloJumpDiffusionDownAndOutOnRealStocks(realStock, maturity, parameters='BhSh', calibrate=True, freq=freq)

    if freq == 'yearly':
        stockTS = (((yf.Ticker(realStock).history(start='2018-12-31', end='2021-12-31'))[
            'Close']).pct_change() * 100).cumsum()
    if freq == 'quarterly':
        stockTS = (((yf.Ticker(realStock).history(start='2021-09-25', end='2022-06-25'))[
            'Close']).pct_change() * 100).cumsum()

    #print('Standard Merton - BhSh Parameters', b)
    #print('Jump-Diffusion Merton - BhSh Parameters', c)
    #print('Down-And-Out Merton - Fixed Barrier', d)
    #print('Montecarlo Down-And-Out Model', e)

    dateIndex = list()
    for date in stockTS.index:
        dateIndex.append(datetime.date(date))
    stockTS.index = pd.Series(dateIndex)

    PDs = pd.concat([b, c, d, e], axis=1).set_axis(['Standard Merton - BhSh Parameters', 'Jump-Diffusion Merton - BhSh Parameters',
                                                    'Down-And-Out Merton - Fixed Barrier', 'Montecarlo Down-And-Out Model'],
                                                   axis=1)

    PDs.index = pd.to_datetime(PDs.index)
    dateIndexPD = list()
    for value in PDs.index:
        dateIndexPD.append(datetime.date(value))
    PDs.index = pd.Series(dateIndexPD)

    PDs = PDs.sort_index()

    # let's set the classification Framework

    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0) & (PDs['Standard Merton - BhSh Parameters'] < 0.05), 'Merton Rating'] = 'AAA'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0.05) & (PDs['Standard Merton - BhSh Parameters'] < 0.09), 'Merton Rating'] = 'AA'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0.09) & (PDs['Standard Merton - BhSh Parameters'] < 0.23), 'Merton Rating'] = 'A'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0.23) & (PDs['Standard Merton - BhSh Parameters'] < 1.16), 'Merton Rating'] = 'BBB'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 1.16) & (PDs['Standard Merton - BhSh Parameters'] < 5.44), 'Merton Rating'] = 'BB'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 5.44) & (PDs['Standard Merton - BhSh Parameters'] < 14.21), 'Merton Rating'] = 'B'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 14.21), 'Merton Rating'] = 'CCC'

    PDs.loc[(PDs['Merton Rating'] == 'AAA') | (PDs['Merton Rating'] == 'AA') | (PDs['Merton Rating'] == 'A'), 'Merton Simplified Rating'] = 'A'
    PDs.loc[(PDs['Merton Rating'] == 'BBB'), 'Merton Simplified Rating'] = 'B'
    PDs.loc[(PDs['Merton Rating'] == 'BB') | (PDs['Merton Rating'] == 'B'), 'Merton Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Merton Rating'] == 'CCC'), 'Merton Simplified Rating'] = 'D'


    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 0.05), 'Jump Diffusion Rating'] = 'AAA'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0.05) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 0.09), 'Jump Diffusion Rating'] = 'AA'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0.09) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 0.23), 'Jump Diffusion Rating'] = 'A'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0.23) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 1.16), 'Jump Diffusion Rating'] = 'BBB'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 1.16) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 5.44), 'Jump Diffusion Rating'] = 'BB'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 5.44) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 14.21), 'Jump Diffusion Rating'] = 'B'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 14.21), 'Jump Diffusion Rating'] = 'CCC'

    PDs.loc[(PDs['Jump Diffusion Rating'] == 'AAA') | (PDs['Jump Diffusion Rating'] == 'AA') | (PDs['Jump Diffusion Rating'] == 'A'), 'JD Simplified Rating'] = 'A'
    PDs.loc[(PDs['Jump Diffusion Rating'] == 'BBB'), 'JD Simplified Rating'] = 'B'
    PDs.loc[(PDs['Jump Diffusion Rating'] == 'BB') | (PDs['Jump Diffusion Rating'] == 'B'), 'JD Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Jump Diffusion Rating'] == 'CCC'), 'JD Simplified Rating'] = 'D'


    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 0.05), 'Down-And-Out Rating'] = 'AAA'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0.05) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 0.09), 'Down-And-Out Rating'] = 'AA'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0.09) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 0.23), 'Down-And-Out Rating'] = 'A'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0.23) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 1.16), 'Down-And-Out Rating'] = 'BBB'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 1.16) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 5.44), 'Down-And-Out Rating'] = 'BB'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 5.44) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 14.21), 'Down-And-Out Rating'] = 'B'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 14.21), 'Down-And-Out Rating'] = 'CCC'

    PDs.loc[(PDs['Down-And-Out Rating'] == 'AAA') | (PDs['Down-And-Out Rating'] == 'AA') | (PDs['Down-And-Out Rating'] == 'A'), 'DaO Simplified Rating'] = 'A'
    PDs.loc[(PDs['Down-And-Out Rating'] == 'BBB'), 'DaO Simplified Rating'] = 'B'
    PDs.loc[(PDs['Down-And-Out Rating'] == 'BB') | (PDs['Down-And-Out Rating'] == 'B'), 'DaO Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Down-And-Out Rating'] == 'CCC'), 'DaO Simplified Rating'] = 'D'


    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0) & (
                PDs['Montecarlo Down-And-Out Model'] < 0.05), 'Montecarlo Down-And-Out Rating'] = 'AAA'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0.05) & (
                PDs['Montecarlo Down-And-Out Model'] < 0.09), 'Montecarlo Down-And-Out Rating'] = 'AA'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0.09) & (
                PDs['Montecarlo Down-And-Out Model'] < 0.23), 'Montecarlo Down-And-Out Rating'] = 'A'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0.23) & (
                PDs['Montecarlo Down-And-Out Model'] < 1.16), 'Montecarlo Down-And-Out Rating'] = 'BBB'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 1.16) & (
                PDs['Montecarlo Down-And-Out Model'] < 5.44), 'Montecarlo Down-And-Out Rating'] = 'BB'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 5.44) & (
                PDs['Montecarlo Down-And-Out Model'] < 14.21), 'Montecarlo Down-And-Out Rating'] = 'B'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 14.21), 'Montecarlo Down-And-Out Rating'] = 'CCC'

    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'AAA') | (PDs['Montecarlo Down-And-Out Rating'] == 'AA') | (PDs['Montecarlo Down-And-Out Rating'] == 'A'), 'MDaO Simplified Rating'] = 'A'
    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'BBB'), 'MDaO Simplified Rating'] = 'B'
    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'BB') | (PDs['Montecarlo Down-And-Out Rating'] == 'B'), 'MDaO Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'CCC'), 'MDaO Simplified Rating'] = 'D'


    tickerCol = pd.DataFrame(np.full(len(PDs['Montecarlo Down-And-Out Rating']), yahooFinanceTicker)).set_index(PDs.index).set_axis(['Ticker'], axis = 1)

    PDs = pd.concat([tickerCol, PDs], axis = 1)

    newIndex = ['2018', '2019', '2020', '2021']

    PDs = PDs.set_index(pd.Series(newIndex))

    #PDs.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Test Dataset\Classification Results\AMZNProva.xlsx")

    return PDs


def allModelClassification2021 (yahooFinanceTicker, maturity, freq = 'yearly'):

    import MertonModel as mm
    import JumpDiffusionModelClosedForm as jp
    import montecarloJumpDiffisionDaO as mdao
    import DownAndOutCall as dao
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

    b = mm.MertonProbabilityOfDefaultRealStocks2021(realStock, maturity, parameters='BhSh', calibrate=True)
    c = jp.jumpDiffusionProbabilityOfDefaultRealStocks2021(realStock, maturity, parameters='BhSh', calibrate=True)
    d = dao.downAndOutProbabilityOfDefaultRealStocks2021(realStock, maturity, parameters='BhSh', calibrate=True)
    e = mdao.montecarloJumpDiffusionDownAndOutOnRealStocks2021(realStock, maturity, parameters='BhSh', calibrate=True)

    #print('Standard Merton - BhSh Parameters', b)
    #print('Jump-Diffusion Merton - BhSh Parameters', c)
    #print('Down-And-Out Merton - Fixed Barrier', d)
    #print('Montecarlo Down-And-Out Model', e)

    PDs = pd.concat([b, c, d, e], axis=1).set_axis(['Standard Merton - BhSh Parameters', 'Jump-Diffusion Merton - BhSh Parameters',
                                                    'Down-And-Out Merton - Fixed Barrier', 'Montecarlo Down-And-Out Model'],
                                                   axis=1)

    PDs.index = pd.to_datetime(PDs.index)
    dateIndexPD = list()
    for value in PDs.index:
        dateIndexPD.append(datetime.date(value))
    PDs.index = pd.Series(dateIndexPD)

    PDs = PDs.sort_index()

    # let's set the classification Framework

    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0) & (PDs['Standard Merton - BhSh Parameters'] < 0.05), 'Merton Rating'] = 'AAA'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0.05) & (PDs['Standard Merton - BhSh Parameters'] < 0.09), 'Merton Rating'] = 'AA'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0.09) & (PDs['Standard Merton - BhSh Parameters'] < 0.23), 'Merton Rating'] = 'A'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 0.23) & (PDs['Standard Merton - BhSh Parameters'] < 1.16), 'Merton Rating'] = 'BBB'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 1.16) & (PDs['Standard Merton - BhSh Parameters'] < 5.44), 'Merton Rating'] = 'BB'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 5.44) & (PDs['Standard Merton - BhSh Parameters'] < 14.21), 'Merton Rating'] = 'B'
    PDs.loc[(PDs['Standard Merton - BhSh Parameters'] >= 14.21), 'Merton Rating'] = 'CCC'

    PDs.loc[(PDs['Merton Rating'] == 'AAA') | (PDs['Merton Rating'] == 'AA') | (PDs['Merton Rating'] == 'A'), 'Merton Simplified Rating'] = 'A'
    PDs.loc[(PDs['Merton Rating'] == 'BBB'), 'Merton Simplified Rating'] = 'B'
    PDs.loc[(PDs['Merton Rating'] == 'BB') | (PDs['Merton Rating'] == 'B'), 'Merton Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Merton Rating'] == 'CCC'), 'Merton Simplified Rating'] = 'D'


    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 0.05), 'Jump Diffusion Rating'] = 'AAA'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0.05) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 0.09), 'Jump Diffusion Rating'] = 'AA'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0.09) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 0.23), 'Jump Diffusion Rating'] = 'A'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 0.23) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 1.16), 'Jump Diffusion Rating'] = 'BBB'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 1.16) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 5.44), 'Jump Diffusion Rating'] = 'BB'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 5.44) & (
                PDs['Jump-Diffusion Merton - BhSh Parameters'] < 14.21), 'Jump Diffusion Rating'] = 'B'
    PDs.loc[(PDs['Jump-Diffusion Merton - BhSh Parameters'] >= 14.21), 'Jump Diffusion Rating'] = 'CCC'

    PDs.loc[(PDs['Jump Diffusion Rating'] == 'AAA') | (PDs['Jump Diffusion Rating'] == 'AA') | (PDs['Jump Diffusion Rating'] == 'A'), 'JD Simplified Rating'] = 'A'
    PDs.loc[(PDs['Jump Diffusion Rating'] == 'BBB'), 'JD Simplified Rating'] = 'B'
    PDs.loc[(PDs['Jump Diffusion Rating'] == 'BB') | (PDs['Jump Diffusion Rating'] == 'B'), 'JD Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Jump Diffusion Rating'] == 'CCC'), 'JD Simplified Rating'] = 'D'


    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 0.05), 'Down-And-Out Rating'] = 'AAA'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0.05) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 0.09), 'Down-And-Out Rating'] = 'AA'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0.09) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 0.23), 'Down-And-Out Rating'] = 'A'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 0.23) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 1.16), 'Down-And-Out Rating'] = 'BBB'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 1.16) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 5.44), 'Down-And-Out Rating'] = 'BB'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 5.44) & (
                PDs['Down-And-Out Merton - Fixed Barrier'] < 14.21), 'Down-And-Out Rating'] = 'B'
    PDs.loc[(PDs['Down-And-Out Merton - Fixed Barrier'] >= 14.21), 'Down-And-Out Rating'] = 'CCC'

    PDs.loc[(PDs['Down-And-Out Rating'] == 'AAA') | (PDs['Down-And-Out Rating'] == 'AA') | (PDs['Down-And-Out Rating'] == 'A'), 'DaO Simplified Rating'] = 'A'
    PDs.loc[(PDs['Down-And-Out Rating'] == 'BBB'), 'DaO Simplified Rating'] = 'B'
    PDs.loc[(PDs['Down-And-Out Rating'] == 'BB') | (PDs['Down-And-Out Rating'] == 'B'), 'DaO Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Down-And-Out Rating'] == 'CCC'), 'DaO Simplified Rating'] = 'D'


    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0) & (
                PDs['Montecarlo Down-And-Out Model'] < 0.05), 'Montecarlo Down-And-Out Rating'] = 'AAA'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0.05) & (
                PDs['Montecarlo Down-And-Out Model'] < 0.09), 'Montecarlo Down-And-Out Rating'] = 'AA'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0.09) & (
                PDs['Montecarlo Down-And-Out Model'] < 0.23), 'Montecarlo Down-And-Out Rating'] = 'A'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 0.23) & (
                PDs['Montecarlo Down-And-Out Model'] < 1.16), 'Montecarlo Down-And-Out Rating'] = 'BBB'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 1.16) & (
                PDs['Montecarlo Down-And-Out Model'] < 5.44), 'Montecarlo Down-And-Out Rating'] = 'BB'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 5.44) & (
                PDs['Montecarlo Down-And-Out Model'] < 14.21), 'Montecarlo Down-And-Out Rating'] = 'B'
    PDs.loc[(PDs['Montecarlo Down-And-Out Model'] >= 14.21), 'Montecarlo Down-And-Out Rating'] = 'CCC'

    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'AAA') | (PDs['Montecarlo Down-And-Out Rating'] == 'AA') | (PDs['Montecarlo Down-And-Out Rating'] == 'A'), 'MDaO Simplified Rating'] = 'A'
    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'BBB'), 'MDaO Simplified Rating'] = 'B'
    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'BB') | (PDs['Montecarlo Down-And-Out Rating'] == 'B'), 'MDaO Simplified Rating'] = 'HY'
    PDs.loc[(PDs['Montecarlo Down-And-Out Rating'] == 'CCC'), 'MDaO Simplified Rating'] = 'D'


    tickerCol = pd.DataFrame(np.full(len(PDs['Montecarlo Down-And-Out Rating']), yahooFinanceTicker)).set_index(PDs.index).set_axis(['Ticker'], axis = 1)

    PDs = pd.concat([tickerCol, PDs], axis = 1)

    newIndex = ['2021']

    PDs = PDs.set_index(pd.Series(newIndex))

    #PDs.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Test Dataset\Classification Results\AMZNProva.xlsx")

    return PDs
