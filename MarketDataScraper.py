def getValueDebtAverage(yahooFinanceTicker):

    import yfinance as yf
    import pandas as pd

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()

    stockDebtR = stock.transpose()[

        ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
    stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
        'Other Current Liab']) + (stockDebtR['Long Term Debt'])

    stockEquity = stock.transpose()['Total Stockholder Equity']
    sigmaEquity = ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
    sigmaDebt = 0.05 + (0.25 * sigmaEquity)

    stockAssets = stockEquity + stockDebt
    sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

    stockValueToDebt = (stockAssets / stockDebt).mean()

    return stockValueToDebt


def getValueToDebt(yahooFinanceTicker):

    import yfinance as yf
    import pandas as pd

    stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()

    stockDebtR = stock.transpose()[

        ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
    stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
        'Other Current Liab']) + (stockDebtR['Long Term Debt'])

    stockEquity = stock.transpose()['Total Stockholder Equity']
    sigmaEquity = ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
    sigmaDebt = 0.05 + (0.25 * sigmaEquity)

    stockAssets = stockEquity + stockDebt
    sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

    stockValueToDebt = (stockAssets / stockDebt)

    return stockValueToDebt


def getDebtValue (yahooFinanceTicker, freq ='yearly', BhSh = False):

    import yfinance as yf
    import pandas as pd

    if freq == 'yearly':
        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    if freq == 'quarterly':
        stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

    if BhSh == False:

        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (stockDebtR['Long Term Debt'])

        return stockDebt

    if BhSh == True:

        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

        return stockDebt


def getAssetValue (yahooFinanceTicker, BhSh = False, freq = 'yearly'):

    import yfinance as yf
    import pandas as pd

    if freq == 'yearly':
        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    if freq == 'quarterly':
        stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

    if BhSh == True:

        stockDebtR = stock.transpose()[
            ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
        stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
            'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

        stockEquity = stock.transpose()['Total Stockholder Equity']
        sigmaEquity = ((yf.Ticker(yahooFinanceTicker).history('1Y', '1d')['Close']).pct_change()).std()
        sigmaDebt = 0.05 + (0.25 * sigmaEquity)

        stockAssets = stockEquity + stockDebt

        return stockAssets

    if BhSh == False:

        stockAssets = stock.transpose()['Total Assets']

        return stockAssets


def getEquityValue (yahooFinanceTicker, freq = 'yearly'):

    import yfinance as yf
    import pandas as pd

    if freq == 'yearly':
        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    if freq == 'quarterly':
        stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

    stockEquity = stock.transpose()['Total Stockholder Equity']

    return stockEquity


def getNewsIndex(yahooFinanceTicker):

    import yfinance as yf
    import pandas as pd

    news = list()
    for newsNumber in range(8):
        news.append(yf.Ticker(yahooFinanceTicker).get_news()[newsNumber]['title'])

    news = pd.Series(news)

    relatedWords = ['drop', 'fall', 'distress', 'Gas', 'shortage', 'hardship', 'credit', 'oil', 'crisis', 'instability',
                    'CDS', 'price', 'Financial', 'bearish', 'situation', 'inflation', 'recession', 'Interest', 'rates',
                    'Spread',
                    'sentiment', 'rating', 'default', 'drawdown', 'volatile', 'decrease', 'decline', 'shock',
                    'oli price',
                    'energy']

    finalNews = list()
    for word in relatedWords:
        finalNews.append(pd.Series(news.str.contains(word, case=False).unique()))

    finalNews = pd.concat([series for series in pd.Series(finalNews)], axis=0)

    newsIndex = finalNews[finalNews == True]

    return len(newsIndex)


def getBhShVolatility (yahooFinanceTicker, calibratedVolatility, freq = 'yearly'):

    import yfinance as yf
    import pandas as pd

    if freq == 'yearly':
        stock = yf.Ticker(yahooFinanceTicker).get_balancesheet()
    if freq == 'quarterly':
        stock = yf.Ticker(yahooFinanceTicker).quarterly_balance_sheet

    stockDebtR = stock.transpose()[
        ['Total Current Liabilities', 'Accounts Payable', 'Other Current Liab', 'Long Term Debt']]
    stockDebt = (stockDebtR['Total Current Liabilities'] - stockDebtR['Accounts Payable'] - stockDebtR[
        'Other Current Liab']) + (0.5 * stockDebtR['Long Term Debt'])

    stockEquity = stock.transpose()['Total Stockholder Equity']
    sigmaEquity = calibratedVolatility
    sigmaDebt = 0.05 + (0.25 * sigmaEquity)

    stockAssets = stockEquity + stockDebt
    sigmaAssetBhSh = ((stockEquity / stockAssets) * sigmaEquity) + ((stockDebt / stockAssets) * sigmaDebt)

    stockValueToDebt = stockAssets / stockDebt

    return sigmaAssetBhSh
