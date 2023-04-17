# Here we try to see the difference between these two models with real stocks

import MertonModel as mm
import JumpDiffusionModelClosedForm as jp
import matplotlib.pyplot as plt
import yfinance as yf


#a = jp.CompareProbabilityOfDefaultRealStocks('SSL', 5, False, True)

b = mm.MertonModelEquityPricingRealStocks('AAPL', 1, True)

print(b['Value Of Total Equity'][3])