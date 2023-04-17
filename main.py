import MertonModel as mm
import pandas as pd
import yfinance as yf
from datetime import datetime

marketCap = yf.Ticker('AMZN').get_balancesheet()

print(marketCap)