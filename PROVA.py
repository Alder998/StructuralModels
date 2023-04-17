# Set the framework for Least-Squares Montecarlo
import math

import pandas as pd
import GraphicalInterface as gr
import numpy as np
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

# Initialize the parameters

dfM = gr.MertonProbabilityOfDefaultRealStocksForGraphicalInterface('AMZN', 1, 'quarterly')
dfJD = gr.JDProbabilityOfDefaultRealStocksForGraphicalInterface('AMZN', 1, 'quarterly')

final = pd.concat([dfM, dfJD], axis = 1)

print(final)

#mdao.compareMontecarloImplementations('HTZ', 1, 'yearly')

#AssetValue = 365725000000
#maturity = 2
#drift = 0.02
#meanOfJump = 0
#volOfJump = 0.1
#lambdaJump = 0.2
#numberOfTimeSteps = 250 * maturity
#numberOfPaths = 10000
#sigmaAssets = 0.3
#valueOfDebt = 258578000000
#valueOfBarrier = 258578000000
#
#mdao.montecarloJumpDiffusionDownAndOutPD(AssetValue, valueOfDebt, valueOfDebt, drift, sigmaAssets, lambdaJump, meanOfJump, volOfJump, numberOfTimeSteps, numberOfPaths, maturity,
                                         #plot = True)

#a = dao.downAndOutProbabilityOfDefaultRealStocks('CO.PA', 1, calibrate = True, freq = 'yearly', parameters = 'MM')

#b = dao.downAndOutProbabilityOfDefault(24627000000, 18937000000, 0.032250787680019176, 0.26289007470771747, 1)

#print('PD (%):',a )

#model.allModelsComparisonOnRealStock('HTZ', 1, freq = 'yearly')


#sample = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Test Dataset\DB Test Clean.xlsx")
#
#sampleY = pd.to_datetime(sample['rating_action_date'])
#index = list()
#for i in sampleY:
#    index.append(str(i.year))
#sample = sample.set_index(pd.Series(index))
#
#tickers = (sample['Ticker']).unique()[41:len(sample['Ticker'])]
#
#ratingList = list()
#for ticker in tickers:
#    a = model.allModelClassification(ticker, 1, freq = 'yearly')
#    realativeDate = pd.Series(sample.index[sample['Ticker'] == ticker])
#    sampleR = sample[sample['Ticker'] == ticker].set_index(realativeDate)
#
#    print(sampleR[['Ticker', 'rating']])
#
#    b = sampleR[['Ticker', 'rating', 'ratingSimplified']].join(a, rsuffix='Estimate')
#    ratingList.append(b)
#
#ratingList = pd.concat([df for df in ratingList], axis = 0)
#
#ratingList.to_excel(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Test Dataset\Classification Results\Prova2021.xlsx")
#
#end = time.time()
#

#print('\n')
#print('Time Elapsed:', round((end-start)/60, 2), 'Minutes')

a = model.allModelsComparisonOnRealStock('ENEL.MI', 1, freq = 'quarterly')
#b = model.allModelsComparisonOnRealStock('IBE.MC', 1, freq = 'quarterly')
#c = model.allModelsComparisonOnRealStock('EDF.PA', 1, freq = 'quarterly')
#d = model.allModelsComparisonOnRealStock('EOAN.DE', 1, freq = 'quarterly')
#
#a.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ENEL.xlsx")
#b.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\IBERDROLA DB.xlsx")
#c.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Electricite de France DB.xlsx")
#d.to_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\EON DB.xlsx")


ENEL = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\ENEL.xlsx").set_index('Unnamed: 0').pct_change()*100
IBER = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\IBERDROLA DB.xlsx").set_index('Unnamed: 0').pct_change()*100
EDF = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\Electricite de France DB.xlsx").set_index('Unnamed: 0').pct_change()*100
EON = pd.read_excel(r"C:\Users\39328\OneDrive\Desktop\Storing di cose di dubbia utilità futura\EON DB.xlsx").set_index('Unnamed: 0').pct_change()*100


plt.figure(figsize = (20, 8))

plt.subplots_adjust(wspace=0.16, hspace=0.4)
plt.rc('font', size=8)

plt.subplot(2, 2, 1)

plt.plot(ENEL['Standard Merton - BhSh Parameters'], color = 'blue')
plt.scatter(ENEL.index, ENEL['Standard Merton - BhSh Parameters'], color = 'blue', label = 'Merton')
plt.plot(ENEL['Jump-Diffusion Merton - BhSh Parameters'], color = 'red')
plt.scatter(ENEL.index, ENEL['Jump-Diffusion Merton - BhSh Parameters'], color = 'red', label = 'Jump-Diff.')
plt.plot(ENEL['Down-And-Out Merton - Fixed Barrier'], color = 'green')
plt.scatter(ENEL.index, ENEL['Down-And-Out Merton - Fixed Barrier'], color = 'green', label = 'Down-And-Out')
plt.plot(ENEL['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange')
plt.scatter(ENEL.index, ENEL['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange', label = 'Montecarlo')
plt.ylabel('PD % Change')
plt.xticks(rotation=45)
plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
plt.title('ENEL')
plt.legend()

plt.subplot(2, 2, 2)

plt.plot(IBER['Standard Merton - BhSh Parameters'], color = 'blue')
plt.scatter(IBER.index, IBER['Standard Merton - BhSh Parameters'], color = 'blue', label = 'Merton')
plt.plot(IBER['Jump-Diffusion Merton - BhSh Parameters'], color = 'red')
plt.scatter(IBER.index, IBER['Jump-Diffusion Merton - BhSh Parameters'], color = 'red', label = 'Jump-Diff.')
plt.plot(IBER['Down-And-Out Merton - Fixed Barrier'], color = 'green')
plt.scatter(IBER.index, IBER['Down-And-Out Merton - Fixed Barrier'], color = 'green', label = 'Down-And-Out')
plt.plot(IBER['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange')
plt.scatter(IBER.index, IBER['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange', label = 'Montecarlo')
plt.ylabel('PD % Change')
plt.xticks(rotation=45)
plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
plt.title('Iberdrola')
plt.legend()


plt.subplot(2, 2, 3)

plt.plot(EDF['Standard Merton - BhSh Parameters'], color = 'blue')
plt.scatter(EDF.index, EDF['Standard Merton - BhSh Parameters'], color = 'blue', label = 'Merton')
plt.plot(EDF['Jump-Diffusion Merton - BhSh Parameters'], color = 'red')
plt.scatter(EDF.index, EDF['Jump-Diffusion Merton - BhSh Parameters'], color = 'red', label = 'Jump-Diff.')
plt.plot(EDF['Down-And-Out Merton - Fixed Barrier'], color = 'green')
plt.scatter(EDF.index, EDF['Down-And-Out Merton - Fixed Barrier'], color = 'green', label = 'Down-And-Out')
plt.plot(EDF['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange')
plt.scatter(EDF.index, EDF['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange', label = 'Montecarlo')
plt.ylabel('PD % Change')
plt.xticks(rotation=45)
plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
plt.title('EDF')
plt.legend()


plt.subplot(2, 2, 4)

plt.plot(EON['Standard Merton - BhSh Parameters'], color = 'blue')
plt.scatter(EON.index, EON['Standard Merton - BhSh Parameters'], color = 'blue', label = 'Merton')
plt.plot(EON['Jump-Diffusion Merton - BhSh Parameters'], color = 'red')
plt.scatter(EON.index, EON['Jump-Diffusion Merton - BhSh Parameters'], color = 'red', label = 'Jump-Diff.')
plt.plot(EON['Down-And-Out Merton - Fixed Barrier'], color = 'green')
plt.scatter(EON.index, EON['Down-And-Out Merton - Fixed Barrier'], color = 'green', label = 'Down-And-Out')
plt.plot(EON['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange')
plt.scatter(EON.index, EON['Montecarlo Jump-Diffusion Down-And-Out'], color = 'orange', label = 'Montecarlo')
plt.ylabel('PD % Change')
plt.xticks(rotation=45)
plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
plt.title('EON')
plt.legend()

plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\Pres_1", dpi = 1500)

plt.show()


#jp.calibrateJumpDiffusion('AMZN', 1)

#print(mkt.getValueDebtAverage('HTZ'))

#mm.compareMertonImplementationOnRealStocks('HTZ', 1, 'yearly')

#for newsNumber in range(8):
#    print(yf.Ticker('AAPL').get_news()[newsNumber]['title'])

# importo la serie storica

#a = pr.barrierProcess('AAPL', 250, 10000)
#b = pr.barrierProcess('HTZ', 250, 10000)

#plt.figure(figsize=(12, 5))
#plt.plot(a, label = 'Healthy Stock')
#plt.plot(b, label = 'Distressed Stock')
#plt.xlabel('Days')
#plt.ylabel('Stock Price')
#plt.title('Barrier Process Comparison')
#plt.legend()
#plt.show()

#a = mdao.montecarloJumpDiffusionDownAndOutOnRealStocks('AAPL', 1, calibrate = True, parameters = 'BhSh', freq = 'yearly')

#print(mkt.getAssetValue('CS', BhSh = True, freq = 'quarterly')[3])

#a = mdao.montecarloJumpDiffusionDownAndOutPDStochasticBarrier('CS', mkt.getAssetValue('CS', BhSh = True, freq = 'quarterly')[3],
#                                                              mkt.getDebtValue('CS', BhSh = True, freq = 'quarterly')[3], drift,
#                                                             sigmaAssets, lambdaJump, meanOfJump,
#                                                              volOfJump, numberOfTimeSteps, numberOfPaths, maturity, plot = True)
#
#print('CS PD:', a)
#

#cc = mdao.montecarloJumpDiffusionDownAndOutOnRealStocks('AAPL', 1, calibrate = True, parameters='BhSh', freq = 'yearly')

#mdao.compareMontecarloImplementations('CS', 1, 'quarterly')

#a = mdao.calibrateStochasticBarrier('AMZN', 500)
#b = mdao.calibrateStochasticBarrier('HTZ', 500)
#
#plt.figure(figsize=(12, 5))
#plt.plot(a, label = 'Healthy stock')
#plt.plot(b, label = 'Distressed Stock')
#plt.legend()
#plt.show()