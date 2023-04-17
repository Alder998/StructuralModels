import MertonModel as mm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

maturityVector = np.linspace(0,5, 10)

df = pd.read_excel(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Big Dataset Calibration\Big_dataset_calibration_File.xlsx')

# get the Optimal Parameters for each maturity
parameterForEachMaturity = list()
driftForEachMaturity = list()
volatilityForEachMaturity = list()
for maturityLevel in maturityVector:
   a = mm.calibrateMerton(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Calibration\Big Dataset Calibration\Big_dataset_calibration_File.xlsx',
                          maturityLevel)
   parameterForEachMaturity.append(a)
   driftForEachMaturity.append(a[0])
   volatilityForEachMaturity.append(a[1])

# plot them

#plt.figure(figsize = (12, 5))
#plt.plot(driftForEachMaturity)
#plt.show()

#plt.figure(figsize = (12, 5))
#plt.plot(volatilityForEachMaturity)
#plt.show()

# let's see the path of Mean Squared Error wrt to maturity

final = list()
for timeIndex in range(0, len(maturityVector)):

    meanBase = list()
    for firm in range(0, len(df['Total Assets Value'])):

         model = mm.MertonModelEquityPricing(df['Total Assets Value'][firm], df['Total Market Debt Value'][firm], driftForEachMaturity[timeIndex],
                                        volatilityForEachMaturity[timeIndex], timeIndex)

         meanBase.append(model)

    mean = ((((pd.Series(meanBase) - df['Total Market Equity Value'])/df['Total Market Equity Value'])/100)**2).mean()

    final.append(mean)

plt.figure(figsize = (12, 5))
plt.scatter(maturityVector, final)
plt.plot(maturityVector,final)
plt.axhline(y = 0, color = 'black', linestyle = 'dashed')
plt.show()
