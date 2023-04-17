# Smart sensitivity analysis on Merton Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MertonModel as mm
import DownAndOutCall as dao

# TABLE OF PD ACCORDING THE 4 POSSIBLE PARAMETERS

lenM = 100
ratioVector = np.linspace(1,3, lenM)    # Value/Debt Ratio
driftVector = np.linspace(-0.5,0.5, lenM)    # Drift
volVector = np.linspace(0.001, 3, lenM)      # Volatility
matVector = np.linspace(0.5, 5, lenM)       # Maturity

# PD according maturity

PDAccordingMaturity = list()
for singleMaturity in matVector:
    mm1 = dao.downAndOutProbabilityOfDefaultWithValueToDebt(2, 0.05, 0.3, singleMaturity)
    PDAccordingMaturity.append(mm1)
PDAccordingMaturity = pd.Series(PDAccordingMaturity)*100

# PD according Value/Debt Ratio

PDAccordingVD = list()
for singleRatio in ratioVector:
    mm2 = dao.downAndOutProbabilityOfDefaultWithValueToDebt(singleRatio, 0.05, 0.3, 1)
    PDAccordingVD.append(mm2)
PDAccordingVD = pd.Series(PDAccordingVD)*100

# PD according drift parameter

PDAccordingDrift = list()
for singleDrift in driftVector:
    mm3 = dao.downAndOutProbabilityOfDefaultWithValueToDebt(2, singleDrift, 0.3, 1)
    PDAccordingDrift.append(mm3)
PDAccordingDrift = pd.Series(PDAccordingDrift)*100

# PD according the volatility parameter

PDAccordingVolatility = list()
for singleVolatility in volVector:
    mm4 = dao.downAndOutProbabilityOfDefaultWithValueToDebt(2, 0.05, singleVolatility, 1)
    PDAccordingVolatility.append(mm4)
PDAccordingVolatility = pd.Series(PDAccordingVolatility)*100

# Start setting the subplots

plt.figure(figsize = (10, 10))

plt.subplots_adjust(hspace=0.300, wspace=0.305)

plt.subplot(2, 2, 1)
plt.scatter(x = matVector, y = PDAccordingMaturity, color = 'blue')
plt.xlabel('Maturity')
plt.ylabel('PD (%)')
plt.title('Figure 2.18', fontsize=10)

plt.subplot(2, 2, 2)
plt.scatter(x = ratioVector, y = PDAccordingVD, color = 'red')
plt.xlabel('Value/Debt Ratio')
plt.ylabel('PD (%)')
plt.title('Figure 2.19', fontsize=10)

plt.subplot(2, 2, 3)
plt.scatter(x = driftVector, y = PDAccordingDrift, color = 'black')
plt.xlabel('Drift')
plt.ylabel('PD (%)')
plt.title('Figure 2.20', fontsize=10)

plt.subplot(2, 2, 4)
plt.scatter(x = volVector, y = PDAccordingVolatility, color = 'green')
plt.xlabel('Volatility')
plt.ylabel('PD (%)')
plt.title('Figure 2.21', fontsize=10)

#plt.show()

plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\Down-And-Out Sensitivity", dpi = 1500)