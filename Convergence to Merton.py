# convergence to Merton for two Analytical models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MertonModel as mm
import JumpDiffusionModelClosedForm as jd
import DownAndOutCall as dao
import montecarloJumpDiffisionDaO as mdao

# TABLE OF PD ACCORDING THE 4 POSSIBLE PARAMETERS

lenM = 100
ratioVector = np.linspace(1,3, lenM)          # Value/Debt Ratio
driftVector = np.linspace(-0.5, 0.5, lenM)    # Drift
volVector = np.linspace(0.001, 2, lenM)      # Volatility
matVector = np.linspace(0.5, 3, lenM)       # Maturity
lamdaVector = np.linspace(5, 0, lenM)     # Intensity of Jump
meanVector = np.linspace(-0.1, 0.1, lenM)    # Mean Of Jump
volJumpVector = np.linspace(-0.1, 2, lenM)    # Volatility Of Jump

# define the comparison Term

MertonBenchmark = list()
for singleMaturity in matVector:
    mm1 = mm.MertonModelEquityPricing(1000, 500, 0.05, 0.3, singleMaturity)
    MertonBenchmark.append(mm1)
MertonBenchmark = pd.Series(MertonBenchmark)

# JD Convergence

lambdas = np.linspace(700, 0, 20)

convergenceForLambda = list()
for diffLambda in lambdas:

    PDAccordingMaturity = list()
    for singleMaturity in matVector:
        mmll = dao.downAndOutEquityPricing(1000, 500, diffLambda, 0.05, 0.3, singleMaturity)
        PDAccordingMaturity.append(mmll)
    convergenceForLambda.append(PDAccordingMaturity)

toPlotlambda = list()
for array in pd.Series(convergenceForLambda):
    toPlotlambda.append(pd.Series(array))
toPlotlambda = pd.concat([series for series in toPlotlambda], axis = 1)
toPlotlambda = toPlotlambda

palette = ['paleturquoise', 'paleturquoise', 'cyan', 'aqua', 'mediumturquoise', 'darkturquoise','deepskyblue', 'dodgerblue', 'blue', 'navy',
           'paleturquoise', 'paleturquoise', 'cyan', 'aqua', 'mediumturquoise', 'darkturquoise','deepskyblue', 'dodgerblue', 'blue', 'navy']
plt.figure(figsize = (12, 7))
for colorCounter in range(len(palette)):
    plt.plot(matVector, toPlotlambda[colorCounter], color = palette[colorCounter])
plt.scatter(x = matVector, y = MertonBenchmark, color = 'red', label = 'Merton values')
plt.legend()
plt.xlabel('Maturity')
plt.ylabel('Equity Price')
plt.title('Figure 1.29')

lambdaLeg = list()
for value in lambdas:
    lambdaLeg.append(round(value, 2))
listLeg = list()
for value in lambdaLeg:
    listLeg.append('H =' + str(value))

plt.legend(listLeg)
#plt.show()

plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\JD convergence Merton.png", dpi = 1500)