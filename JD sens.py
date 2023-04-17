# Smart sensitivity analysis on Jump-Diffusion Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import JumpDiffusionModelClosedForm as jp

# TABLE OF PD ACCORDING THE 4 POSSIBLE PARAMETERS

lenM = 100
ratioVector = np.linspace(1,3, lenM)          # Value/Debt Ratio
driftVector = np.linspace(-0.5, 0.5, lenM)    # Drift
volVector = np.linspace(0.001, 2, lenM)      # Volatility
matVector = np.linspace(0.5, 5, lenM)       # Maturity
lamdaVector = np.linspace(0, 20, lenM)     # Intensity of Jump
meanVector = np.linspace(-0.1, 0.1, lenM)    # Mean Of Jump
volJumpVector = np.linspace(-0.1, 2, lenM)    # Volatility Of Jump

# PD according maturity

PDAccordingMaturity = list()
for singleMaturity in matVector:
    mm1 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(2, 0.05, 0.1, 0.2, 0.5, 0.3, singleMaturity)
    PDAccordingMaturity.append(mm1)
PDAccordingMaturity = pd.Series(PDAccordingMaturity)*100

# PD according Value/Debt Ratio

PDAccordingVD = list()
for singleRatio in ratioVector:
    mm2 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(singleRatio, 0.05, 0.1, 0.2, 0.5, 0.3, 1)
    PDAccordingVD.append(mm2)
PDAccordingVD = pd.Series(PDAccordingVD)*100

# PD according drift parameter

PDAccordingDrift = list()
for singleDrift in driftVector:
    mm3 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(2, singleDrift, 0.1, 0.2, 0.5, 0.3, 1)
    PDAccordingDrift.append(mm3)
PDAccordingDrift = pd.Series(PDAccordingDrift)*100

# PD according the volatility parameter

PDAccordingVolatility = list()
for singleVolatility in volVector:
    mm4 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(2, 0.05, 0.1, 0.2, 0.5, singleVolatility, 1)
    PDAccordingVolatility.append(mm4)
PDAccordingVolatility = pd.Series(PDAccordingVolatility)*100

PDAccordingLambda = list()
for singleLambda in lamdaVector:
    mm5 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(2, 0.05, 0.1, 0.2, singleLambda, 0.3, 1)
    PDAccordingLambda.append(mm5)
PDAccordingLambda = pd.Series(PDAccordingLambda)*100

PDAccordingMeanOfJump = list()
for singleMean in meanVector:
    mm6 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(2, 0.05, singleMean, 0.2, 0.5, 0.3, 1)
    PDAccordingMeanOfJump.append(mm6)
PDAccordingMeanOfJump = pd.Series(PDAccordingMeanOfJump)*100

PDAccordingVolatilityOfJump = list()
for singleVolatilityJump in volJumpVector:
    mm7 = jp.JumpDiffusionProbabilityOfDefaultWithValueDebtRatio(2, 0.05, 0.1, singleVolatilityJump, 0.5, 0.3, 1)
    PDAccordingVolatilityOfJump.append(mm7)
PDAccordingVolatilityOfJump = pd.Series(PDAccordingVolatilityOfJump)*100

# Start setting the subplots

plt.figure(figsize = (10, 10))

plt.subplots_adjust(hspace=0.335, wspace=0.305)

plt.subplot(3, 3, 1)
plt.scatter(x = matVector, y = PDAccordingMaturity, color = 'blue')
plt.xlabel('Maturity')
plt.ylabel('PD (%)')
plt.title('Figure 2.10', fontsize=10)

plt.subplot(3, 3, 2)
plt.scatter(x = ratioVector, y = PDAccordingVD, color = 'red')
plt.xlabel('Value/Debt Ratio')
plt.ylabel('PD (%)')
plt.title('Figure 2.11', fontsize=10)

plt.subplot(3, 3, 3)
plt.scatter(x = driftVector, y = PDAccordingDrift, color = 'black')
plt.xlabel('Drift')
plt.ylabel('PD (%)')
plt.title('Figure 2.12', fontsize=10)

plt.subplot(3, 3, 4)
plt.scatter(x = volVector, y = PDAccordingVolatility, color = 'green')
plt.xlabel('Volatility')
plt.ylabel('PD (%)')
plt.title('Figure 2.13', fontsize=10)

plt.subplot(3, 3, 5)
plt.scatter(x = lamdaVector, y = PDAccordingLambda, color = 'orange')
plt.xlabel('Lambda')
plt.ylabel('PD (%)')
plt.title('Figure 2.14', fontsize=10)

plt.subplot(3, 3, 6)
plt.scatter(x = meanVector, y = PDAccordingMeanOfJump, color = 'purple')
plt.xlabel('Mean of Jump Component')
plt.ylabel('PD (%)')
plt.title('Figure 2.15', fontsize=10)

plt.subplot(3, 3, 7)
plt.scatter(x = volJumpVector, y = PDAccordingMeanOfJump, color = 'teal')
plt.xlabel('Volatility of Jump Component')
plt.ylabel('PD (%)')
plt.title('Figure 2.16', fontsize=10)

#plt.show()

plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\Jump-Diffusion Sensitivity",
            dpi = 1500)
