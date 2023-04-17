# Let's perform a sensitivity analysis of our Merton Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MertonModel as mm
import JumpDiffusionModelClosedForm as jp

# PROBABILITY OF DEFAULT ACCORDING THE RATIO VALUE-DEBT AND THE MATURITY

lenM = 100
ratioVector = np.linspace(1,3, lenM)
matVector = np.linspace(1, 10, lenM)

final = list()

for maturityLevel in matVector:
    accRatio = list()
    for ratioLevel in ratioVector:
        jnk = jp.JumpDiffusionProbabilityOfDefault(ratioLevel, 0.02, 0.005, 0.4, 1, 0.3, maturityLevel)
        accRatio.append(jnk)
    final.append(accRatio)

finalS = list()
for series in final:
    finalS.append(pd.Series(series))

finalS = pd.concat([series for series in finalS], axis = 1)

z = list()
for value in ratioVector:
    zp = np.full(shape = len(ratioVector), fill_value=value)
    z.append(zp)

# Final Plot Setup

#plt.figure(figsize = (12, 12))
#ax = plt.axes(projection = '3d')
#for DSeries in finalS:
    # Scatter 3d
    #ax.scatter3D(driftVector, z[DSeries], finalS[dSeries], color = 'blue')
    # Area 3D
    #ax.plot_surface(matVector, z, finalS, cmap = 'viridis', edgecolor = 'none')

#plt.xlabel('Maturity')
#plt.ylabel('Value-Debt Ratio')
#ax.set_zlabel('Merton Probability of Default')
#plt.title('Probability of Default according Ratio Value-Debt and Maturity (Volatility fixed at 0.3, drift fixed at 0.02)')
#ax.view_init(19, 45)

#plt.show()

# PROBABILITY OF DEFAULT ACCORDING THE DRIFT AND THE VOLATILITY PARAMETER

lenM = 100
driftVector = np.linspace(0.001,0.2, lenM)
volVector = np.linspace(0.001, 1, lenM)

final = list()

for driftLevel in driftVector:
    accDrift = list()
    for volLevel in volVector:
        jnk1 = jp.JumpDiffusionProbabilityOfDefault(1.5, 0.02, driftLevel, volLevel, 1, 0.3, 1)
        accDrift.append(jnk1)
    final.append(accDrift)

finalS = list()
for series in final:
    finalS.append(pd.Series(series))

finalS = pd.concat([series for series in finalS], axis = 1)

z = list()
for value in volVector:
    zp = np.full(shape = len(ratioVector), fill_value=value)
    z.append(zp)

# Final Plot Setup

plt.figure(figsize = (12, 12))
ax = plt.axes(projection = '3d')
for DSeries in finalS:
    # Scatter 3d
    # ax.scatter3D(driftVector, z[DSeries], finalS[dSeries], color = 'blue')
    # Area 3D
    ax.plot_surface(driftVector, z, finalS, cmap = 'viridis', edgecolor = 'none')

plt.xlabel('Drift Level')
plt.ylabel('Volatility Level')
ax.set_zlabel('Merton Probability of Default')
plt.title('Probability of Default according Drift and Volatility Parameter (ratio fixed at 2, maturity fixed at 1Y)')
ax.view_init(45, -66)
plt.show()

#plt.savefig(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Sensitivity Analysis\Probability of Default according Drift and Volatility Parameter', dpi = 700)
