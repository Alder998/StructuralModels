# Let's perform a sensitivity analysis of our Down-and-Out Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import DownAndOutCall as do

# PROBABILITY OF DEFAULT ACCORDING THE LEVEL OF DEBT AND THE MATURITY

lenM = 100
debtVector = np.linspace(10e-20, 1000000, lenM)
matVector = np.linspace(1, 20, lenM)

final = list()

for maturityLevel in matVector:
    accRatio = list()
    for debtLevel in debtVector:
        jnk = do.downAndOutProbabilityOfDefault(1000000, debtLevel, 0.02, 0.3, maturityLevel)
        accRatio.append(jnk)
    final.append(accRatio)

finalS = list()
for series in final:
    finalS.append(pd.Series(series))

finalS = pd.concat([series for series in finalS], axis = 1)

z = list()
for value in debtVector:
    zp = np.full(shape = len(debtVector), fill_value=value)
    z.append(zp)

# Final Plot Setup

plt.figure(figsize = (12, 12))
ax = plt.axes(projection = '3d')
for DSeries in finalS:
    # Scatter 3d
    #ax.scatter3D(matVector, z[DSeries], finalS[DSeries], color = 'blue')
    # Area 3D
    ax.plot_surface(matVector, z, finalS, cmap = 'viridis', edgecolor = 'none')

plt.xlabel('Maturity')
plt.ylabel('Value-Debt Ratio')
ax.set_zlabel('Merton Probability of Default')
plt.title('Probability of Default according Ratio Value-Debt and Maturity (Volatility fixed at 0.3, drift fixed at 0.02)')
ax.view_init(19, 45)
plt.show()

#plt.savefig(r'C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Merton Model\Sensitivity Analysis\Probability of Default according Ratio Value-Debt and Maturity', dpi = 700)

