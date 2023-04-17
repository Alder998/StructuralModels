# Let's perform a sensitivity analysis of our Merton Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MertonModel as mm

# PROBABILITY OF DEFAULT ACCORDING THE DRIFT AND THE VOLATILITY PARAMETER

lenM = 100
ratioVector = np.linspace(1,3, lenM)
driftVector = np.linspace(-0.5,0.5, lenM)
volVector = np.linspace(0.001, 3, lenM)

final = list()

for driftLevel in driftVector:
    accDrift = list()
    for volLevel in volVector:
        jnk1 = mm.MertonProbabilityOfDefaultWithValueToDebt(2, driftLevel, volLevel, 1)
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

fig = plt.figure(figsize = (12, 12))

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
ax.view_init(25, -30)
#plt.show()

plt.savefig(r"C:\Users\39328\OneDrive\Desktop\Davide\UNIMI Statale\Tesi di merda\Immagini & Grafici\3D Mean-Variance", dpi = 1500)
