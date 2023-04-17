# We are gonna estimate A and sigmaAssets with th claculation-intensive procedure by KMV
import numpy as np

import pandas as pd
import MertonModel as mm
from sympy import Symbol
from sympy.solvers import nsolve
from sympy import sin, tan
from sympy import exp, sqrt, ln
from sympy.stats import Normal
from sympy.stats import cdf
from sympy import S
from sympy import simplify
import matplotlib.pyplot as plt

r = 0.05
sigmaA = 0.072538
T = 1

levelOfDebt = mm.getDebtValue('HTZ', TimeSeries=True)
levelOfEquity = mm.getEquityValue('HTZ', TimeSeries=True)

#startValue = 100000
#obs = 100
#f = list()
#for i in range(obs):
#    start = startValue
#    inc = np.random.normal(loc=0, scale=0.5, size=1)
#    w = (start + inc).cumsum()
#    f.append(w.reshape(1, 1))
#a = np.array(f)
#levelOfDebt = (pd.DataFrame(a.reshape(obs, 1)).T).transpose()

timeSeriesA = list()
for time in range(len(levelOfDebt)):

    print(levelOfDebt[time])

    A = Normal("A", 0, S(1))
    d = (simplify(cdf( (ln(A/levelOfDebt[time]) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T)) )(0)))
    d1 = (simplify(cdf( ((ln(A/levelOfDebt[time]) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T))) - (sigmaA*sqrt(T)) )(0)))

    finalEq = nsolve(A*((cdf( (ln(A/levelOfDebt[time]) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T)) )(0)))
                      - levelOfDebt[time] * exp(-r*T) * ((cdf( ((ln(A/levelOfDebt[time]) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T)))
                                                               - (sigmaA*sqrt(T)) )(0))) - levelOfEquity[time], A, (1.2,))

    print(finalEq)
    timeSeriesA.append(finalEq)

finalSTD = (((pd.Series(timeSeriesA).pct_change())))
finalSTD = finalSTD[1:len(finalSTD)].std()


