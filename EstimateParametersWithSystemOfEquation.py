# Here we try to solve the for A and sigmaA using a system of equations

import numpy as np

import pandas as pd
import MertonModel as mm
from sympy import Symbol
from sympy.solvers import nsolve
from sympy import sin, tan
from sympy import exp, sqrt, ln
from sympy.stats import Normal
from sympy.stats import cdf
from sympy import S, Eq
from sympy import simplify
import matplotlib.pyplot as plt

r = 0.05
T = 1
sigmaEquity = 0.005

levelOfDebt = mm.getDebtValue('HTZ')
levelOfEquity = mm.getEquityValue('HTZ')

A = Normal("A", 0, S(1))
sigmaA = Normal("sigmaA", 0, S(1))

system = [Eq(A*((cdf( (ln(A/levelOfDebt) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T)) )(0))) - levelOfDebt * exp(-r*T) * ((cdf( ((ln(A/levelOfDebt) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T))) - (sigmaA*sqrt(T)) - levelOfEquity)(0)))),
             Eq( (A/levelOfEquity) * ((cdf( (ln(A/levelOfDebt) - (r - 0.5*sigmaA)*T)/ (sigmaA*sqrt(T))))) * sigmaA )]

finalEq = nsolve(system, [A, sigmaA], (1.2,))

print(finalEq)
