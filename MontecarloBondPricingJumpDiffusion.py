
import numpy as np
import matplotlib.pyplot as plt

T = 1
steps = 350

numberOfSimulation = 10000

# Inizialize the Parameters

var = 0.01225
var_pi = 0.015**2
mean_pi = 0.00
lam = 0.00005
rFreeRate = 0.005
v = 0.01

# MONTECARLO

WjList = list()
for i in range(1, numberOfSimulation + 1):

    mmm = (rFreeRate - var - lam*v) * (T/steps)
    xi = np.random.normal((rFreeRate - var - lam*v) * (T/steps), var*(T/steps), steps)
    pGrecoi = np.random.normal(mean_pi, var_pi, steps)

    prob = lam*(T/steps)
    yi = np.random.binomial(1, prob, steps)

    X0 = 10**(-123)
    X = np.zeros(steps)
    X[0] = X0

    # Creiamo la dinamica
    for timeIndex in range(1, steps):
        X[timeIndex] = (X[timeIndex - 1]) * np.exp(xi[timeIndex] + yi[timeIndex] * pGrecoi[timeIndex])


    w = np.random.uniform(0, 0.6)
    Wj = np.where(np.log(X) <= X0, X*w, 0)

    WjList.append(Wj)

plt.plot(X)
plt.show()

print(WjList)

notional = 100
WjList = np.array(WjList)
bondPrice = notional*(np.exp(-rFreeRate*T) * (1 - ((WjList.sum()/numberOfSimulation))))

print('\n')
print('The Bond Price under the Jump Diffusion Model is:', bondPrice)

# Estract the probability of Default

p = (1-(bondPrice/notional)) / (1-0.6)   # Recovery supposed to be 0.4

print('Implied (%) Probability of Default at 1Y:', p*100, '%')
