from math import sqrt, exp, log, pi
from scipy.stats import norm #normal distribution function
import numpy as np
import matplotlib.pyplot as plt

# Function for calculating d1 and d2 for option calculation in Black Scholes model
def d(sigma, S, K, r, t):
    d1 = 1 / (sigma * sqrt(t)) * ( log(S/K) + (r + sigma**2/2) * t)
    d2 = d1 - sigma * sqrt(t)
    return d1, d2

#Enter function for the call price

def callprice(sigma, S, K, r, t, d1, d2):
    C = norm.cdf(d1) * S - norm.cdf(d2) * K * exp(-r * t)
    return C

S = 100.0
K = 105.0
r = 0.01
t = 30.0/365
C0 = 2.30

#  We need a starting guess for the implied volatility.  We chose 0.5
#  arbitrarily.
vol = 0.5

epsilon = 1.0          #  Define variable to check stopping conditions
tol = 1e-4             #  Stop calculation when abs(epsilon) < this number

i = 0                  #  Variable to count number of iterations
max_iter = 1e3         #  Max number of iterations before aborting

##debug
sigma = np.linspace(0,1)
d1,d2 = d(sigma, S, K, r, t)
C = callprice(sigma, S, K, r, t, d1, d2)

plt.plot(sigma, C - C0)
plt.show()

while epsilon > tol:
    i = i + 1
    if i > max_iter:
        print ('Program failed to find a root.  Exiting.')
        break;

    orig = vol
    d1, d2 = d(vol, S, K, r, t)
    function_value = callprice(vol, S, K, r, t, d1, d2) - C0
    vega = S * norm.pdf(d1) * sqrt(t)
    vol = -function_value/vega + vol
    epsilon = abs(function_value)

print ('Implied volatility = ',  vol)
print ('Code required', i, 'iterations.')
