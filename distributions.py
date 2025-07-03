from __future__ import division
import math
from matplotlib import pyplot as plt
def uniform_pdf(x):
    return 1 if x>=0 and x < 1 else 0

def uniform_cdf(x):
    if x < 0 : return 0
    elif x < 1 : return x
    else : return 1 

def normal_pdf(x, mu = 0, sigma = 1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

def normal_cdf(x, mu=0, sigma=1):
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma)) /2

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) # bottom right
plt.title("Various Normal cdfs")
plt.show()

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance =0.00001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p,tolerance=tolerance)

    low_z, low_p = -10.0, 0
    high_z, high_p = 10.0, 1
    while high_z - low_z > tolerance :
        mid_z = (low_z + high_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            high_z, high_p = mid_z, mid_P
        else : break
    return mid_z

def normal_approximation_to_binomeal(n,p):
    mu = p * n 
    sigma = math.sqrt(p * (1 - p) * n)
    return my, sigma

normal_probability_below = normal_cdf
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo,mu,sigma)

def normal_probability_bw(lo,hi,mu=0,sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo,hi,mu=0,sigma=1):
    return 1 - normal_probability_bw(lo,hi,mu,sigma)
