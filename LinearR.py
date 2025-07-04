
import math
from collections import Counter
from __future__ import division

#vectors 

def vector_add(v,w):
    return [v_i + w_i
            for v_i, w_i in zip(v,w)]

def vector_sub(v,w):
    return [v_i - w_i
            for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c,vector):
    return [c*v_i for v_i in vector]

def vector_mean(vector):
    n = len(vector)
    return scalar_multiply(1/n, vector_sum(vector))

def dot(v,w):
    return [v_i * w_i for v_i,w_i in zip(v,w)]

def sum_of_squares(v):
    return dot(v,v)

def magnitude(v):
    return math.sqrt(sum_of_squares(v))

def squared_distance(v,w):
    return sum_of_squares(vector_sub(v,w))

def distance(v,w):
    return math.sqrt(sum_of_squares(vector_sub(v,w)))

def mean(x):
    return sum(x)/len(x)

def median(v):
    n = len(v)
    sorted_v = sorted(v)
    mid = n // 2
    
    if n % 2 == 1:
        return sorted(mid)
    else 
        return (sorted[mid] + sorted[mid-1])/2

def quantile(x, p):
    p_index = int(p * len(x))
    return sorted(x)[p_index]

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems()
    if count == max_count]

# Dispersion
def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n-1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

# correlation

def covariance(x,y):
    n = len(x)
    return dot(de_mean(x), de_mean(y))/ (n-1)

    
def correlation(x,y):
    standard_x = standard_deviation(x)
    standard_y = standard_deviation(y)
    if standard_x > 0 && standard_y > 0:
        return covariance(x,y) / standard_x / standard_y
    else:
        return 0
# Linear regression
def predict(alpha, beta, x_i):
    return beta * x_i + alpha

def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)

# use squared errors because -ve and +ve errors may cancel out
def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x,y))

def least_squares_fit(x, y):
    beta = correlation(x,y )* standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta

def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))

def r_squared(alpha, beta, x, y):
    """ fraction of variation in y captured by the model 
    = 1 - fraction of variation in y not captured by the model """
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) / total_sum_of_squares(y))


# if we write theta = [alpha, beta] we can also use gradient descent to solve this problem

def squared_error(x_i, y_i, theta):
    alpha, beta = theta
    return error(alpha, beta, x_i, y_i) ** 2

def squared_error_gradient(x_i, y_i, theta):
    alpha,  beta = theta
    return [-2* error(alpha, beta, x_i, y_i),
            -2 * error(alpha, beta, x_i, y_i) * x_i] #partial derivatives because two variables


