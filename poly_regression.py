
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



def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))

# poly regression
def predict(x_i, beta):
    return dot(x_i, beta)

# error to minimize
def error(x_i, y_i, beta):
    return y_i - predict(x_i, beta)

def squared_error(x_i, y_i, beta):
    return error9x_i, y_i, beta) ** 2

# gradient
def squared_error_gradient(x_i, y_i, beta):
    return [-2 * x_ij * error(x_i, y_i, beta)
            for x_ij in x_i]


def step(v, direction, step_size):
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v,direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f

# stochastic GD
def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x,y, theta_0, alpha_0=0.01):
    data = zip(x, y)
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float('inf')
    iterations_w_no_improvement = 0
    while iterations_w_no_improvement < 0:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data)
        if value < min_value:
            min_theta, min_value = theta, value
            iterations_w_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_w_no_improvement += 1
            alpha *= 0.9 #shrink step size

        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta
       
# estimate beta
def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
return minimize_stochastic(squared_error,
                           squared_error_gradient,
                           x, y,
                           beta_initial, 0.001)

# check goodness of fit
def multiple_r_squared(x, y , beta):
    sum_of_squared_errors = sum(error(x_i, y_i, beta) ** 2
                                for x_i, y_i in zip(x, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)


def bootstrap_sample(data):
    """ randomly samples len(Data) elements with replacement"""
    return [random.choice(data) for _ in data]

def bootstrap_statistic(data, stats_fn, num_samples):
    return [stats_fn(bootstrap_samples(data))
            for _ in range(num_samples)]

# standard errors of regression coefficients
def estimate_sample_beta(sample):
    """sample is a list of pairs(x_i, y_i)"""
    x_sample, y_sample = zip(*sample)
    return estimate_beta(x_sample, y_sample)

def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        # if coefficient is +ve we need to compute twice the probability of seeing an even larger value
        return 2 * ( 1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)

# regularization
def ridge_penalty(beta, alpha):
    return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x_i, y_i, beta, alpha):
    return error(x_i, y_i, beta) ** 2 + ridge_penalty(beta, alpha)

def ridge_penalty_gradient(beta, alpha):
    return [0] + [2 * alpha * beta_j for beta_j in beta[1:]]

def squared_error_ridge_gradient(x_i, y_i, beta, alpha):
    return vector_add(squared_error_gradient(x_i, y_i, beta),
                      ridge_penalty_gradient(beta, alpha))

def estimate_beta_ridge(x, y, alpha):
    """ use GD to fit a ridge regression with penalty alpha"""
    beta_initial = [random.random() for x_i in x[0]]
return minimize_stochastic(partial(squared_error_ridge, alpha = alpha),
                           partial(squared_error_ridge_gradient,
                                   alpha=alpha),
                           x, y,
                           beta_initial,
                           0.001)
# lasso
def laso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])
