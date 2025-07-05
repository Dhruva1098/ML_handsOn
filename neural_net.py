
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

# Neural network
def step_function(x):
    return 1 if x >= 1 else 0

def perceptron_output(weights, bias, x):
    calc =  dot(weights, x) + bias
    return step_function(calc)

# feed forward
def sigmoid(x):
    return 1/(1 + math.exp(-x))

def neuron_output(weights, x): # no bias because included in weight vector
    return sigmoid(dot(weights, x))

def feed_forward(neural_network, input_vector):
    """ neural network is nothing but lists of lists of weights"""
    outputs = []

    #process one layer at a time
    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)
        # input to next layer is output of this one
        input_vector = output
    return outputs

#backpropagation
def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # output * (1-output) is derivative from sigmoid
    output_deltas = [output * (1 - output) * (output - target)
                     for output, target in zip(outputs, targets)]

    #adjust weights for op layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        #focus in ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            #adjust jth weight based on both
            #this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                    dot(output_deltas, [n[i] for n in output_layer])
                    for i, hidden_output in enumerate(hidden_outputs)]
    #adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input


        
    

