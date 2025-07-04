from __future__ import division
import math
from collections import Counter

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

# k-means
# reduce k until we find a winner
def majority_vote(labels):
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])

def knn_classify(k, labeled_points, new_point):
    #order by distance
    by_distance = sorted(labeled_points, key = lambda pair: distance(pair[0], new_point))
    k_nearest_labels = [label for _, label in by_distance[:k]]
    
    return majority_vote(k_nearest_labels)


