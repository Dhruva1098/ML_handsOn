from __future__ import division
import math
from collections import defaultdict
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

# naive bayes
def tokenize(message):
    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    return set(all_words)

def count_words(training_set):
    counts = defaultdict(lambda:[0,0])
    for message, is_spam in training_set:
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):
    """ return w, p(w | span) and p(w | ~spam)"""
    return [(w,
             (spam+k)/(total_spams + 2*k),
            (non_spam + k) / (total_non_spams + 2*k))
            for w, (spam,non_spam) in counts.iteritems()]

def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0
    # iterate through each word in our vocab
    for word, prob_if_spam, prob_if_not_spam in word_probs:
        #if it appears in message add the log prpbab of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # if word doesnt appear in message
        # add log prob if not seing it 
        # which is log(1-prob of seeing it)
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)


# just craete a clas vro
class NaiveBayesClassifier:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        #count spam and non spam messages
        num_spams = len([is_spam
                        for message, is_spam in training_set
                        if is_spam])
        num_non_spams = len(training_set) - num_spams

        #running training data through our pipeline
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts, num_spams, num_non_spams, self.k)

    def classify(self, message):
        return spam_probability(self.word_probs, message)



