import math
from __future__ import division
from collections import defaultdict
def entropy(clss_probabilities):
    return sum(-p * math.log(p,2)
               for p in class_probabilities
                if p)

def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]

def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)

def partition_entropy(subsets):
    total_count = sum(len(subset) for subset in subsets)
    return sum( data_entropy(subset) * len(subset) / total_count
               for subset in subsets)

# create decision tree
def partition_by(inputs, attribute):
    groups = defaultdict(list)
    for input in inputs:
        key = input[0][attribute] # value of specified attribute
        groups[key].append(input) # and then add this input to correct list
    return groups

def partition_entropy_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


def classify(tree, input):
    """ classify the input using the given decision tree"""
    # return value if leaf node
    if tree in [True, False]
        return tree

    #otherwise tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values of are subtrees to consider next
    attribute, subtree_dict = tree
    subtree_key = input.get(attribute) # none if input is missing attribute
    
    if subtree_key not in subtree_dict: # if no subtree for key, we use the none subtree
        subtree_key = None
        
        subtree = subtree_dict[subtree_key] #choose appropriate subtree and use it to classigy
        return classify(subtree, input)

# build tree representation from our training data
def build_tree_id3(imputs, split_candidates=None):
    # for first pass all keys of first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    #count true and false in inputs
    num_inputs = len(inputs)
    num_trues = len([label for item, label in inputs if label])
    num_falses = num_inputs - num_trues

    if num_trues == 0: return False # no trues means false leaf
    if num_falses == 0: return True
    
    if not split_candidates:
        return num_trues >= num_falses #if no split candidates are left we retirn majority leaf
    
    #otherwise we split on best attribute
    best_attribute = min(split_candidates,
                         key=partial(partition_entropy_by, inputs))

    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates
                        if a != best_attribute]
    #now just recursively build the subtrees
    subtrees = { sttribute_value : build_tree_id3(subset, new_candidates)
                for attribute_value, subset in partitions.iteritems()}

    subtrees[None] = num_trues > num_falses #default
    return (best_attribute, subtrees)

# random forst
def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]
