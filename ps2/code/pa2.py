import os
import sys
import numpy as np
from scipy.misc import logsumexp
from collections import Counter
import random
import itertools

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior
alpha = 0.1
# alpha = 0

def renormalize(cnt):
  '''
  renormalize a Counter()
  '''
  tot = 1. * sum(cnt.values())
  for a_i in cnt:
    cnt[a_i] /= tot
  return cnt

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------
class NBCPT(object):
  '''
  NB Conditional Probability Table (CPT) for a child attribute.  Each child
  has only the class variable as a parent.
  '''

  def __init__(self, A_i):
    '''
    TODO: create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT

    Params:
      - A_i: the index of the child variable

    '''

    self.index =  A_i
    self.P_Ai_given_C = {}
    for i in range(2):
        self.P_Ai_given_C[i] = {}
    # raise NotImplementedError()

  def learn(self, A, C):
    '''
    TODO: populate any instance variables specified in __init__ we need
    to learn the parameters for this CPT

    Params:
     - A: a (n,k) numpy array where each row is a sample of assignments
     - C: a (n,) numpy array where the elements correspond to the
       class labels of the rows in A
    Return:
     - None
     '''

    n, k = A.shape
    c_1_n = 0.0
    c_1_d = 0.0
    c_0_n = 0.0
    c_0_d = 0.0

    for i in range(n):
      if C[i] == 1:
        c_1_n += 1.0 * A[i][self.index]
        c_1_d += 1.0
      else:
        c_0_n += 1.0 * A[i][self.index]
        c_0_d += 1.0

    self.P_Ai_given_C[0][0] = 1 - (c_0_n + alpha) / (c_0_d + 2 * alpha)
    self.P_Ai_given_C[0][1] = (c_0_n + alpha) / (c_0_d + 2 * alpha)
    self.P_Ai_given_C[1][0] = 1 - (c_1_n  + alpha) / (c_1_d + 2 * alpha)
    self.P_Ai_given_C[1][1] = (c_1_n + alpha) / (c_1_d + 2 * alpha)

    # raise NotImplementedError()

  def get_cond_prob(self, entry, c):
    '''
    TODO: return the conditional probability P(A_i | C) for the values
    specified in the example entry and class label c

    Params:
     - entry: full assignment of variables
        e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
     - c: the class
    Returns:
     - p: a scalar, the conditional probability P(A_i | C)

    '''

    return self.P_Ai_given_C[c][entry[self.index]]
    # raise NotImplementedError()

class NBClassifier(object):
  '''
  NB classifier class specification.
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO: create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to self._train
    Suggestions for the attributes in the classifier:
        - self.P_c: a dictionary for the probabilities for the class variable C
        - self.cpts: a list of NBCPT objects
    '''

    n, k = A_train.shape

    self.P_c = {}
    self.cpts = {}

    for i in range(k):
        self.cpts[i] = NBCPT(i)
    self._train(A_train, C_train)

    # raise NotImplementedError()

  def _train(self, A_train, C_train):
    '''
    TODO: train your NB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
    Params:
      - A_train: a (n,k) numpy array where each row is a sample of assignments
      - C_train: a (n,)  numpy array where the elements correspond to
        the class labels of the rows in A
    Returns:
     - None

    '''

    n,k = A_train.shape

    self.P_c[1] = 1.0 * (sum(C_train) + alpha) / (len(C_train) + 2 * alpha)
    self.P_c[0] = 1.0 - self.P_c[1]

    for i in range(k):
        self.cpts[i].learn(A_train, C_train)

    # raise NotImplementedError()

  def classify(self, entry):
    '''
    TODO: return the log probabilites for class == 0 or class == 1 as a
    tuple for the given entry

    Params:
      - entry: full assignment of variables
        e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
    Returns:
     - c_pred: the predicted label, one of {0, 1}
     - logP_c_pred: the log of the conditional probability of the label |c_pred|


    '''

    k = entry.shape[0]

    # print(list(entry))
    m1_count = sum([1 if i == -1 else 0 for i in entry])
    all_entries = []

    for i in itertools.product([0, 1], repeat=m1_count):
      c = 0
      # print(i)
      new_entry = []
      for j in range(k):
        if entry[j] == 0 or entry[j] == 1:
          new_entry.append(entry[j])
        else:
          new_entry.append(i[c])
          c += 1
      all_entries.append(new_entry)
      # print(new_entry)
    if all_entries == []:
      all_entries.append(entry)

    p_a_c1 = [1.0 for _ in range(len(all_entries))]
    p_a_c0 = [1.0 for _ in range(len(all_entries))]

    for i in range(len(all_entries)):
      for j in range(k):
        p_a_c1[i] *= self.cpts[j].get_cond_prob(all_entries[i], 1)
        p_a_c0[i] *= self.cpts[j].get_cond_prob(all_entries[i], 0)

    num = 0.0
    den = 0.0

    for i in range(len(p_a_c1)):
      num += p_a_c1[i] * self.P_c[1]
      den += (p_a_c1[i] * self.P_c[1] + p_a_c0[i] * self.P_c[0])

    c_pred = num / den
    clas = 0 if c_pred < 0.5 else 1
    log_c_pred = np.math.log(c_pred if clas == 1 else 1 - c_pred)
    return (clas, log_c_pred)

    # p_a_c1 = 1.0
    # p_a_c0 = 1.0
    #
    # for i in range(k):
    #     p_a_c1 *= self.cpts[i].get_cond_prob(entry, 1)
    #     p_a_c0 *= self.cpts[i].get_cond_prob(entry, 0)
    #
    # c_pred = p_a_c1 * self.P_c[1] / (p_a_c1 * self.P_c[1] + p_a_c0 * self.P_c[0])
    # clas = 0 if c_pred < 0.5 else 1
    # c_pred_class = c_pred if clas == 1 else 1 - c_pred
    # log_c_pred = np.math.log(c_pred_class)
    #
    # # raise NotImplementedError()
    # return (clas, log_c_pred)

  def unobserved_probability(self, entry, index):
    k = entry.shape[0]

    # print(list(entry))
    m1_count = sum([1 if i == -1 else 0 for i in entry])
    all_entries = []

    for i in itertools.product([0, 1], repeat=m1_count):
      c = 0
      # print(i)
      new_entry = []
      for j in range(k):
        if entry[j] == 0 or entry[j] == 1:
          new_entry.append(entry[j])
        else:
          new_entry.append(i[c])
          c += 1
      all_entries.append(new_entry)
      # print(new_entry)
    if all_entries == []:
      all_entries.append(entry)

    p_a_c1 = [1.0 for _ in range(len(all_entries))]
    p_a_c0 = [1.0 for _ in range(len(all_entries))]

    for i in range(len(all_entries)):
      for j in range(k):
        p_a_c1[i] *= self.cpts[j].get_cond_prob(all_entries[i], 1)
        p_a_c0[i] *= self.cpts[j].get_cond_prob(all_entries[i], 0)

    num = 0.0
    den = 0.0

    for i in range(len(p_a_c1)):
      if all_entries[i][index] == 1:
        num += (p_a_c1[i] * self.P_c[1] + p_a_c0[i] * self.P_c[0])
      den += (p_a_c1[i] * self.P_c[1] + p_a_c0[i] * self.P_c[0])

    c_pred = num / den
    clas = 0 if c_pred < 0.5 else 1
    log_c_pred = np.math.log(c_pred)
    return (clas, log_c_pred)


#--------------------------------------------------------------------------
# TANB CPT and classifier
#--------------------------------------------------------------------------
class TANBCPT(object):
  '''
  TANB CPT for a child attribute.  Each child can have one other attribute
  parent (or none in the case of the root), and the class variable as a
  parent.
  '''

  def __init__(self, A_i, A_p):
    '''
    TODO: create any persistent instance variables you need that hold the
    state of the learned parameters for this CPT

    Params:
     - A_i: the index of the child variable
     - A_p: the index of its parent variable (in the Chow-Liu algorithm,
       the learned structure will have up to a single parent for each child)

    '''
    self.index = A_i
    self.parent_index = A_p
    self.P_Ai_given_C_p = {}
    for i in range(2):
      self.P_Ai_given_C_p[i] = {}
      for j in range(2):
        self.P_Ai_given_C_p[i][j] = {}
    # raise NotImplementedError()

  def learn(self, A, C):
    '''
    TODO: populate any instance variables specified in __init__ we need to learn
    the parameters for this CPT

    Params:
     - A: a (n,k) numpy array where each row is a sample of assignments
     - C: a (n,)  numpy array where the elements correspond to the class
       labels of the rows in A
    Returns:
     - None

    '''
    n, k = A.shape

    c_1_1_n = 0.0
    c_1_1_d = 0.0
    c_1_0_n = 0.0
    c_1_0_d = 0.0
    c_0_1_n = 0.0
    c_0_1_d = 0.0
    c_0_0_n = 0.0
    c_0_0_d = 0.0

    for i in range(n):
      if C[i] == 1:
        if A[i][self.parent_index] == 1:
          c_1_1_n += 1.0 * A[i][self.index]
          c_1_1_d += 1.0
        else:
          c_1_0_n += 1.0 * A[i][self.index]
          c_1_0_d += 1.0
      else:
        if A[i][self.parent_index] == 1:
          c_0_1_n += 1.0 * A[i][self.index]
          c_0_1_d += 1.0
        else:
          c_0_0_n += 1.0 * A[i][self.index]
          c_0_0_d += 1.0

    self.P_Ai_given_C_p[0][0][0] = 1 - (c_0_0_n + alpha) / (c_0_0_d + 2 * alpha)
    self.P_Ai_given_C_p[0][0][1] = (c_0_0_n + alpha) / (c_0_0_d + 2 * alpha)
    self.P_Ai_given_C_p[0][1][0] = 1 - (c_0_1_n + alpha) / (c_0_1_d + 2 * alpha)
    self.P_Ai_given_C_p[0][1][1] = (c_0_1_n + alpha) / (c_0_1_d + 2 * alpha)
    self.P_Ai_given_C_p[1][0][0] = 1 - (c_1_0_n + alpha) / (c_1_0_d + 2 * alpha)
    self.P_Ai_given_C_p[1][0][1] = (c_1_0_n + alpha) / (c_1_0_d + 2 * alpha)
    self.P_Ai_given_C_p[1][1][0] = 1 - (c_1_1_n + alpha) / (c_1_1_d + 2 * alpha)
    self.P_Ai_given_C_p[1][1][1] = (c_1_1_n + alpha) / (c_1_1_d + 2 * alpha)

    # raise NotImplementedError()

  def get_cond_prob(self, entry, c):
    '''
    TODO: return the conditional probability P(A_i | Pa(A_i)) for the values
    specified in the example entry and class label c
    Note: in the expression above, the class C is also a parent of A_i!

    Params;
        - entry: full assignment of variables
          e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
        - c: the class
    Returns:
     - p: a scalar, the conditional probability P(A_i | Pa(A_i))

    '''
    # raise NotImplementedError()
    # p = None
    p = self.P_Ai_given_C_p[c][entry[self.parent_index]][entry[self.index]]
    return p



class TANBClassifier(NBClassifier):
  '''
  TANB classifier class specification
  '''

  def __init__(self, A_train, C_train):
    '''
    TODO: create any persistent instance variables you need that hold the
    state of the trained classifier and populate them with a call to self._train

    Params:
      - A_train: a (n,k) numpy array where each row is a sample of assignments
      - C_train: a (n,)  numpy array where the elements correspond to
        the class labels of the rows in A

    '''
    # super.__init__(A_train, C_train)
    n, k = A_train.shape

    self.mst = get_mst(A_train, C_train)
    self.root = get_tree_root(self.mst)
    self.parent = {}

    for (x, y) in get_tree_edges(self.mst, self.root):
      self.parent[y] = x

    self.P_c = {}
    self.cpts = {}

    for i in range(k):
      if i == self.root:
        self.cpts[i] = NBCPT(i)
      else:
        self.cpts[i] = TANBCPT(i, self.parent[i])
    self._train(A_train, C_train)

    # raise NotImplementedError()


  def _train(self, A_train, C_train):
    '''
    TODO: train your TANB classifier with the specified data and class labels
    hint: learn the parameters for the required CPTs
    hint: you will want to look through and call functions imported from tree.py:
        - get_mst(): build the mst from input data
        - get_tree_root(): get the root of a given mst
        - get_tree_edges(): iterate over all edges in the rooted tree.
          each edge (a,b) => a -> b

    Params:
      - A_train: a (n,k) numpy array where each row is a sample of assignments
      - C_train: a (n,)  numpy array where the elements correspond to
        the class labels of the rows in A
    Returns:
     - None

    '''
    # super._train(A_train, C_train)
    n, k = A_train.shape

    self.P_c[1] = 1.0 * (sum(C_train) + alpha) / (len(C_train) + 2 * alpha)
    self.P_c[0] = 1.0 - self.P_c[1]

    for i in range(k):
      self.cpts[i].learn(A_train, C_train)
    # raise NotImplementedError()

  def classify(self, entry):
    '''
    TODO: return the log probabilites for class == 0 and class == 1 as a
    tuple for the given entry

    Params:
     - entry: full assignment of variables
        e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
    Returns:
     - c_pred: the predicted label in {0, 1}
     - logP_c_pred: the log conditional probability of predicting the label |c_pred|

    NOTE: this class inherits from NBClassifier, and optionally, it is possible to
    write this method in NBClassifier, such that this implementation can
    be removed.

    '''
    # return super.classify(entry)
    k = entry.shape[0]

    # print(list(entry))
    m1_count = sum([1 if i == -1 else 0 for i in entry])
    all_entries = []

    for i in itertools.product([0, 1], repeat=m1_count):
      c = 0
      # print(i)
      new_entry = []
      for j in range(k):
        if entry[j] == 0 or entry[j] == 1:
          new_entry.append(entry[j])
        else:
          new_entry.append(i[c])
          c += 1
      all_entries.append(new_entry)
      # print(new_entry)
    if all_entries == []:
      all_entries.append(entry)

    p_a_c1 = [1.0 for _ in range(len(all_entries))]
    p_a_c0 = [1.0 for _ in range(len(all_entries))]

    for i in range(len(all_entries)):
      for j in range(k):
        p_a_c1[i] *= self.cpts[j].get_cond_prob(all_entries[i], 1)
        p_a_c0[i] *= self.cpts[j].get_cond_prob(all_entries[i], 0)

    num = 0.0
    den = 0.0

    for i in range(len(p_a_c1)):
      num += p_a_c1[i] * self.P_c[1]
      den += (p_a_c1[i] * self.P_c[1] + p_a_c0[i] * self.P_c[0])

    c_pred = num / den
    clas = 0 if c_pred < 0.5 else 1
    log_c_pred = np.math.log(c_pred if clas == 1 else 1 - c_pred)
    return (clas, log_c_pred)

    # k = entry.shape[0]
    #
    # p_a_c1 = 1.0
    # p_a_c0 = 1.0
    #
    # for i in range(k):
    #   p_a_c1 *= self.cpts[i].get_cond_prob(entry, 1)
    #   p_a_c0 *= self.cpts[i].get_cond_prob(entry, 0)
    #
    # c_pred = p_a_c1 * self.P_c[1] / (p_a_c1 * self.P_c[1] + p_a_c0 * self.P_c[0])
    # clas = 0 if c_pred < 0.5 else 1
    # log_c_pred = np.math.log(c_pred if clas == 1 else 1 - c_pred)


    return (clas, log_c_pred)
    # raise NotImplementedError()

  def unobserved_probability(self, entry, index):
    k = entry.shape[0]

    # print(list(entry))
    m1_count = sum([1 if i == -1 else 0 for i in entry])
    all_entries = []

    for i in itertools.product([0, 1], repeat=m1_count):
      c = 0
      # print(i)
      new_entry = []
      for j in range(k):
        if entry[j] == 0 or entry[j] == 1:
          new_entry.append(entry[j])
        else:
          new_entry.append(i[c])
          c += 1
      all_entries.append(new_entry)
      # print(new_entry)
    if all_entries == []:
      all_entries.append(entry)

    p_a_c1 = [1.0 for _ in range(len(all_entries))]
    p_a_c0 = [1.0 for _ in range(len(all_entries))]

    for i in range(len(all_entries)):
      for j in range(k):
        p_a_c1[i] *= self.cpts[j].get_cond_prob(all_entries[i], 1)
        p_a_c0[i] *= self.cpts[j].get_cond_prob(all_entries[i], 0)

    num = 0.0
    den = 0.0

    for i in range(len(p_a_c1)):
      if all_entries[i][index] == 1:
        num += (p_a_c1[i] * self.P_c[1] + p_a_c0[i] * self.P_c[0])
      den += (p_a_c1[i] * self.P_c[1] + p_a_c0[i] * self.P_c[0])

    c_pred = num / den
    clas = 0 if c_pred < 0.5 else 1
    log_c_pred = np.math.log(c_pred)
    return (clas, log_c_pred)


# =========================================================================


# load all data
A_base, C_base = load_vote_data()

def evaluate(classifier_cls, train_subset=False):
  '''
  =======* DO NOT MODIFY this function *=======

  evaluate the classifier specified by classifier_cls using 10-fold cross
  validation
  Params:
   - classifier_cls: either NBClassifier or TANBClassifier
   - train_subset: train the classifier on a smaller subset of the training
    data
  Returns:
   - accuracy as a proportion
   - total number of predicted samples

  '''
  global A_base, C_base

  A, C = A_base, C_base

  # score classifier on specified attributes, A, against provided labels, C
  def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
      c_pred, _ = classifier.classify(entry)
      results.append((c_pred == c))
      pp.append(_)
    return results

  # partition train and test set for 10 rounds
  M, N = A.shape
  tot_correct = 0
  tot_test = 0
  step = M // 10
  for holdout_round, i in enumerate(range(0, M, step)):
    A_train = np.vstack([A[0:i,:], A[i+step:,:]])
    C_train = np.hstack([C[0:i], C[i+step:]])
    A_test = A[i:i+step,:]
    C_test = C[i:i+step]
    if train_subset:
      A_train = A_train[:16,:]
      C_train = C_train[:16]

    # train the classifiers
    classifier = classifier_cls(A_train, C_train)

    train_results = get_classification_results(classifier, A_train, C_train)
    test_results = get_classification_results(classifier, A_test, C_test)
    tot_correct += sum(test_results)
    tot_test += len(test_results)

  return 1.*tot_correct/tot_test, tot_test

def evaluate_incomplete_entry(classifier_cls):
  '''
  TODO: Fill out the function to compute marginal probabilities.

  Params:
   - classifier_cls: either NBClassifier or TANBClassifier
   - train_subset: train the classifier on a smaller subset of the training
    data
  Returns:
   - P_c_pred: P(C = 1 | A_observed) as a scalar.
   - PA_12_eq_1: P(A_12 = 1 | A_observed) as a scalar.

  '''
  global A_base, C_base

  # train a TANB classifier on the full dataset
  classifier = classifier_cls(A_base, C_base)

  # load incomplete entry 1
  entry = load_incomplete_entry()

  c_pred, logP_c_pred = classifier.classify(entry)
  P_c_pred = np.exp(logP_c_pred)
  print('  P(C={}|A_observed) = {:2.4f}'.format(c_pred, P_c_pred))

  # TODO: write code to compute this!
  a_pred, logP_a_pred = classifier.unobserved_probability(entry, index=11)
  PA_12_eq_1 = np.exp(logP_a_pred)
  print('  P(A={}|A_observed) = {:2.4f}'.format(12, PA_12_eq_1))

  # PA_12_eq_1 = None

  return P_c_pred, PA_12_eq_1

def main():
  '''
  (optional) TODO: modify or add calls to evaluate your implemented classifiers.
  '''

  print('Naive Bayes')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples))

  print('TANB Classifier')
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples))

  print('Naive Bayes Classifier on missing data')
  evaluate_incomplete_entry(NBClassifier)

  print('TANB Classifier on missing data')
  evaluate_incomplete_entry(TANBClassifier)

  print('Naive Bayes (Test accuracy on a smaller subset)')
  accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples))

  print('TANB Classifier (Test accuracy on a smaller subset)')
  accuracy, num_examples = evaluate(TANBClassifier, train_subset=True)
  print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    accuracy, num_examples))

if __name__ == '__main__':
  main()
