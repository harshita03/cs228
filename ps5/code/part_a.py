"""
CS 228: Probabilistic Graphical Models
Winter 2019 (instructor: Stefano Ermon)
Starter Code for Part A
"""

from utils import *
import numpy as np
import math
from matplotlib import pyplot as plt


def gaussian(X, mu, sigma):
    # X : 2, mu : 2, sigma : 2 X 2
    assert X.shape == (2,)
    assert mu.shape == (2,)
    assert sigma.shape == (2,2)

    exp_term = -0.5 * np.matmul(np.matmul(np.matrix([(X - mu).T]), np.linalg.inv(sigma)), np.matrix([X - mu]).T)[0,0]
    return np.exp(exp_term) / (2 * np.pi) / np.sqrt(np.linalg.det(sigma))


def estimate_params(X, Z):
    """Perform MLE estimation of model 1 parameters.

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference

    Output: A dictionary with the following keys/values
        pi: (float), estimate of party proportions
        mu0: size (2,) numpy array econding the estimate of the mean of class 0
        mu1: size (2,) numpy array econding the estimate of the mean of class 1
        sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
        sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1

    This function will be autograded.

    Note: your algorithm should work for any value of N and M
    """
    # print(X.shape, Z.shape)
    N, M, _ = X.shape
    pi = 0.0
    mu0 = 0.0
    mu1 = 0.0
    sigma0 = 0.0
    sigma1 = 0.0

    pi = np.sum(1.0 * Z) / (5*M)


    mu0_n = np.array([0.0, 0.0])
    mu0_d = 0.0
    mu1_n = np.array([0.0, 0.0])
    mu1_d = 0.0
    for i in range(N):
        for j in range(M):
            if Z[i, j] == 0:
                mu0_n += X[i, j]
                mu0_d += 1.0
            else:
                mu1_n += X[i, j]
                mu1_d += 1.0
    mu0 = mu0_n / mu0_d
    mu1 = mu1_n / mu1_d

    sigma0_n = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma0_d = 0.0
    sigma1_n = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma1_d = 0.0
    for i in range(N):
        for j in range(M):
            if Z[i, j] == 0:
                sigma0_n += np.matmul(np.matrix(X[i, j] - mu0).T, np.matrix(X[i, j] - mu0))
                sigma0_d += 1.0
            else:
                sigma1_n += np.matmul(np.matrix(X[i, j] - mu1).T, np.matrix(X[i, j] - mu1))
                sigma1_d += 1.0
    sigma0 = sigma0_n / sigma0_d
    sigma1 = sigma1_n / sigma1_d

    return {'pi': pi, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}


def em_update(X, params):
    """ Perform one EM update based on unlabeled data X
    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary, the previous parameters, see the description in estimate_params
    Output: The updated parameter. The output format is identical to estimate_params

    This function will be autograded.

    Note: You will most likely need to use the function estimate_z_prob_given_x
    """
    N, M, _ = X.shape

    mu0 = params['mu0']
    mu1 = params['mu1']

    Z = estimate_z_prob_given_x(X, params)

    new_pi = np.sum(1.0 * Z) / (N * M)

    mu0_n = np.array([0.0, 0.0])
    mu0_d = 0.0
    mu1_n = np.array([0.0, 0.0])
    mu1_d = 0.0
    for i in range(N):
        for j in range(M):
            mu0_n += X[i, j] * (1 - Z[i, j])
            mu0_d += (1 - Z[i, j])
            mu1_n += X[i, j] * Z[i, j]
            mu1_d += Z[i, j]
    new_mu0 = mu0_n / mu0_d
    new_mu1 = mu1_n / mu1_d

    sigma0_n = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma0_d = 0.0
    sigma1_n = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma1_d = 0.0
    for i in range(N):
        for j in range(M):
            sigma0_n += (1 - Z[i, j]) * np.matmul(np.matrix(X[i, j] - new_mu0).T, np.matrix(X[i, j] - new_mu0))
            sigma0_d += (1 - Z[i, j])
            sigma1_n += Z[i, j] * np.matmul(np.matrix(X[i, j] - new_mu1).T, np.matrix(X[i, j] - new_mu1))
            sigma1_d += Z[i, j]
    new_sigma0 = sigma0_n / sigma0_d
    new_sigma1 = sigma1_n / sigma1_d

    params = {'pi': new_pi, 'mu0': new_mu0, 'mu1': new_mu1, 'sigma0': new_sigma0, 'sigma1': new_sigma1}

    return params


def estimate_z_prob_given_x(X, params):
    """ Estimate p(z_{ij}|x_{ij}, theta)
    Input:
        X: Identical to the function em_update
        params: Identical to the function em_update
    Output: A 2D numpy array z_prob with the same size as X.shape[0:2],
            z_prob[i, j] should store the value of p(z_{ij}|x_{ij}, theta)
            Note: it should be a normalized probability

    This function will be autograded.
    """
    # print(X.shape)
    N, M, _ = X.shape
    Z = np.ndarray(shape=(N, M))

    pi = params['pi']
    mu0 = params['mu0']
    mu1 = params['mu1']
    sigma0 = params['sigma0']
    sigma1 = params['sigma1']

    z_prob = 0.0

    for i in range(N):
        for j in range(M):
            x = X[i,j]
            Z[i, j] = pi * gaussian(x, mu1, sigma1) / (pi * gaussian(x, mu1, sigma1) + (1 - pi) * gaussian(x, mu0, sigma0))

    return Z
    # return z_prob


def compute_log_likelihood(X, params):
    """ Estimate the log-likelihood of the entire data log p(X|theta)
    Input:
        X: Identical to the function em_update
        params: Identical to the function em_update
    Output A real number representing the log likelihood

    This function will be autograded.

    Note: You will most likely need to use the function estimate_z_prob_given_x
    """
    # print(X.shape)
    N, M, _ = X.shape
    likelihood = 0.0

    pi = params['pi']
    mu0 = params['mu0']
    mu1 = params['mu1']
    sigma0 = params['sigma0']
    sigma1 = params['sigma1']

    for i in range(N):
        for j in range(M):
            x = X[i][j]
            # t0 = np.log(1 - pi) + np.log(gaussian(x, mu0, sigma0))
            # t1 = np.log(pi) + np.log(gaussian(x, mu1, sigma1))
            # if math.isnan(t0) or math.isnan(t1):
            #     print("nan: ", x, t0, t1, gaussian(x, mu0, sigma0), gaussian(x, mu1, sigma1))
            #     break
            # tmax = max(t0, t1)
            # likelihood += tmax
            # likelihood += np.log(np.exp(t0 - tmax) + np.exp(t1 - tmax))

            likelihood += np.log(pi * gaussian(x, mu1, sigma1) + (1 - pi) * gaussian(x, mu0, sigma0))
    print(likelihood)
    return likelihood



if __name__ == '__main__':
    #===============================================================================
    # This runs the functions that you have defined to produce the answers to the
    # assignment problems
    #===============================================================================

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix()
    X_unlabeled = read_unlabeled_matrix()

    # pt a.i
    params = estimate_params(X_labeled, Z_labeled)

    colorprint("MLE estimates for PA part a.i:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    # pt a.ii

    params = estimate_params(X_labeled, Z_labeled)  # Initialize
    params = get_random_params()
    likelihoods = []
    while True:
        likelihoods.append(compute_log_likelihood(X_unlabeled, params))
        if len(likelihoods) > 2 and likelihoods[-1] - likelihoods[-2] < 0.01:
            break
        params = em_update(X_unlabeled, params)

    colorprint("MLE estimates for PA part a.ii:", "teal")
    colorprint("\tpi: %s\n\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s"
        %(params['pi'], params['mu0'], params['mu1'], params['sigma0'], params['sigma1']), "red")

    plt.plot(likelihoods)
    plt.show()
