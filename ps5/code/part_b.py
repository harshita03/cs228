"""
CS 228: Probabilistic Graphical Models
Winter 2019 (instructor: Stefano Ermon)
Starter Code for Part B
Author: Shengjia Zhao (sjzhao@stanford.edu)
"""

from utils import *
import numpy as np
import math
from part_a import gaussian
from part_a import estimate_z_prob_given_x

def bgaussian(X, mu, sigma):
    N, M, _ = X.shape
    bg = np.ndarray(shape=(N, M))

    for i in range(N):
        for j in range(M):
            bg[i, j] = gaussian(X[i, j], mu, sigma)

    return bg

def verify_marginal_joint(X, params):
    # log p(y_i=1 | X, theta)
    # log p(z_{ij}=1|X, theta)
    yi, zij = compute_yz_marginal(X, params)
    # log p(y_i=u, z_{ij}=v|X, params)
    yz = compute_yz_joint(X, params)
    N, M,_ = X.shape
    joint = np.ndarray(shape=(N, M))

    for i in range(N):
        for j in range(M):
            joint[i, j] = log_sum_exp(yz[i, j, 0, 1], yz[i, j, 1, 1])

    print((zij - joint) < 1e-8)


def estimate_phi_lambda(Z):
    """Perform MLE estimation of phi and lambda as described in B(i)
    Assumes that Y variables have been estimated using heuristic proposed in the question.
    Input:
        Z: A numpy array of size (N, M), where Z[i, j] = 0 or 1 indicating the party preference
    Output:
        MLE_phi: a real number, estimate of phi
        MLE_lambda: a real number, estimate of lambda

    This function will be autograded.
    """
    MLE_phi = 0.0
    MLE_lambda = 0.0

    M, N = Z.shape
    Y = 1.0 * (np.sum(Z, axis=1) > 0.5*M)

    MLE_phi = np.sum(1.0 * Y) / M
    Y1 = np.array([Y]).T
    MLE_lambda = np.sum(1.0 * (Z * Y1 + (1 - Z) * (1 - Y1))) / (M*N)

    return {'phi': MLE_phi, 'lambda': MLE_lambda}


def compute_yz_marginal(X, params):
    """Evaluate log p(y_i=1|X) and log p(z_{ij}=1|X)

    Input:
        X: A numpy array of size (N, M, 2), where X[i, j] is the 2-dimensional vector
            representing the voter's properties
        params: A dictionary with the current parameters theta, elements include:
            phi: (float), as stated in the question
            lambda: (float), as stated in the question
            mu0: size (2,) numpy array econding the estimate of the mean of class 0
            mu1: size (2,) numpy array econding the estimate of the mean of class 1
            sigma0: size (2,2) numpy array econding the estimate of the covariance of class 0
            sigma1: size (2,2) numpy array econding the estimate of the covariance of class 1
    Output:
        y_prob: An numpy array of size X.shape[0:1]; y_prob[i] store the value of log p(y_i=1|X, theta)
        z_prob: An numpy array of size X.shape[0:2]; z_prob[i, j] store the value of log p(z_{ij}=1|X, theta)

    You should use the log-sum-exp trick to avoid numerical overflow issues (Helper functions in utils.py)
    This function will be autograded.
    """

    N, M, _ = X.shape
    print(N, M)

    phi = params['phi']
    lambdaa = params['lambda']
    mu0 = params['mu0']
    mu1 = params['mu1']
    sigma0 = params['sigma0']
    sigma1 = params['sigma1']

    log_y1 = np.array([0.0 for _ in range(N)])
    log_y0 = np.array([0.0 for _ in range(N)])

    for i in range(N):
        log_y1[i] += np.log(phi)
        log_y0[i] += np.log(1 - phi)
        for j in range(M):
            x = X[i, j]
            log_y1[i] += np.log(gaussian(x, mu0, sigma0) * (1 - lambdaa) + gaussian(x, mu1, sigma1) * lambdaa)
            log_y0[i] += np.log(gaussian(x, mu0, sigma0) * lambdaa + gaussian(x, mu1, sigma1) * (1 - lambdaa))
            if math.isnan(log_y1[i]) or math.isnan(log_y0[i]):
                print("nan: ", gaussian(x, mu0, sigma0), gaussian(x, mu1, sigma1), lambdaa)

    # to normalize
    y_prob = np.log(np.exp(log_y1) / (np.exp(log_y1) + np.exp(log_y0)))
    # print(y_prob)
    # return y_prob, None

    # Takes much longer
    # A => y = 0 and B => y = 1
    z1_A = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    z1_B = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    z0_A = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    z0_B = np.array([[0.0 for _ in range(M)] for _ in range(N)])

    for i in range(N):
        # if i % 10 == 0:
        #     print(i)
        for j in range(M):
            x = X[i, j]
            if phi == 0 or phi == 1 or gaussian(x, mu1, sigma1) == 0 or gaussian(x, mu0, sigma0) == 1 or lambdaa == 0 or lambdaa == 1:
                print('oh no')
                print(phi, gaussian(x, mu1, sigma1), gaussian(x, mu0, sigma0), lambdaa)
                assert False
            z1_A[i, j] += np.log(1 - phi) + np.log(gaussian(x, mu1, sigma1)) + np.log(1 - lambdaa)
            z1_B[i, j] += np.log(phi) + np.log(gaussian(x, mu1, sigma1)) + np.log(lambdaa)
            z0_A[i, j] += np.log(1 - phi) + np.log(gaussian(x, mu0, sigma0)) + np.log(lambdaa)
            z0_B[i, j] += np.log(phi) + np.log(gaussian(x, mu0, sigma0)) + np.log(1 - lambdaa)
            for jp in range(M):
                if j != jp:
                    xp = X[i, jp]
                    z1_A[i, j] += np.log(gaussian(xp, mu0, sigma0) * lambdaa + gaussian(xp, mu1, sigma1) * (1 - lambdaa))
                    z1_B[i, j] += np.log(gaussian(xp, mu0, sigma0) * (1 - lambdaa) + gaussian(xp, mu1, sigma1) * lambdaa)
                    z0_A[i, j] += np.log(gaussian(xp, mu0, sigma0) * lambdaa + gaussian(xp, mu1, sigma1) * (1 - lambdaa))
                    z0_B[i, j] += np.log(gaussian(xp, mu0, sigma0) * (1 - lambdaa) + gaussian(xp, mu1, sigma1) * lambdaa)

            if math.isnan(z1_A[i, j]):
                print('nan z1_A', z1_A[i, j])
                assert False
            if math.isnan(z1_B[i, j]):
                print('nan z1_B', z1_B[i, j])
                assert False
            if math.isnan(z0_A[i, j]):
                print('nan z0_A', z0_A[i, j])
                assert False
            if math.isnan(z0_B[i, j]):
                print('nan z0_B', z0_B[i, j])
                assert False
    print(np.min(z1_A), np.min(z1_B), np.min(z0_A), np.min(z0_B))
    z1 = np.exp(log_sum_exp(z1_A, z1_B))
    z0 = np.exp(log_sum_exp(z0_A, z0_B))
    z_prob = np.log(z1 / (z1 + z0))

    # This is an attempt to broadcast it. But something is clearly wrong here.
    # my0 = np.sum(np.log(bgaussian(X, mu0, sigma0) * lambdaa + bgaussian(X, mu1, sigma1) * (1 - lambdaa)), axis=1)
    # my1 = np.sum(np.log(bgaussian(X, mu0, sigma0) * (1 - lambdaa) + bgaussian(X, mu1, sigma1) * lambdaa), axis=1)
    #
    # mz1_y0 = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    # mz1_y1 = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    # mz0_y0 = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    # mz0_y1 = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    #
    # for i in range(N):
    #     for j in range(M):
    #         x = X[i, j]
    #         mz1_y0[i, j] += my0[i] - np.log(gaussian(x, mu0, sigma0)) - np.log(lambdaa) + np.log(1 - phi)
    #         mz1_y1[i, j] += my1[i] - np.log(gaussian(x, mu0, sigma0)) - np.log(1 - lambdaa) + np.log(phi)
    #         mz0_y0[i, j] += my0[i] - np.log(gaussian(x, mu1, sigma1)) - np.log(1 - lambdaa) + np.log(1 - phi)
    #         mz0_y1[i, j] += my1[i] - np.log(gaussian(x, mu1, sigma1)) - np.log(lambdaa) + np.log(phi)
    # print(np.min(mz1_y0), np.min(mz1_y1), np.min(mz0_y0), np.min(mz0_y1))
    # z1 = np.exp(log_sum_exp(mz1_y0, mz1_y1))
    # z0 = np.exp(log_sum_exp(mz0_y0, mz0_y1))
    # z_prob1 = np.log(z1 / (z1 + z0))
    #
    # for i in range(N):
    #     for j in range(M):
    #         if z_prob[i,j] - z_prob1[i,j] > 1e-8:
    #             print('diff', i, j)
    #             print(mz1_y0[i, j], z1_A[i, j])
    #             print(mz1_y1[i, j], z1_B[i, j])
    #             print(mz0_y0[i, j], z0_A[i, j])
    #             print(mz0_y1[i, j], z0_B[i, j])
    #             assert False
    #         else:
    #             print('no diff', i, j)
    #             print(mz1_y0[i, j], z1_A[i, j])
    #             print(mz1_y1[i, j], z1_B[i, j])
    #             print(mz0_y0[i, j], z0_A[i, j])
    #             print(mz0_y1[i, j], z0_B[i, j])
    #             assert False
    # print(np.sum((z_prob - z_prob1) < 1e-8)/M/N)
    # print(z1_A - mz1_y0)
    return y_prob, z_prob


def compute_yz_joint(X, params):
    """ Compute the joint probability of log p(y_i, z_{ij}|X, params)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        yz_prob: A array of shape (X.shape[0], X.shape[1], 2, 2);
            yz_prob[i, j, u, v] should store the value of log p(y_i=u, z_{ij}=v|X, params)
            Don't forget to normalize your (conditional) probability

    Note: To avoid numerical overflow, you should use log_sum_exp trick (Helper functions in utils.py)

    This function will be autograded.
    """
    yz_prob = 0.0

    N, M, _ = X.shape
    print(N, M)

    phi = params['phi']
    lambdaa = params['lambda']
    mu0 = params['mu0']
    mu1 = params['mu1']
    sigma0 = params['sigma0']
    sigma1 = params['sigma1']

    z1_A = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    z1_B = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    z0_A = np.array([[0.0 for _ in range(M)] for _ in range(N)])
    z0_B = np.array([[0.0 for _ in range(M)] for _ in range(N)])

    for i in range(N):
        # if i % 10 == 0:
        #     print(i)
        for j in range(M):
            x = X[i, j]
            if phi == 0 or phi == 1 or gaussian(x, mu1, sigma1) == 0 or gaussian(x, mu0,
                                                                                 sigma0) == 1 or lambdaa == 0 or lambdaa == 1:
                print('oh no')
                print(phi, gaussian(x, mu1, sigma1), gaussian(x, mu0, sigma0), lambdaa)
                assert False
            z1_A[i, j] += np.log(1 - phi) + np.log(gaussian(x, mu1, sigma1)) + np.log(1 - lambdaa)
            z1_B[i, j] += np.log(phi) + np.log(gaussian(x, mu1, sigma1)) + np.log(lambdaa)
            z0_A[i, j] += np.log(1 - phi) + np.log(gaussian(x, mu0, sigma0)) + np.log(lambdaa)
            z0_B[i, j] += np.log(phi) + np.log(gaussian(x, mu0, sigma0)) + np.log(1 - lambdaa)
            for jp in range(M):
                if j != jp:
                    xp = X[i, jp]
                    z1_A[i, j] += np.log(
                        gaussian(xp, mu0, sigma0) * lambdaa + gaussian(xp, mu1, sigma1) * (1 - lambdaa))
                    z1_B[i, j] += np.log(
                        gaussian(xp, mu0, sigma0) * (1 - lambdaa) + gaussian(xp, mu1, sigma1) * lambdaa)
                    z0_A[i, j] += np.log(
                        gaussian(xp, mu0, sigma0) * lambdaa + gaussian(xp, mu1, sigma1) * (1 - lambdaa))
                    z0_B[i, j] += np.log(
                        gaussian(xp, mu0, sigma0) * (1 - lambdaa) + gaussian(xp, mu1, sigma1) * lambdaa)

            if math.isnan(z1_A[i, j]):
                print('nan z1_A', z1_A[i, j])
                assert False
            if math.isnan(z1_B[i, j]):
                print('nan z1_B', z1_B[i, j])
                assert False
            if math.isnan(z0_A[i, j]):
                print('nan z0_A', z0_A[i, j])
                assert False
            if math.isnan(z0_B[i, j]):
                print('nan z0_B', z0_B[i, j])
                assert False
    print(np.min(z1_A), np.min(z1_B), np.min(z0_A), np.min(z0_B))
    # z1 = np.exp(log_sum_exp(z1_A, z1_B))
    # z0 = np.exp(log_sum_exp(z0_A, z0_B))
    # z_prob = np.log(z1 / (z1 + z0))

    lse_trick = np.ndarray(shape=(N, M))
    for i in range(N):
        for j in range(M):
            lse_trick[i, j] = log_sum_exp(log_sum_exp(log_sum_exp(z1_A[i, j], z1_B[i, j]), z0_A[i, j]), z0_B[i, j])

    yz_prob = np.ndarray(shape=(N, M, 2, 2))
    for i in range(N):
        for j in range(M):
            yz_prob[i, j, 0, 0] = z0_A[i, j] - lse_trick[i, j]
            yz_prob[i, j, 0, 1] = z1_A[i, j] - lse_trick[i, j]
            yz_prob[i, j, 1, 0] = z0_B[i, j] - lse_trick[i, j]
            yz_prob[i, j, 1, 1] = z1_B[i, j] - lse_trick[i, j]

    return yz_prob


def em_step(X, params):
    """ Make one EM update according to question B(iii)
    Input:
        X: As usual
        params: A dictionary containing the old parameters, refer to compute compute_yz_marginal
    Output:
        new_params: A dictionary containing the new parameters

    This function will be autograded.
    """
    N, M, _ = X.shape
    new_params = {}

    y_prob, z_prob = compute_yz_marginal(X, params)
    # A array of shape(X.shape[0], X.shape[1], 2, 2); yz_prob[i, j, u, v] should
    # store the value of log p(y_i=u, z_{ij} = v | X, params)
    yz_prob = compute_yz_joint(X, params)

    y_prob = np.exp(y_prob)
    z_prob = np.exp(z_prob)
    yz_prob = np.exp(yz_prob)

    phi = np.sum(y_prob) / N

    lambdaa = 0.0
    for i in range(N):
        for j in range(M):
            lambdaa += yz_prob[i, j, 0, 0] + yz_prob[i, j, 1, 1]
    lambdaa /= (M*N)

    # lambdaa = np.sum(yz_prob, axis=(0, 1))[0 ,0] + np.sum(yz_prob, axis=(0, 1))[0 ,0]

    mu0_n = np.array([0.0, 0.0])
    mu0_d = 0.0
    mu1_n = np.array([0.0, 0.0])
    mu1_d = 0.0
    for i in range(N):
        for j in range(M):
            x = X[i, j]
            z = z_prob[i, j]
            mu0_n += x * (1 - z)
            mu0_d += (1 - z)
            mu1_n += x * z
            mu1_d += z

    mu0 = mu0_n / mu0_d
    mu1 = mu1_n / mu1_d
    # mu0 = np.sum(X * (1 - z_prob)) / np.sum(1 - z_prob)
    # mu1 = np.sum(X * z_prob) / np.sum(z_prob)

    sigma0_n = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma0_d = 0.0
    sigma1_n = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    sigma1_d = 0.0
    for i in range(N):
        for j in range(M):
            x = X[i, j]
            z = z_prob[i, j]
            sigma0_n += (1 - z) * np.matmul(np.matrix(x - mu0).T, np.matrix(x - mu0))
            sigma0_d += (1 - z)
            sigma1_n += z * np.matmul(np.matrix(x - mu1).T, np.matrix(x - mu1))
            sigma1_d += z
    sigma0 = sigma0_n / sigma0_d
    sigma1 = sigma1_n / sigma1_d

    # sigma0 = np.ndarray(shape=(2,2))
    # sigma1 = np.ndarray(shape=(2, 2))
    # for i in range(N):
    #     for j in range(M):
    #         x = X[i, j]
    #         sigma0[i, j] = np.matmul(np.matrix(x - mu0).T, np.matrix(x - mu0)) * (1 - z_prob[i, j])
    #         sigma1[i, j] = np.matmul(np.matrix(x - mu1).T, np.matrix(x - mu1)) * z_prob[i, j]

    new_params = {'phi': phi, 'lambda': lambdaa, 'mu0': mu0, 'mu1': mu1, 'sigma0': sigma0, 'sigma1': sigma1}
    return new_params


def compute_log_likelihood(X, params):
    """ Compute the log likelihood log p(X) under current parameters.
    To compute this you can first call the function compute_yz_joint

    Input:
        X: As usual
        params: As in the description for compute_yz_joint
    Output: A real number representing log p(X)

    This function will be autograded
    """
    N, M, _ = X.shape
    likelihood = 0.0

    phi = params['phi']
    lambdaa = params['lambda']
    mu0 = params['mu0']
    mu1 = params['mu1']
    sigma0 = params['sigma0']
    sigma1 = params['sigma1']

    px_z0_y0 = np.log(bgaussian(X, mu0, sigma0)) + np.log(lambdaa)
    px_z1_y0 = np.log(bgaussian(X, mu1, sigma1)) + np.log(1 - lambdaa)
    px_z0_y1 = np.log(bgaussian(X, mu0, sigma0)) + np.log(1 - lambdaa)
    px_z1_y1 = np.log(bgaussian(X, mu1, sigma1)) + np.log(lambdaa)

    px_y0 = np.ndarray(shape=(N, M))
    px_y1 = np.ndarray(shape=(N, M))
    for i in range(N):
        for j in range(M):
            px_y0[i, j] = log_sum_exp(px_z0_y0[i, j], px_z1_y0[i, j])
            px_y1[i, j] = log_sum_exp(px_z0_y1[i, j], px_z1_y1[i, j])

    pxr_y0 = np.sum(px_y0, axis=1) + np.log(1 - phi)
    pxr_y1 = np.sum(px_y1, axis=1) + np.log(phi)

    pxr = np.ndarray(shape=N)
    for i in range(N):
        pxr[i] = log_sum_exp(pxr_y0[i], pxr_y1[i])

    px = np.sum(pxr)

    return px #likelihood


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Read data
    X_labeled, Z_labeled = read_labeled_matrix()
    X_unlabeled = read_unlabeled_matrix()

    # Question B(i)
    from part_a import estimate_params
    params = estimate_params(X_labeled, Z_labeled)
    params.update(estimate_phi_lambda(Z_labeled))

    colorprint("MLE estimates for PA part b.i:", "teal")
    colorprint("\tMLE phi: %s\n\tMLE lambda: %s\n"%(params['phi'], params['lambda']), 'red')

    # # Question B(ii)
    # params = get_random_params()
    # verify_marginal_joint(X_unlabeled, params)
    # y_prob, z_prob = compute_yz_marginal(X_unlabeled, params)   # Get the log probability of y and z conditioned on x
    # colorprint("Your predicted party preference:", "teal")
    # colorprint(str((y_prob > np.log(0.5)).astype(np.int)), 'red')
    #
    # plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
    #             c=np.array(['red', 'blue'])[(z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    # plt.plot(params['mu0'][0], params['mu0'][1], 'ko')
    # plt.plot(params['mu1'][0], params['mu1'][1], 'ko')
    # plt.show()


    # # # Question B(iii)
    likelihoods = []
    for i in range(10):
        likelihoods.append(compute_log_likelihood(X_unlabeled, params))
        params = em_step(X_unlabeled, params)
    # colorprint("MLE estimates for PA part b.iv:", "teal")
    # colorprint("\tmu_0: %s\n\tmu_1: %s\n\tsigma_0: %s\n\tsigma_1: %s\n\tphi: %s\n\tlambda: %s\n"
    #            % (params['mu0'], params['mu1'], params['sigma0'], params['sigma1'], params['phi'], params['lambda']), "red")
    # plt.plot(likelihoods)
    # plt.show()
    # # #
    # # # Question B(iv)
    y_prob, z_prob = compute_yz_marginal(X_unlabeled, params)
    colorprint("Your predicted party preference:", "teal")
    colorprint(str((y_prob > np.log(0.5)).astype(np.int)), 'red')
    plt.scatter(X_unlabeled[:, :, 0].flatten(), X_unlabeled[:, :, 1].flatten(),
                c=np.array(['red', 'blue'])[(z_prob > np.log(0.5)).astype(np.int).flatten()], marker='+')
    plt.plot(params['mu0'][0], params['mu0'][1], 'ko')
    plt.plot(params['mu1'][0], params['mu1'][1], 'ko')
    plt.show()
    #
    # N, = y_prob.shape
    # for i in range(N):
    #     print(np.exp(y_prob[i]))

