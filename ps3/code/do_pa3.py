###############################################################################
# Finishes PA 3
# author: Ya Le, Billy Jun, Xiaocheng Li
# date: Jan 25, 2018


## Edited by Zhangyuan Wang, 01/2019
###############################################################################

## Utility code for PA3
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from factor_graph import *
from factors import *


def loadLDPC(name):
    """
    :param - name: the name of the file containing LDPC matrices

    return values:
    G: generator matrix
    H: parity check matrix

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    """
    A = sio.loadmat(name)
    G = A['G']
    H = A['H']
    return G, H


def loadImage(fname, iname):
    '''
    :param - fname: the file name containing the image
    :param - iname: the name of the image
    (We will provide the code using this function, so you don't need to worry too much about it)

    return: image data in matrix form
    '''
    img = sio.loadmat(fname)
    return img[iname]


def applyChannelNoise(y, epsilon):
    '''
    :param y - codeword with 2N entries
    :param epsilon - the probability that each bit is flipped to its complement

    return corrupt message yTilde
    yTilde_i is obtained by flipping y_i with probability epsilon

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    ###############################################################################
    # TODO: Your code here!
    m, _ = y.shape
    yFlat = y.reshape(m)
    yTildeFlat = np.array([i if np.random.random() > epsilon else 1 - i for i in yFlat])
    yTilde = yTildeFlat.reshape(m, 1)

    # raise NotImplementedError()
    ###############################################################################
    assert y.shape == yTilde.shape
    return yTilde


def encodeMessage(x, G):
    '''
    :param - x orginal message
    :param[in] G generator matrix
    :return codeword y=Gx mod 2

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    return np.mod(np.dot(G, x), 2)


def constructFactorGraph(yTilde, H, epsilon):
    '''
    Args
    - yTilde: np.array, shape [2N, 1], observed codeword containing 0's and 1's
    - H: np.array, shape [N, 2N], parity check matrix
    - epsilon: float, probability that each bit is flipped to its complement

    Returns: FactorGraph

    You should consider two kinds of factors:
    - M unary factors
    - N each parity check factors

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    N = H.shape[0]
    M = H.shape[1]
    G = FactorGraph(numVar=M, numFactor=N+M)
    G.var = list(range(M))
    ##############################################################
    # To do: your code starts here
    index = 0
    #     #
    for i in range(M):
        scope = [i]
        card = [2]
        # print(yTilde, yTilde[i])
        val = np.array([1 - epsilon if yTilde[i] == 0 else epsilon, epsilon if yTilde[i] == 0 else 1 - epsilon])
        factor = Factor(scope=scope, card=card, val=val)
        G.factors.append(factor)
        G.varToFactor[i].append(index)
        G.factorToVar[index].append(i)
        index += 1

    for i in range(N):
        scope = [j for j in range(M) if H[i][j] == 1]
        card = [2 for _  in range(len(scope))]
        val = np.zeros(card)
        for indexes in itertools.product([0, 1], repeat=len(card)):
            val[indexes] = 1.0 if np.mod(np.sum(indexes), 2) == 0 else 0.0
        factor = Factor(scope=scope, card=card, val=val)
        G.factors.append(factor)
        for s in scope:
            G.varToFactor[s].append(index)
            G.factorToVar[index].append(s)
        index += 1

    # raise NotImplementedError()
    ##############################################################
    return G


def do_part_a():
    yTilde = np.array([1, 1, 1, 1, 1, 1]).reshape(6, 1)
    print("yTilde.shape", yTilde.shape)
    H = np.array([
        [0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 1]])
    epsilon = 0.05
    G = constructFactorGraph(yTilde, H, epsilon)
    ##############################################################
    # To do: your code starts here
    # Design two invalid codewords ytest1, ytest2 and one valid codewords
    # ytest3. Report their weights respectively.
    ytest1 = np.array([0, 1, 1, 0, 1, 0])
    ytest2 = np.array([1, 0, 1, 1, 0, 1])
    ytest3 = np.array([1, 0, 1, 1, 1, 1])
    ##############################################################
    print(G.evaluateWeight(ytest1),
          G.evaluateWeight(ytest2),
          G.evaluateWeight(ytest3))


def sanity_check_noise():
    '''
    Sanity check applyChannelNoise to make sure bits are flipped at
    a reasonable proportion.
    '''
    N = 256
    epsilon = 0.05
    err_percent = 0
    num_trials = 1000
    x = np.zeros((N, 1), dtype='int32')
    for _ in range(num_trials):
        x_noise = applyChannelNoise(x, epsilon)
        err_percent += 1.0*np.sum(x_noise)/N
    err_percent /= num_trials
    assert abs(err_percent-epsilon) < 0.005


def do_part_b(fixed=False, npy_file=None):
    '''
    We provide you an all-zero initialization of message x. If fixed=True and
    `npy_file` is not given, you should apply noise on y to get yTilde.
    Otherwise, load in the npy_file to get yTilde. Then do loopy BP to obtain
    the marginal probabilities of the unobserved y_i's.

    Args
    - fixed: bool, False if using random noise, True if loading from given npy file
    - npy_file: str, path to npy file, must be specified when fixed=True
    '''
    G, H = loadLDPC('ldpc36-128.mat')

    print((H.shape))
    epsilon = 0.05
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)
    if not fixed:
        yTilde = applyChannelNoise(y, epsilon)
        print("Applying random noise at eps={}".format(epsilon))
    else:
        assert npy_file is not None
        yTilde = np.load(npy_file)
        print("Loading yTilde from {}".format(npy_file))
    ##########################################################################################
    # To do: your code starts here
    G = constructFactorGraph(yTilde, H, epsilon)
    G.runParallelLoopyBP(50)
    print(yTilde.reshape((1, yTilde.shape[0]))[0])
    print(G.getMarginalMAP())
    flippedBits = np.sum(np.abs(np.array(G.getMarginalMAP()) - yTilde.reshape((1, yTilde.shape[0]))[0]))
    print('number of flipped bits : ', flippedBits)
    ones_prob = [G.estimateMarginalProbability(v)[1] for v in G.var]

    plt.figure()
    plt.title('Plot of the estimated posterior probability P(Yi=1|Y~)')
    plt.ylabel('Probability of Bit Being 1')
    plt.xlabel('Bit Index of Received Message')
    # plt.bar(range(len(G.var)), values)
    plt.plot(range(len(G.var)), ones_prob)
    # plt.savefig('5c', bbox_inches='tight')
    plt.show()
    plt.close()

    # Verify we get a valid codeword.
    MMAP = G.getMarginalMAP()
    print("The probability of our assignment is %s." % G.evaluateWeight(MMAP))


##############################################################


def do_part_cd(numTrials, error, iterations=50):
    '''
    param - numTrials: how many trials we repreat the experiments
    param - error: the transmission error probability
    param - iterations: number of Loopy BP iterations we run for each trial
    '''
    G, H = loadLDPC('ldpc36-128.mat')
    epsilon = error
    N = G.shape[1]
    x = np.zeros((N, 1), dtype='int32')
    y = encodeMessage(x, G)

    plt.figure()
    plt.title('Hamming Distance v/s number of iterations for different trials')
    plt.xlabel('Number of iterations')
    plt.ylabel('Hamming distance')

    ##############################################################
    # To do: your code starts here
    for t in range(numTrials):
        yTilde = applyChannelNoise(y, epsilon)
        G = constructFactorGraph(yTilde, H, epsilon)
        hammingDist = []
        for it in range(iterations):
            if it % 10 == 0:
                print(t, it)
            G.runParallelLoopyBP(1)
            ypred = G.getMarginalMAP()
            hammingDist.append(sum(ypred))
        plt.plot(range(iterations), hammingDist)
    plt.show()

    ##############################################################


def do_part_ef(error):
    '''
    param - error: the transmission error probability
    '''
    G, H = loadLDPC('ldpc36-1600.mat')
    img = loadImage('images.mat', 'cs242')
    print(G.shape, H.shape, img.shape)
    ##############################################################
    # To do: your code starts here
    # You should flattern img first and treat it as the message x in the previous parts.
    imgFlat = img.reshape((1600, 1))
    y = encodeMessage(imgFlat, G)
    yTilde = applyChannelNoise(y, error)
    G = constructFactorGraph(yTilde, H, error)
    print('factor graph constructed')
    plt.imshow(img)
    plt.savefig('5f')
    for i in range(31):
        print(i)
        G.runParallelLoopyBP(1)
        if i in [0, 1, 2, 3, 5, 10, 20, 30]:
            imgNew = np.array(G.getMarginalMAP()[:len(G.getMarginalMAP())/2]).reshape(img.shape)
            plt.imshow(imgNew)
            plt.savefig('5f'+str(i))
    
    ################################################################


if __name__ == '__main__':
    print('Doing part (a): Should see 0.0, 0.0, >0.0')
    do_part_a()
    print("Doing sanity check applyChannelNoise")
    sanity_check_noise()
    print('Doing part (b) fixed')
    # do_part_b(fixed=True, npy_file='part_b_test_1.npy')    # This should perfectly recover original code
    # do_part_b(fixed=True, npy_file='part_b_test_2.npy')    # This may not recover at perfect probability
    print('Doing part (b) random')
    # do_part_b(fixed=False)
    print('Doing part (c)')
    # do_part_cd(10, 0.06)
    print('Doing part (d)')
    # do_part_cd(10, 0.08)
    # do_part_cd(10, 0.10)
    print('Doing part (e)')
    # do_part_ef(0.06)
    print('Doing part (f)')
    do_part_ef(0.10)
