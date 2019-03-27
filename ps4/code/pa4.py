# Gibbs sampling algorithm to denoise an image
# Author : Gunaa AV, Isaac Caswell
# Edits : Bo Wang, Kratarth Goel, Aditya Grover, Stephanie Wang
# Date : 2/10/2019

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

MAX_BURNS=100
MAX_SAMPLES=1000
# "add /" at the end.
OUTPUT_DIR=""

def markov_blanket(i, j, Y, X):
    '''Gets the values of the Markov blanket of Y_ij.

    Args
    - i: int, row index of Y
    - j: int, col index of Y
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns
    - blanket: list, values of the Markov blanket of Y_ij

    Example: if i = j = 1, the function should return
        [Y[0,1], Y[1,0], Y[1,2], Y[2,1], X[1,1]]

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    blanket = []
    ########
    # TODO: Your code here!
    blanket = [Y[i-1, j], Y[i, j-1], Y[i, j+1], Y[i+1, j], X[i, j]]

    # raise NotImplementedError()
    ########
    return blanket


def sampling_prob(markov_blanket):
    '''Computes P(X=1 | MarkovBlanket(X)).

    Args
    - markov_blanket: list, values of a variable's Markov blanket, e.g. [1,1,-1,1,-1]
        Because beta = eta in this case, the order doesn't matter. See part (a)

    Returns
    - prob: float, the probability of the variable being 1 given its Markov blanket

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    ########
    # TODO: Your code here!

    prob = 1 / (1 + np.exp(-2.0 * sum(markov_blanket)))

    # raise NotImplementedError()
    ########
    return prob


def sample(i, j, Y, X, dumb_sample=False):
    '''Samples a value for Y_ij. It should be sampled by:
    - if dumb_sample=True: the probability conditioned on all other variables
    - if dumb_sample=False: the consensus of Markov blanket

    Args
    - i: int, row index of Y
    - j: int, col index of Y
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns: -1 or +1

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    blanket = markov_blanket(i, j, Y, X)

    if not dumb_sample:
        prob = sampling_prob(blanket)
        return np.random.choice([+1, -1], p=[prob, 1 - prob])
    else:
        return 1 if sum(blanket) > 0 else -1


def compute_energy(Y, X):
    '''Computes the energy E(Y, X) of the current assignment for the image.

    Args
    - Y: np.array, shape [H, W], note that read_txt_file() pads the image with
            0's to help you take care of corner cases
    - X: np.array, shape [H, W]

    Returns: float

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.

    This function can be efficiently implemented in one line with numpy parallel operations.
    You can give it a try to speed up your own experiments. This is not required.
    '''
    energy = 0.0
    ########
    # TODO: Your code here!
    H,W = Y.shape
    energy = -(np.sum(X * Y) + np.sum(Y[:, :W - 1] * Y[:, 1:]) + np.sum(Y[:H - 1, :] * Y[1:, :]))

    # energy = -1 * (np.sum(Y * X) + np.sum(Y * np.roll(Y, 1, axis=0)) + np.sum(Y * np.roll(Y, 1, axis=1)))

    # raise NotImplementedError()
    ########
    return energy


def get_posterior_by_sampling(filename, max_burns, max_samples,
                              initialization='same', logfile=None,
                              dumb_sample=False):
    '''Performs Gibbs sampling and computes the energy of each  assignment for
    the image specified in filename.
    - If dumb_sample=False: runs max_burns iterations of burn in and then
        max_samples iterations for collecting samples
    - If dumb_sample=True: runs max_samples iterations and returns final image

    Args
    - filename: str, file name of image in text format, ends in '.txt'
    - max_burns: int, number of iterations of burn in
    - max_samples: int, number of iterations of collecting samples
    - initialization: str, one of ['same', 'neg', 'rand']
    - logfile: str, file name for storing the energy log (to be used for
        plotting later), see plot_energy()
    - dumb_sample: bool, True to use the trivial reconstruction in part (e)

    Returns
    - posterior: np.array, shape [H, W], type float64, value of each entry is
        the probability of it being 1 (estimated by the Gibbs sampler)
    - Y: np.array, shape [H, W], type np.int32,
        the final image (for dumb_sample=True, in part (e))
    - frequencyZ: dict, keys: count of the number of 1's in the Z region,
        values: frequency of such count

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    print ('Reading file:', filename)
    X = read_txt_file(filename)
    H, W = X.shape
    Y = None

    file_write = None
    if logfile is not None:
        file_write = open(logfile, 'w')

    if initialization == 'same':
        Y = np.copy(X)
    elif initialization == 'neg':
        Y = np.copy(-1 * X)
    elif initialization == 'rand':
        Y = np.random.choice([-1, 1], size = X.shape)

    posterior = np.zeros_like(Y)
    frequencyZ = {}

    ########
    # TODO: Your code here!
    # if not dumb_sample:
    if True:

        for b in range(max_burns):
            for h in range(1, H - 1):
                for w in range(1, W - 1):
                    Y[h, w] = sample(h, w, Y, X, dumb_sample=dumb_sample)
            if file_write is not None:
                file_write.write(str(b) + '\t' + str(compute_energy(Y, X)) + '\tB\n')
            if b % 10 == 0:
                print(str(b) + '/' + str(max_burns) + '\tenergy: ' + str(compute_energy(Y, X)))

        for s in range(max_samples):
            for h in range(1, H - 1):
                for w in range(1, W - 1):
                    Y[h, w] = sample(h, w, Y, X, dumb_sample=dumb_sample)
            ones_count = int(np.sum(1.0 * (Y[125:163, 143:175] == 1)))
            if ones_count not in frequencyZ.keys():
                frequencyZ[ones_count] = 1
            else:
                frequencyZ[ones_count] += 1
            if file_write is not None:
                file_write.write(str(s + max_burns) + '\t' + str(compute_energy(Y, X)) + '\tS\n')
            posterior += 1 * (Y==1)
            if s % 10 == 0:
                print(str(s) + '/' + str(max_samples) + '\tenergy: ' + str(compute_energy(Y, X)))

        posterior = posterior / max_samples

    if file_write is not None:
        file_write.close()
    # raise NotImplementedError()
    ########
    return posterior, Y, frequencyZ


def denoise_image(filename, max_burns, max_samples, initialization='rand',
                  logfile=None, dumb_sample=False):
    '''Performs Gibbs sampling on the image.

    Args:
    - filename: str, file name of image in text format, ends in '.txt'
    - max_burns: int, number of iterations of burn in
    - max_samples: int, number of iterations of collecting samples
    - initialization: str, one of ['same', 'neg', 'rand']
    - logfile: str, file name for storing the energy log (to be used for
        plotting later), see plot_energy()
    - dumb_sample: bool, True to use the trivial reconstruction in part (e)

    Returns
    - denoised: np.array, shape [H, W], type float64,
        denoised image scaled to [0, 1], the zero padding is also removed
    - frequencyZ: dict, keys: count of the number of 1's in the Z region,
        values: frequency of such count

    THIS FUNCTION WILL BE CALLED BY THE AUTOGRADER.
    '''
    posterior, Y, frequencyZ = get_posterior_by_sampling(
        filename, max_burns, max_samples, initialization, logfile=logfile,
        dumb_sample=dumb_sample)

    if dumb_sample:
        denoised = 0.5 * (Y + 1.0)  # change Y scale from [-1, +1] to [0, 1]
    else:
        denoised = np.zeros(posterior.shape, dtype=np.float64)
        denoised[posterior > 0.5] = 1
    return denoised[1:-1, 1:-1], frequencyZ


# ===========================================
# Helper functions for plotting etc
# ===========================================
def plot_energy(filename):
    '''Plots the energy as a function of the iteration number.

    Args
    - filename: str, path to energy log file, each row has three terms
        separated by a '\t'
        - iteration: iteration number
        - energy: the energy at this iteration
        - 'S' or 'B': indicates whether it's burning in or a sample

    e.g.
        1   -202086.0   B
        2   -210446.0   S
        ...
    '''
    x = np.genfromtxt(filename, dtype=None, encoding='utf8')
    its, energies, phases = zip(*x)
    its = np.asarray(its)
    energies = np.asarray(energies)
    phases = np.asarray(phases)

    burn_mask = (phases == 'B')
    samp_mask = (phases == 'S')
    assert np.sum(burn_mask) + np.sum(samp_mask) == len(x), 'Found bad phase'

    its_burn, energies_burn = its[burn_mask], energies[burn_mask]
    its_samp, energies_samp = its[samp_mask], energies[samp_mask]

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_samp, energies_samp, 'b')
    plt.title(filename)
    plt.xlabel('iteration number')
    plt.ylabel('energy')
    plt.legend([p1, p2], ['burn in', 'sampling'])
    plt.savefig('%s.png' % filename)
    # plt.show()
    plt.close()


def read_txt_file(filename):
    '''Reads in image from .txt file and adds a padding of 0's.

    Args
    - filename: str, image filename, ends in '.txt'

    Returns
    - Y: np.array, shape [H, W], type int32, padded with a border of 0's to
        take care of edge cases in computing the Markov blanket
    '''
    f = open(filename, 'r')
    lines = f.readlines()
    height = int(lines[0].split()[1].split('=')[1])
    width = int(lines[0].split()[2].split('=')[1])
    Y = np.zeros([height+2, width+2], dtype=np.int32)
    for line in lines[2:]:
        i, j, val = [int(entry) for entry in line.split()]
        Y[i+1, j+1] = val
    return Y


def convert_to_png(denoised_image, title):
    '''Saves an array as a PNG file with the given title.

    Args
    - denoised_image: np.array, shape [H, W]
    - title: str, title and filename for figure
    '''
    plt.imshow(denoised_image, cmap='gray_r')
    plt.title(title)
    plt.savefig(title + '.png')


def get_error(img_a, img_b):
    '''Computes the fraction of all pixels that differ between two images.

    Args
    - img_a: np.array, shape [H, W]
    - img_b: np.array, shape [H, W]

    Returns: float
    '''
    assert img_a.shape == img_b.shape
    N = img_a.shape[0] * img_a.shape[1]  # number of pixels in an image
    return np.sum(np.abs(img_a - img_b) > 1e-5) / float(N)


#==================================
# doing part (c), (d), (e), (f)
#==================================

def perform_part_c():
    '''
    Run denoise_image() with different initializations and plot out the energy
    functions.
    '''
    ########
    # TODO: Your code here!
    # filenames = 'small.txt', 'noisy_10.txt', 'noisy_20.txt'
    filenames = 'noisy_10.txt', 'noisy_20.txt'
    for filename in filenames:
        for init in ['same', 'neg', 'rand']:
            print(filename, init)
            # output_file = 'output/small_out_' + init
            output_file = 'output/' + filename[:-4] + '_' + init
            denoised_img, _ = denoise_image('data/' + filename, max_burns=100, max_samples=1000, initialization=init, logfile=output_file, dumb_sample=False)
            convert_to_png(denoised_img, output_file + '_ref')
            # convert_to_png(denoised_img, 'output/small_out_' + str(init))
            plot_energy(output_file)

    # raise NotImplementedError()
    ########

    #### plot out the energy functions
    # plot_energy('log_rand')
    # plot_energy('log_neg')
    # plot_energy('log_same')


def perform_part_d():
    '''
    Run denoise_image() with different noise levels of 10% and 20%, and report
    the errors between denoised images and original image. Strip the 0-padding
    before computing the errors. Also, don't forget that denoise_image() strips
    the zero padding and scales them into [0, 1].
    '''
    ########
    # TODO: Your code here!
    filenames = ['noisy_10.txt', 'noisy_20.txt']
    # filenames = ['small.txt']
    X_orig = read_txt_file('data/orig.txt')
    # X_orig = read_txt_file('data/small.txt')
    print('X_orig:' + str(X_orig.shape))

    for filename in filenames:
        output_file = 'output/d_' + filename[:-4]
        print(filename, output_file)
        denoised_img, _ = denoise_image('data/' + filename, max_burns=100, max_samples=1000, logfile=output_file, dumb_sample=False)
        # denoised_img, _ = denoise_image('data/' + filename, max_burns=100, max_samples=100, logfile=output_file, dumb_sample=False)
        convert_to_png(denoised_img, output_file + '_ref_d')
        # print(denoised_img, np.sum(denoised_img))
        X = (X_orig[1:-1, 1:-1] + 1) / 2
        # print(X, np.sum(X))
        print('denoised: ' + str(denoised_img.shape))


        # X_denoised = read_txt_file(output_file)
        quality_of_restoration = 1.0 * np.sum(X == denoised_img) / X.size
        print('Quality of restoration for ' + filename + ' is : ' + str(quality_of_restoration))


    # raise NotImplementedError()
    ########

    #### save denoised images and original image as PNG
    # convert_to_png(denoised_10, 'denoised_10')
    # convert_to_png(denoised_20, 'denoised_20')
    # convert_to_png(orig_img, 'orig_img')


def perform_part_e():
    '''
    Run denoise_image() using dumb sampling with different noise levels of 10%
    and 20%.
    '''
    ########
    # TODO: Your code here!
    filenames = 'noisy_10.txt', 'noisy_20.txt'
    X_orig = read_txt_file('data/orig.txt')
    print('X_orig:' + str(X_orig.shape))

    for filename in filenames:
        output_file = 'output/e_' + filename[:-4]
        print(output_file)
        denoised_img, _ = denoise_image('data/' + filename, max_burns=100, max_samples=1000, initialization='same', logfile=output_file, dumb_sample=True)
        convert_to_png(denoised_img, output_file + '_ref_e')
        X = (X_orig[1:-1, 1:-1] + 1) / 2
        print('denoised: ' + str(denoised_img.shape))

        # X_denoised = read_txt_file(output_file)
        quality_of_restoration = 1.0 * np.sum(X == denoised_img) / X.size
        print('Quality of restoration for ' + filename + ' is : ' + str(quality_of_restoration))

    # raise NotImplementedError()
    ########

    #### save denoised images as PNG
    # convert_to_png(denoised_dumb_10, 'denoised_dumb_10')
    # convert_to_png(denoised_dumb_20, 'denoised_dumb_20')


def perform_part_f():
    '''
    Run Z square analysis
    '''
    MAX_BURNS = 100
    MAX_SAMPLES = 1000

    _, f = denoise_image('./data/noisy_10.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    width = 1.0
    plt.clf()
    plt.bar(list(f.keys()), list(f.values()), width, color='b')
    plt.show()
    _, f = denoise_image('./data/noisy_20.txt', MAX_BURNS, MAX_SAMPLES, initialization='same')
    plt.clf()
    plt.bar(list(f.keys()), list(f.values()), width, color='b')
    plt.show()


if __name__ == '__main__':
    # perform_part_c()
    # perform_part_d()
    # perform_part_e()
    perform_part_f()
