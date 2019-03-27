"""
CS 228: Probabilistic Graphical Models
Winter 2018
Programming Assignment 1: Bayesian Networks

Author: Aditya Grover
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.io import loadmat
# from scipy.special import logsumexp

def plot_histogram(data, title='histogram', xlabel='value', ylabel='frequency', savefile='hist'):
	'''
	Plots a histogram.
	'''

	plt.figure()
	plt.hist(data)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(savefile, bbox_inches='tight')
	plt.show()
	plt.close()

	return

def get_p_z1(z1_val):
	'''
	Helper. Computes the prior probability for variable z1 to take value z1_val.
	P(Z1=z1_val)
	'''

	return bayes_net['prior_z1'][z1_val]

def get_p_z2(z2_val):
	'''
	Helper. Computes the prior probability for variable z2 to take value z2_val.
	P(Z2=z2_val)
	'''

	return bayes_net['prior_z2'][z2_val]

def get_p_xk_cond_z1_z2(z1_val, z2_val, k):
	'''
	Note: k ranges from 1 to 784.
	Helper. Computes the conditional probability that variable xk assumes value 1
	given that z1 assumes value z1_val and z2 assumes value z2_val
	P(Xk = 1 | Z1=z1_val , Z2=z2_val)
	'''

	return bayes_net['cond_likelihood'][(z1_val, z2_val)][0, k-1]

def get_p_x_cond_z1_z2(z1_val, z2_val):
	'''
	Computes the conditional probability of the entire vector x,
	given that z1 assumes value z1_val and z2 assumes value z2_val
	TODO
	'''
	return np.array([get_p_xk_cond_z1_z2(z1_val, z2_val, i+1) for i in range(784)])
	# pass

def get_pixels_sampled_from_p_x_joint_z1_z2():
	'''
	This function should sample from the joint probability distribution specified by the model,
	and return the sampled values of all the pixel variables (x).
	Note that this function should return the sampled values of ONLY the pixel variables (x),
	discarding the z part.
	TODO.
	'''
	z1 = bayes_net['prior_z1']
	z2 = bayes_net['prior_z2']
	z1_val = np.random.choice(z1.keys(), 1, p=z1.values())[0]
	z2_val = np.random.choice(z2.keys(), 1, p=z2.values())[0]
	px = get_p_x_cond_z1_z2(z1_val, z2_val)
	pixels = np.array([1 if np.random.random() < px[i] else 0 for i in range(784)])
	return pixels
	# pass

def get_conditional_expectation(data):
	'''
	TODO
	'''
	n, _ = data.shape
	pdata = np.exp(get_ll(data))
	print(pdata.shape)
	Z1 = np.linspace(-3.0, 3.0, 25)
	Z2 = np.linspace(-3.0, 3.0, 25)
	cond_ps = {}
	for z1 in Z1:
		cond_ps[z1] = {}
		for z2 in Z2:
			pz1 = get_p_z1(z1)
			pz2 = get_p_z2(z2)
			px_cond = get_p_x_cond_z1_z2(z1, z2)
			ones_p = data * px_cond
			zeros_p = (1 - data) * (1 - px_cond)
			total_p = ones_p + zeros_p
			p_x_cond_z = np.prod(total_p, axis = 1)
			p_z_cond_x = p_x_cond_z * pz1 * pz2 / pdata
			cond_ps[z1][z2] = p_z_cond_x

	avg_z1 = np.zeros(n)
	avg_z2 = np.zeros(n)
	for z1 in cond_ps.keys():
		for z2 in cond_ps[z1].keys():
			avg_z1 += cond_ps[z1][z2] * z1
			avg_z2 += cond_ps[z1][z2] * z2

	return avg_z1, avg_z2

	# pass

def q4():
	'''
	Plots the pixel variables sampled from the joint distribution as 28 x 28 images.
	Your job is to implement get_pixels_sampled_from_p_x_joint_z1_z2
	'''

	plt.figure()
	for i in range(5):
	    plt.subplot(1, 5, i+1)
	    plt.imshow(get_pixels_sampled_from_p_x_joint_z1_z2().reshape(28, 28), cmap='gray')
	    plt.title('Sample: ' + str(i+1))
	plt.tight_layout()
	plt.savefig('a4', bbox_inches='tight')
	plt.show()
	plt.close()

	return

def q5():
	'''
	Plots the expected images for each latent configuration on a 2D grid.
	Your job is to implement get_p_x_cond_z1_z2
	'''

	canvas = np.empty((28*len(disc_z1), 28*len(disc_z2)))
	for i, z1_val in enumerate(disc_z1):
	    for j, z2_val in enumerate(disc_z2):
	        canvas[(len(disc_z1)-i-1)*28:(len(disc_z2)-i)*28, j*28:(j+1)*28] = \
	        get_p_x_cond_z1_z2(z1_val, z2_val).reshape(28, 28)

	plt.figure()
	plt.imshow(canvas, cmap='gray')
	plt.tight_layout()
	plt.savefig('a5', bbox_inches='tight')
	plt.show()
	plt.close()

	return

def get_p_z1_z2_x(z1_val, z2_val, data):
	# print('======== get_p_z1_z2_x')
	each_val = get_p_x_cond_z1_z2(z1_val, z2_val)
	# print(each_val[:10])
	p = 1
	each_val_p = []
	for i in range(784):
		if data[i] == 1:
			p *= each_val[i]
			each_val_p.append(each_val[i])
		else:
			p *= (1 - each_val[i])
			each_val_p.append(1 - each_val[i])
	# print("old approach: \n", each_val_p[:10])
	# print(np.prod(np.array(each_val_p)), "\n========")
	# print(p * get_p_z1(z1_val) * get_p_z2(z1_val))
	return p * get_p_z1(z1_val) * get_p_z2(z1_val)
	# return reduce(lambda x,y: x*y if y>0 else x, each_val*data, 1) * reduce(lambda x,y: x*(1-y) if y>0 else x, each_val*(1-data), 1) * get_p_z1(z1_val) * get_p_z2(z1_val)

def get_p_x(data):
	p = 0
	for z1_val in np.linspace(-3.0, 3.0, 25):
		for z2_val in np.linspace(-3.0, 3.0, 25):
			p += get_p_z1_z2_x(z1_val, z2_val, data)
			# break
		# break
	print(p)
	return p

def get_ll_x(data):
	return np.log(get_p_x(data))

def get_p_given_z1_z2(z1_val, z2_val, data):
	each_val = get_p_x_cond_z1_z2(z1_val, z2_val)
	ones_p = each_val * data
	zeros_p = (1 - each_val) * (1 - data)
	total_p = ones_p + zeros_p
	log_p = np.log(total_p)
	return np.sum(log_p, axis = 1)
	# return np.prod(total_p, axis = 1)

def get_p(data):
	n, d = data.shape
	p = np.array([0.0 for i in range(n)])
	logps = np.empty([n, 25, 25])
	x = 0

	for z1 in np.linspace(-3.0, 3.0, 25):
		y = 0
		for z2 in np.linspace(-3.0, 3.0, 25):
			# p_z1_z2 = get_p_given_z1_z2(z1, z2, data) * get_p_z1(z1) * get_p_z1(z2)
			p_z1_z2 = get_p_given_z1_z2(z1, z2, data) + np.log(get_p_z1(z1)) + np.log(get_p_z1(z2))
			logps[:, x, y] = p_z1_z2
			y += 1
			p += np.exp(p_z1_z2)
		x += 1
		print(z1)

	print("logps: ", logps.shape)
	log_ps = np.reshape(logps, (n, 25 * 25))
	print("log_ps: ", log_ps.shape)
	max_ps = np.max(log_ps, axis = 1)
	print("max_ps: ", max_ps.shape)
	norm_ps = log_ps - np.array([max_ps]).T
	print("norm_ps: ", norm_ps.shape)

	return max_ps + np.log(np.sum(np.exp(norm_ps), axis = 1))
	# print(np.min(p))
	# print(sorted(list(p))[:25])
	# p = np.sum(np.exp(logps), axis = 1)
	# return p

def get_ll(data):
	# return np.log(get_p(data))
	# due to log-exp-exp trick, i can't take log here.
	# it has to be done as part of computing p itself.
	return get_p(data)

def q6():
	'''
	Loads the data and plots the histograms. Rest is TODO.
	'''

	mat = loadmat('q6.mat')
	val_data = mat['val_x']
	test_data = mat['test_x']

	'''
	TODO
	'''

	print(val_data.shape)
	print(test_data.shape)

	ll = get_ll(val_data)
	avg = np.mean(ll)
	std = np.std(ll)
	print(avg, std)
	is_corrupt = lambda x : True if x < (avg - 3 * std) or x > (avg + 3 * std) else False
	test_ll = get_ll(test_data)
	# print(sorted(list(test_ll)))

	real_marginal_log_likelihood = list(filter(lambda x : not is_corrupt(x), test_ll))
	corrupt_marginal_log_likelihood = list(filter(lambda x : is_corrupt(x), test_ll))

	plot_histogram(real_marginal_log_likelihood, title='Histogram of marginal log-likelihood for real data',
			 xlabel='marginal log-likelihood', savefile='a6_hist_real')

	plot_histogram(corrupt_marginal_log_likelihood, title='Histogram of marginal log-likelihood for corrupted data',
		xlabel='marginal log-likelihood', savefile='a6_hist_corrupt')

	return

def q7():
	'''
	Loads the data and plots a color coded clustering of the conditional expectations. Rest is TODO.
	'''

	mat = loadmat('q7.mat')
	data = mat['x']
	labels = mat['y']

	print(data.shape)
	print(labels.shape)
	mean_z1, mean_z2 = get_conditional_expectation(data)

	plt.figure()
	plt.scatter(mean_z1, mean_z2, c=labels)
	plt.colorbar()
	plt.grid()
	plt.savefig('a7', bbox_inches='tight')
	plt.show()
	plt.close()

	return

def load_model(model_file):
	'''
	Loads a default Bayesian network with latent variables (in this case, a variational autoencoder)
	'''

	with open('trained_mnist_model', 'rb') as infile:
		cpts = pkl.load(infile)

	model = {}
	model['prior_z1'] = cpts[0]
	model['prior_z2'] = cpts[1]
	model['cond_likelihood'] = cpts[2]

	return model

def main():

	global disc_z1, disc_z2
	n_disc_z = 25
	disc_z1 = np.linspace(-3, 3, n_disc_z)
	disc_z2 = np.linspace(-3, 3, n_disc_z)

	global bayes_net
	bayes_net = load_model('trained_mnist_model')

	'''
	TODO: Using the above Bayesian Network model, complete the following parts.
	'''
	# q4()
	# q5()
	# q6()
	q7()

	return

if __name__== '__main__':

	main()
