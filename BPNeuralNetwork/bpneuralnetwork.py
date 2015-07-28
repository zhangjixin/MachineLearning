import random
import sys

import numpy as np

def CrossEntropyCost(a, y):
	return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

def CrossEntropyCostDerivation(a, y):
	return (a - y)

def Sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

def SigmoidDerivative(z):
	return Sigmoid(z) * (1 - Sigmoid(z))
	
sigmoid_vec = np.vectorize(Sigmoid)
sigmoid_derivative_vec = np.vectorize(SigmoidDerivative)

class BPNeuralNetwork() :
	def __init__(self, layers) :
		self.layers_num = len(layers)
		self.layer_sizes = layers
		self.weight_initializer()
	def weight_initializer(self):
		self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
		self.weights = [np.random.randn(y, x)/np.sqrt(x) 
						for x, y in zip(self.layer_sizes[:-1],self.layer_sizes[1:])]
	def RunBPNeuralNetwork(self, in_data):
		ret = in_data
		for w, b in zip(self.weights, self.biases):
			ret = sigmoid_vec(np.dot(w, ret) + b)
		return ret
	def TrainModel(self, train_data, epochs, mini_batch_size, eta, lmbda = 0):
		n = len(train_data)
		for j in range(epochs) :
			random.shuffle(train_data)
			mini_batches = [train_data[k:k+mini_batch_size]
							for k in range(0, n, mini_batch_size)]
			for batch in mini_batches :
				self.update(batch, eta, lmbda, n)
	def update(self, data, eta, lmbda, n):
		m = len(data)
		iter_b = [np.zeros(b.shape) for b in self.biases]
		iter_w = [np.zeros(w.shape) for w in self.weights]		
		for x, y in data :
			delta_b, delta_w = self.backpropagation(x, y)
			iter_b = [nb + dnb for nb, dnb in zip(iter_b, delta_b)]
			iter_w = [nw + dnw for nw, dnw in zip(iter_w, delta_w)]
		
		self.weights = [(1-eta*(lmbda/n))*w - (eta/m)*nw for w, nw in zip(self.weights, iter_w)]
		self.biases = [b - (eta/m)*nb for b, nb in zip(self.biases, iter_b)]
		
	def backpropagation(self, x, y):
		afun = x
		afuns = [x]
		zfuns = []
		for b, w in zip(self.biases, self.weights):
			zfun = np.dot(w, afun) + b
			zfuns.append(zfun)
			afun = sigmoid_vec(zfun)
			afuns.append(afun)
		delta_b = [np.zeros(b.shape) for b in self.biases]
		delta_w = [np.zeros(w.shape) for w in self.weights]
		delta = CrossEntropyCostDerivation(afuns[-1], y)
		delta_b[-1] = delta
		delta_w[-1] = np.dot(delta, afuns[-2].transpose())
		
		for layer in range(2, self.layers_num):
			zfun = zfuns[-layer]
			derivative = sigmoid_derivative_vec(zfun)
			delta = np.dot(self.weights[-layer + 1].transpose(), delta) * derivative
			delta_b[-layer] = delta
			delta_w[-layer] = np.dot(delta, afuns[-layer-1].transpose())
		return (delta_b, delta_w)

		