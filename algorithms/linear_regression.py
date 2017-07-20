import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

class LinearRegression():
	"""
		LinearRegression class to fit y values to some features X

		::API:		
-----------------------------------------------------------------------------------------------

		.fit(X,y,learning_rate=0.01,n_iters=True,normalize=False)
		
		trains the model on the training set

		<PARAMETERS>
		X : input features of training set (numpy array, list)
		y : values to map to of training set (numpy array, list)
		n_iters :  number of iterations (integer)
		normalize : whether to normalize input or not (Boolean)
		
		<return type>
		returns None

-------------------------------------------------------------------------------------------------		

		.score(X,y)
		
		returns the score on test set

		<PARAMETERS>
		X : input features of testing set (numpy array, list)
		y : values to map to of testing set (numpy array, list)
		
		<return type>
		returns score (int)

-------------------------------------------------------------------------------------------------		

		.predict(X)

		predicts the value of new features

		<PARAMETERS>
		X : input features (numpy array, list)

		<return type>
		return predictions (numpy array,list)

-------------------------------------------------------------------------------------------------		

		.plot_costs()

		plots the cost over the iterations

-------------------------------------------------------------------------------------------------		
	"""

	def __init__(self):
		self.costs = np.array([])  # for plotting

	def add_bias(self,X):
		"""

		This is a helper function which add bias column to input features

		"""
		return np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

	def normalize(self,X):
		"""

		This is a helper function which normalizes input features

		"""
		return (X - X.mean(axis=0))/X.std(axis=0)

	def fit(self,X,y,learning_rate=0.01,n_iter=200,normalize=False):
		"""
		trains the model on the training set

		<PARAMETERS>
		X : input features of training set (numpy array, list)
		y : values to map to of training set (numpy array, list)
		n_iters :  number of iterations (integer)
		normalize : whether to normalize input or not (Boolean)
		
		<return type>
		returns None

		"""
		y = np.reshape(y,(y.shape[0],1))
		self.normalize_ = normalize
		if len(X.shape) == 1:
			X = np.reshape(X,(X.shape[0],1))
		if self.normalize_ == True:
			X = self.normalize(X)
		self.X = X
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.y = y
		self.X = self.add_bias(self.X)
		self.params = np.zeros(self.X.shape[1]).reshape(self.X.shape[1],1)
		self.length = len(self.X)
		for i in range(n_iter):
			self.predict(self.X)
			self.cost_ = self.cost()
			self.costs = np.append(self.costs,self.cost_)
			self.grad = self.gradient()
			self.gradient_descent()
		return self	

	def predict(self,X):
		"""
		predicts the value of new features

		<PARAMETERS>
		X : input features (numpy array, list)

		<return type>
		return predictions (numpy array,list)

		"""	
		self.predictions = np.dot(X,self.params)
		return self.predictions

	def cost(self):
		"""

		This is a helper function which calculates squared error cost

		"""
		squared_errors = np.square((self.predictions - self.y))
		return sum(squared_errors)/(2*self.length)

	def gradient(self):
		"""

		This is a helper function which calculates gradient

		"""
		return np.sum((self.predictions - self.y) * self.X,axis=0)/self.length

	def gradient_descent(self):
		"""

		This is a helper function updates parameters based on the gradient and minimizes the cost

		"""
		self.params = self.params - np.reshape((self.learning_rate * self.grad),(self.X.shape[1],1))

	def plot_costs(self):
		"""

		plots the cost over the iterations

		"""
		plt.plot(self.costs)
		plt.ylabel('cost')
		plt.show()

	def score(self,X_test,y_test):
		"""
		returns the score on test set

		<PARAMETERS>
		X : input features of testing set (numpy array, list)
		y : values to map to of testing set (numpy array, list)
		
		<return type>
		returns score (int)

		"""
		y_test = np.reshape(y_test,(y_test.shape[0],1))
		if self.normalize_ == True:
			X_test = self.normalize(X_test)
		X_test = self.add_bias(X_test)
		self.predict(X_test)
		u = np.square(y_test - self.predictions).sum()
		v = np.square(y_test - y_test.mean()).sum()
		self.score = 1 - u/v
		return self.score


