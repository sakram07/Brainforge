import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
	"""
		LinearRegression class takes a input learning rate whose
		default value is 0.01

		<<Methods>>

		.fit(X,y,learning_rate)
		here X are the features of the training set and y are the labels

		.score(X,y)
		here X are the features of test set and y are the labels

		.predict(X)
		here X are the features of test set

	"""

	def __init__(self):
		self.costs = np.array([])  # for plotting

	def fit(self,X,y,learning_rate=0.01,n_iter=200):
		# print(y.shape)
		y = np.reshape(y,(y.shape[0],1))
		# print(y.shape)
		# print(X.shape)
		if len(X.shape) == 1:
			X = np.reshape(X,(X.shape[0],1))
		self.X = np.zeros((X.shape[0],X.shape[1] + 1),dtype=np.int64)
		self.y = np.zeros((y.shape[0],y.shape[1]),dtype=np.int64)
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.y = y
		self.X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1	)
		self.params = np.zeros(self.X.shape[1]).reshape(self.X.shape[1],1)
		self.length = len(self.X)
		for i in range(n_iter):
			self.LinearRegressionCost()

	def predict(self,X):
		self.predictions = np.dot(X,self.params)
		return self.predictions

	def LinearRegressionCost(self):
		self.predict()
		# print(self.predictions.shape)
		# print(self.y.shape)
		squared_errors = np.zeros((self.predictions.shape),dtype=np.int64)
		squared_errors = np.square((self.predictions - self.y))
		self.cost = sum(squared_errors)/(2*self.length)
		# print(self.cost)
		self.costs = np.append(self.costs,self.cost)
		self.gradient()

	def gradient(self):
		self.grad = np.sum((self.predictions - self.y) * self.X,axis=0)/self.length
		self.gradient_descent()

	def gradient_descent(self):
		print(self.params.shape)
		print(np.reshape((self.learning_rate * self.grad),(self.X.shape[1],1)))
		self.params = self.params - np.reshape((self.learning_rate * self.grad),(self.X.shape[1],1))

	def plot_costs(self):
		plt.plot(self.costs)
		plt.ylabel('cost')
		plt.show()

	def score(self,X_test,y_test):	
		self.predict(X_test)

			

if __name__ == '__main__':
	lr = LinearRegression()
	df = pd.read_csv('ex1data2.txt')
	data = df.as_matrix()
	X = data[:,:2]
	y = data[:,2]
	lr.fit((X - X.mean(axis=0))/X.std(axis=0),np.reshape(y,(y.shape[0],1)),learning_rate=0.1,n_iter=200)
	lr.plot_costs()

