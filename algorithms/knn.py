import numpy as np

class KNN(object):

	# Setup how much nearest neighbors we want to count, 5 by default
	def __init__(self, n_neighbors=5):
		self.k = n_neighbors

	# Calculate the Euclidean Distance between 2 points - x, y
	def euclidean_distance(self, x, y):
		# Square root of the sum of squared distances between each numbers(features) in array(point)
		return np.sqrt(np.sum((x-y)**2))

	# Takes point as parameter
	# Returns classes of neighbors to this point
	def get_neighbors(self, x):
		distances = []
		# Calculate distance between given point and each point in known data
		for i in range(len(self.model_data)):
			dist = self.euclidean_distance(x, self.model_data[i])
			distances.append((self.model_data_classes[i], dist))

		#Sort the array of distances and get first K elements as nearest neighbors
		distances.sort(key=lambda k: k[1])
		neighbors = []
		for x in range(self.k):
			neighbors.append(distances[x][0])
		return neighbors

	# Takes neighbors as parameter
	# Returns predicted class
	def get_class(self, neighbors):
		class_votes = {}
		# Count neighbors by belonging to each class in neighbors list
		for x in neighbors:
			if x in class_votes:
				class_votes[x] += 1
			else:
				class_votes[x] = 1
		# returns class that has most votes in class_votes dict
		return sorted(class_votes, key=class_votes.__getitem__, reverse=True)[0]

	# Calculate accuracy by given new data and classes
	# Returns percentage of success
	def score(self, data, classes):
		correct = 0
		predictions = self.predict(data)
		for x in range(len(predictions)):
			if predictions[x] == classes[x]:
				correct += 1
		return (correct/float(len(data)))

	# Just store known points in class variables
	def fit(self, data, classes):
		self.model_data = data
		self.model_data_classes = classes

	# Takes non-classified data as parameter
	# Returns predictions
	def predict(self, data):
		predictions = []
		try:
			# for each point in data get nearest neighbors and calculate to which class it belongs to
			for point in data:
				predictions.append(self.get_class(self.get_neighbors(point)))
		except NameError:
			return None # returns None if fit() function was not called before
		return predictions