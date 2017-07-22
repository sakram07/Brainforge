# This is Linear SVM

# array handling with numpy
import numpy as np

# matplotlib lib is optional
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

class Linear_SVM(object):

    def __init__(self, visualization=True):
        # visualization is set to True by default, define it as False
        # to disable it
        self.visualization = visualization

        # the positive class is red
        # the negative is blue
        self.colors = {1:'r', -1:'b'}

        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        self.data = data

        # opt_dict is the magnitude of W vector {|| W ||}
        # and it is essentially dictionary holding [w, b] as key: value
        opt_dict = {}

        # iterate through transforms and get the product of each value
        # with W passing through all possible values
        transforms = [[1, 1],
                      [-1, 1],
                      [-1,-1],
                      [1, -1]]

        # Get the maximum and minimum possible values
        # for optimizing
        all_data = []

        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None


        # svm = yi(xi.w+b) = 1
        # Defining the step sizes for the convex optimization
        # starting from bigger (0.1) to smaller (0.001) steps
        # so we don't take too much small steps
        # TODO make step_sizes dynamic
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]  # point of expense

        # point of expense: high
        # b does not need as smaller steps to be efficient
        # the smaller it gets the bigger time and cost
        b_range_multiple = 5 # very expensive

        # no need of bigger steps with b
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        
        # start the stepping process
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # Set optimized as false by default
            # until there is no more steps to be taken
            # in the convex
            optimized = False

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value *
                                         b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        # yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break # TODO
                                    # print(yi*(np.dot(w_t, xi) + b))

                        if found_option:
                            # the magnitude of the vector
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('Optimizing a step') # TODO remove after test is done
                else:
                    w = w - step
            # sorting the list of magnitudes in opt_dict
            norms = sorted([n for n in opt_dict])
            # get the lowest magnitude in opt_dict
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        # classification is the sign(Xi.W + b)
        # b = bias
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
       
        # this step is optional and its for visualizing the classification
        # checks if classification is not empty and
        # there is self.visualization
        if classification is not 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200,
                            marker='*', c=self.colors[classification])
        # either way, it returns the classification
        return classification

    def visualize(self, data_dict):
        # this functions is all about visualizing
        # not vital for calculating the actual svm algorithm
        # can be removed/not used if plotting is not needed
        # or you want to use your own function and/or plotting method
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
          for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            # embeded function
            # hyperplane is x.w+b
            # v = x.w+b
            # positive_sv = 1
            # negative_sv = -1
            # decision_boundry = 0
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9,
                     self.max_feature_value * 1.1)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x + b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x + b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x + b) = 0
        # decision boundry support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()