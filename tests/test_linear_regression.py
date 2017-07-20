# importing os and sys to import linear regression from previous directory
import os
import sys
sys.path.append(os.path.join('..', 'algorithms'))

# import linear regression class
from linear_regression import LinearRegression
import pandas as pd


# create Linear regression object
lr = LinearRegression()

# read csv data from text file
df = pd.read_csv('dummy_data_linear_regression.txt')

# convert pandas dataframe to matrix
data = df.as_matrix()

# pick out feature values from the matrix
X = data[:,:2]

# pick out regression values
y = data[:,2]

# fit to linear regression model with different parameters (make normalize = true if the dataset has too much variance)
lr.fit(X[:30],y[:30],learning_rate=0.1,n_iter=200,normalize=True)

# plot cost in different iterations to tweak the learning rate
lr.plot_costs()

# check the R^2 score used in sklearn models
print(lr.score(X[31:46],y[31:46]))