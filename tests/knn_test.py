import numpy as np
import knn as bf
import pandas as pd

# Get data from file
df = pd.read_csv('breast-cancer-wisconsin.data', delimiter=',', encoding="utf-8")
# Replace undefined data with -9999 so it not affect accuracy 
df.replace('?', -9999, inplace=True)
# Drop unrelative data
df.drop(['id'], 1, inplace=True)
# Convert features to numeric format
df.apply(pd.to_numeric)
df['bare_nuclei'] = df['bare_nuclei'].astype('int')

# Store data in ndarrays and split it into train and testing data
y = np.array(df['class'])
X = np.array(df.drop(['class'], 1))

train_threshold = round(len(y) * 0.8)
X_train = X[:train_threshold]
X_test = X[train_threshold:]

y_train = y[:train_threshold]
y_test = y[train_threshold:]

knn = bf.KNN() #init KNN class
knn.fit(data=X_train, classes=y_train) # Fit the data
print(knn.predict(data=X_test))
accuracy = knn.score(data=X_test, classes=y_test) # Calculate accuracy
print(accuracy)