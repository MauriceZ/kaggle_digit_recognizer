from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
import misc


print("Loading data...")
training_data = misc.load_pickle('train')

print(training_data.shape)

lr = linear_model.LinearRegression()

X = training_data[:, 1:training_data.shape[1]]
Y = training_data[:, 0]

print("Fitting...")

lr.fit(X, Y)
