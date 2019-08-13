# Random Forest Algorithm
# data preparation
import math
from utils import *
import pandas as pd

training = "training_result.csv"
train_data = pd.read_csv(training, sep=',')

testing = "testing_result.csv"
test_data = pd.read_csv(testing, sep=',')
ID = test_data.ix[:, :1].values
test_data = test_data.ix[:, 1:].values

# training~~~~~~~~~~~~~~~~~~
n_folds = 3
max_depth = 10
min_size = 1
sample_size = 1.0
n_trees = 5
n_features = 11
test_ratio = 0.8

# initial and fit random_forest with train_data
rf = random_forest(train_data, n_folds, max_depth, min_size, sample_size, n_trees, n_features, test_ratio)
rf.best_trained_tree()  # select best trees amount all folds

# testing~~~~~~~~~~~~~~~~~~
predictions = rf.predict_test_data(test_data)

# write the testing result in a text file
file = open('max_output.txt', 'w')
for i in range(len(predictions)):
    temp = str(ID[i]) + ": " + str(int(predictions[i]))
    file.write(temp)
    file.write('\n')
file.close()
