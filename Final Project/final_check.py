# Random Forest Algorithm
# The code is modified from the code by Siraj Raval
# https://github.com/llSourcell/random_forests/blob/master/Random%20Forests%20.ipynb

import pickle
import numpy as np
import pandas as pd
from random import randrange, shuffle
from collections import Counter
from script import quadratic_weighted_kappa

class random_forest():

    def __init__(self, train_data, n_folds, max_depth, min_size, sample_size, n_trees, n_features, test_ratio):
        '''
        :param train_data: numpy array
        :param n_folds: int, Split a dataset into n folds, the first n-1 folds are training and the n-th fold is testing
        cross-validation process repeat n times
        :param max_depth: int, the maximum depth of the decision tree
        :param min_size: int, the minimum number of element in a branch
        :param sample_size: float, the percentage of data being selected for training
        :param n_trees: int, the number of trees per cross-validation
        :param n_features: int, the number of features
        :param test_ratio: float, the ratio of training data in original data
        '''
        self.test_ratio = test_ratio
        self.train_data = train_data.ix[:int(len(train_data) * test_ratio), 1:].values
        self.temp_test = train_data.ix[int(len(train_data) * test_ratio):, 1:].values
        self.n_folds = n_folds
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.n_features = n_features

    def train_dataset(self):
        self.scores, self.predicted, self.forest = self.evaluation_algo(self.train_data, self.random_forest, self.n_folds,
                                                                        self.max_depth, self.min_size, self.sample_size,
                                                                        self.n_trees, self.n_features)
        print("n_features:", self.n_features)
        print('Scores: ', self.scores)
        print('Mean Accuracy: ', (sum(self.scores) / float(len(self.scores))))

    def best_trained_tree(self):
        max_score = 0
        self.index = -1
        for i in range(len(self.forest)):
            score = 0
            predictions = [self.bagging_predict(self.forest[i], row) for row in self.temp_test]
            actual = [row[-1] for row in self.temp_test]
            for j in range(len(predictions)):
                if predictions[j] == actual[j]:
                    score += 1
            result = quadratic_weighted_kappa(actual, predictions)
            print("kaggle:", result)
            score = score / float(len(self.temp_test))
            if result > max_score:
                max_score = result
                self.index = i
            print(i, ") accuracy:", score)
        print("The most accurate tree:", self.index)

    def predict_test_data(self, test_data):
        max_trees = self.forest[self.index]
        predictions = [self.bagging_predict(max_trees, row) for row in test_data]
        return predictions

    def split_dataset(self, dataset, num_folds):
        res = [[] for n in range(num_folds)]
        ls = [n for n in range(num_folds)]
        for i in range(len(dataset)):
            temp = i % num_folds
            if temp == 0:
                shuffle(ls)
            res[ls[temp]].append(dataset[i])
        return res

    def accuracy_comp(self, actual, predict):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predict[i]:
                correct += 1
        return correct / float(len(actual))

    def evaluation_algo(self, dataset, algorithm, n_folds, *args):
        folds = self.split_dataset(dataset, n_folds)
        scores = []
        predicted = []
        forest = []
        for i in range(n_folds):
            print(i, " fold~~~~~~~")
            train_set = folds[:i] + folds[i + 1:]
            train_set = [row for fold in train_set for row in fold]
            test_set = folds[i]
            # test: 4000 127; train: 16000 127
            predict, trees = algorithm(train_set, test_set, *args)
            forest.append(trees)
            predicted.append(predict)
            actual = [row[-1] for row in folds[i]]
            accuracy = self.accuracy_comp(actual, predict)
            scores.append(accuracy)
        return scores, predicted, forest

    def test_split(self, index, value, dataset):
        left = []
        right = []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right


    def gini_help(self, ls, class_values):
        temp = [row[-1] for row in ls]
        num = Counter(temp)
        res = [num[n] / float(len(temp)) for n in num]
        return 1 - np.sum(np.square(res))

    def gini_index(self, groups, class_values):
        gini = 0.0
        # print("gini_index")
        size = sum([len(group) for group in groups])
        for group in groups:
            temp_gini = self.gini_help(group, class_values)
            gini += temp_gini * len(group) / float(size)
        return gini

    def get_split(self, dataset, n_features):
        class_values = list(set(row[-1] for row in dataset))  # number of classes
        b_index, b_value, b_score, b_group = 999, 999, 999, None
        features = []
        while (len(features)) < n_features:
            ind = randrange(len(dataset[0]) - 1)
            if ind not in features:
                features.append(ind)
        for index in features:
            dic = {}
            for row in dataset:
                if dic.get(row[index]) == None:
                    dic[row[index]] = 1
                    groups = self.test_split(index, row[index], dataset)  # return left, right
                    gginin = self.gini_help(dataset, class_values)
                    gini = self.gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_group = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_group}

    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        print(len(outcomes), "  ", max(set(outcomes), key=outcomes.count))
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']  # checkout left and right from root
        del node['groups']
        if (len(left) == 0) or (len(right) == 0):
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:  # check max depth
            print("too deep~~")
            if len(left) == 0 or len(right) == 0:
                node['left'] = node['right'] = self.to_terminal(left + right)
            else:
                node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # create a terminal node if the group of rows is too small,
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth + 1)

        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth + 1)

    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def subsample(self, dataset, ratio):
        sample = []
        n_sample = len(dataset) * ratio
        while len(sample) < n_sample:
            temp_ind = randrange(len(dataset))
            sample.append(dataset[temp_ind])
        return sample

    def bagging_predict(self, trees, row):
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        trees = []
        for i in range(n_trees):
            sample = self.subsample(train, sample_size)
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)
            print("tree (", i)
        predictions = [self.bagging_predict(trees, row) for row in test]
        return predictions, trees


# main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# data preparation

training = "imputed_training.csv"
train_data = pd.read_csv(training, sep=',')
print(train_data.shape)

testing = "imputed_testing.csv"
test_data = pd.read_csv(testing, sep=',')
print(test_data.shape)
ID = test_data.ix[:, :1].values
test_data = test_data.ix[:, 1:].values

# training~~~~~~~~~~~~~~~~~~
n_folds = 3
max_depth = 10
min_size = 1
sample_size = 2.0
n_trees = 15
n_features = 20
test_ratio = 0.8

# initial and fit random_forest with train_data
rf = random_forest(train_data, n_folds, max_depth, min_size, sample_size, n_trees, n_features, test_ratio)
rf.train_dataset()
rf.best_trained_tree()  # select best trees amount all folds
with open('trained_trees3.pkl', 'wb') as output:
    pickle.dump(rf, output, pickle.HIGHEST_PROTOCOL)
del rf

# testing~~~~~~~~~~~~~~~~~~
with open('trained_trees.pkl', 'rb') as input:
    rf = pickle.load(input)
predictions = rf.predict_test_data(test_data)

# write the testing result in a text file
file = open('max_output.txt', 'w')
for i in range(len(predictions)):
    temp = str(ID[i]) + ": " + str(predictions[i])
    file.write(temp)
    file.write('\n')
file.close()
