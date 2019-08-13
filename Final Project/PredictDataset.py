import pickle
import pandas as pd
from utils import *

testing = "testing_result.csv"
test_data = pd.read_csv(testing, sep=',')
ID = test_data.ix[:, :1].values
test_data = test_data.ix[:, 1:].values

# testing~~~~~~~~~~~~~~~~~~
with open('trained_trees.pkl', 'rb') as input:
    rf = pickle.load(input)
predictions = rf.predict_test_data(test_data)

# write the testing result in a text file
file = open('test_output.txt', 'w')
for i in range(len(predictions)):
    temp = str(ID[i]) + ": " + str(int(predictions[i]))
    file.write(temp)
    file.write('\n')
file.close()
print('Predict Testing Data Successfully')
