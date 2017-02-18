
# use a sample dataset from sklearn
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np


iris = load_iris()

print(iris.feature_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print(iris.target_names)
#['setosa' 'versicolor' 'virginica']


#print(iris.data)
#[ [list of features] ]

#print(iris.target)
#[list of targets as numbers]


# printing out entire data and target
# for i in range(len(iris.target)):
#     print("Number: " + str(i) + " Data: " + str(iris.data[i]) + " Target: " + str(iris.target[i]))


# setting up indices for testing data in loaded dataset
testDataIndices = [0,50,100]

# creating training data by removing data at those indices
train_target = np.delete(iris.target, testDataIndices)
# np.delete for list of lists
#http://stackoverflow.com/questions/40348269/numpy-delete-list-element-from-list-of-lists
train_data = np.delete(iris.data, testDataIndices, axis=0) # delete elements at indices at axis 0

#print(train_data)
# total_lists = (len(train_data)//4)
# np.reshape(train_data, (4,-1) )

#print(train_data)
#print(train_target)
# creating testing data by using the indices above
test_target = iris.target[testDataIndices]
test_data = iris.data[testDataIndices]


classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)


print(test_target)
print(classifier.predict(test_data))


# sudo python3 -m pip install ipython
# sudo python3 -m pip install pydot
# sudo python3 -m pip install pydotplus
# sudo apt-get install graphviz
import pydotplus
from IPython.display import Image
dot_data = tree.export_graphviz(classifier, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")



















