
# use a sample dataset from sklearn
from sklearn.datasets import load_iris

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









