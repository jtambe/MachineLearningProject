from sklearn import datasets
iris = datasets.load_iris()

X = iris.data   #features
Y = iris.target #labels

from sklearn.cross_validation import train_test_split

# split X,Y feature and label in train and test
# test_size =.5 means, half data is used for training and remaining for test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)


from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train,Y_train)

predictions = classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print("Decision tree accuracy: " + str(accuracy_score(Y_test,predictions)))


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train,Y_train)

predictions = classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print("Kneighbour accuracy: " + str(accuracy_score(Y_test,predictions)))