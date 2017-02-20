
import random
from scipy.spatial import distance

# a = point from training data
# b = point from testing data
def EuclideanDistance(a,b):
    return distance.euclidean(a,b)

class SampleKNN:
    def fit(self, X_train, Y_train):
        self.Xtrain = X_train
        self.Ytrain = Y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = EuclideanDistance(row, self.Xtrain[0])
        best_index = 0
        for i in range(1, len(self.Xtrain)):
            dist = EuclideanDistance(row,self.Xtrain[i])
            if(dist < best_dist):
                best_dist = dist
                best_index = i
        return self.Ytrain[best_index]



from sklearn import datasets
iris = datasets.load_iris()

X = iris.data   #features
Y = iris.target #labels

from sklearn.cross_validation import train_test_split
# split X,Y feature and label in train and test
# test_size =.5 means, half data is used for training and remaining for test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)


#from sklearn.neighbors import KNeighborsClassifier
classifier = SampleKNN()
classifier.fit(X_train,Y_train)

predictions = classifier.predict(X_test)
#print(predictions)

from sklearn.metrics import accuracy_score
print("Kneighbour accuracy: " + str(accuracy_score(Y_test,predictions)))