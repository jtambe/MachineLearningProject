# print("hello world")
# import numpy as np
#
# arr = np.array([8,3,4,4,4])
# print(arr)

from sklearn import tree

#Features = weight, surface
# smooth = 0
# bumpy = 1
features = [[140,0], [130,0], [150, 1], [170,1]]
# labels are output categories
# apple = 0
# orange = 1
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features,labels)

print(classifier.predict([150,1]))



