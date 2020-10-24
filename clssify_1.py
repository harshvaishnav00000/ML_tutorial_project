from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


# loading the datasets
iris = datasets.load_iris()

features = iris.data
label = iris.target

# training the classifier
clf = KNeighborsClassifier()
clf.fit(features, label)

pred = clf.predict([[1, 1, 2, 5]])
print(pred)
