# download the iris dataset
import os
cmd = 'curl -Lo iris.csv http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
os.system(cmd)

# read iris dataset to csv
from pandas import read_csv
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
dataset = read_csv('iris.csv', names=names)

# split the iris dataset to training set & validation set
from sklearn.model_selection import train_test_split
array = dataset.values
X = array[:,0:3]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.25, random_state=1)

# use the KNeighborsClassifier algorithm to train
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)

# validate the model using the validation set
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# save the trained model as iris_knn_model.pk
import dill as pickle
with open('iris_knn_model.pk', 'wb') as file:
	pickle.dump(model, file)
