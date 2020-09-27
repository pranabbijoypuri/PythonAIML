# ================================================================================
# Machine Learning Using Scikit-Learn | 2 | Nearest Neighbors
# ================================================================================

import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Loading iris dataset
iris = datasets.load_iris()

# Split iris.data into two sets names X_train and X_test. Also, split iris.target into two sets Y_train and Y_test.

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(iris.data, iris.target,  random_state=30)
# Print the shape of X_train dataset
print(X_train.shape)

# Print the shape of X_test dataset.
print(X_test.shape)

knn_clf = KNeighborsClassifier()

knn_clf = knn_clf.fit(X_train, Y_train)

# Evaluate the model accuracy on training data set and print it's score.
print(knn_clf.score(X_train,Y_train))

# Evaluate the model accuracy on testing data set and print it's score.
print( knn_clf.score(X_test,Y_test))


# Fit multiple K nearest neighbors models on X_train data and Y_train labels with n_neighbors parameter value changing from 3 to 10

print("-----------------------")
nneighbors = 3
maxscore = 0
for x in range(3, 11):
    knn_clf = KNeighborsClassifier(n_neighbors=x)
    knn_clf = knn_clf.fit(X_train, Y_train)

    score = knn_clf.score(X_test,Y_test)
    print(score)
    if(maxscore <= score):
        nneighbors = x
        maxscore = score

print(nneighbors)