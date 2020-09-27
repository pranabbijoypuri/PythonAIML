# ================================================================================
# Machine Learning Using Scikit-Learn | 3 | Decision Trees
# ================================================================================

import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(100)
# Load popular Boston dataset from sklearn.datasets module and assign it to variable boston.
boston = datasets.load_boston()

# Split boston.data into two sets names X_train and X_test. Also, split boston.target into two sets Y_train and Y_test

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(boston.data, boston.target,  random_state=30)
# Print the shape of X_train dataset
print(X_train.shape)

# Print the shape of X_test dataset.
print(X_test.shape)

# Build a Decision tree Regressor model from X_train set and Y_train labels, with default parameters. Name the model as dt_reg

dt_Regressor = DecisionTreeRegressor()

dt_reg = dt_Regressor.fit(X_train, Y_train)

print(dt_reg.score(X_train,Y_train))

print(dt_reg.score(X_test,Y_test))

predicted = dt_reg.predict(X_test[:2])
print(predicted)

# Get the max depth

maxdepth = 2
maxscore = 0
for x in range(2, 6):
    dt_Regressor = DecisionTreeRegressor(max_depth=x)
    dt_reg = dt_Regressor.fit(X_train, Y_train)
    score = dt_reg.score(X_test, Y_test)
    if(maxscore < score):
        maxdepth = x
        maxscore = score

print(maxdepth)