# ================================================================================
# Machine Learning Using Scikit-Learn | 4 | Ensemble Methods
# ================================================================================

import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
import numpy as np
from sklearn.ensemble import RandomForestRegressor

np.random.seed(100)
# Load popular Boston dataset from sklearn.datasets module and assign it to variable boston.
boston = datasets.load_boston()

# Split boston.data into two sets names X_train and X_test. Also, split boston.target into two sets Y_train and Y_test

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(boston.data, boston.target,  random_state=30)
# Print the shape of X_train dataset
print(X_train.shape)

# Print the shape of X_test dataset.
print(X_test.shape)

# Build a Random Forest Regressor model from X_train set and Y_train labels, with default parameters. Name the model as rf_reg.
rf_reg = RandomForestRegressor()
rf_reg = rf_reg.fit(X_train, Y_train)

print(rf_reg.score(X_train, Y_train))
print(rf_reg.score(X_test, Y_test))

predicted = rf_reg.predict(X_test[:2])
print(predicted)

maxdepth = 3
mnestimators = 50
maxscore = 0
for x in range(3, 6):
    for y in [50, 100, 200]:
        rf_reg = RandomForestRegressor(max_depth=x,n_estimators=y)
        rf_reg = rf_reg.fit(X_train, Y_train)
        score = rf_reg.score(X_test, Y_test)
        if(maxscore < score):
            maxdepth = x
            mnestimators = y
            maxscore = score
res = (maxdepth,mnestimators )
print(res)