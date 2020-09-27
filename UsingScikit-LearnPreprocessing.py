#   =================================================================================
#   Machine Learning Using Scikit-Learn | 1 | Preprocessing
#   =================================================================================

import sklearn.preprocessing as preprocessing
import sklearn.datasets as datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

# Loading iris dataset
iris = datasets.load_iris()
# print(iris)

# Perform Normalization on iris.data with l2 norm and save the transformed data in variable iris_normalized.
normalizer = preprocessing.Normalizer(norm='l2').fit(iris.data)
iris_normalized = normalizer.transform(iris.data)

# Print the mean of every column using the below command
print(iris_normalized.mean(axis=0))


# Convert the categorical integer list iris.target into three binary attribute representation and store the result in variable iris_target_onehot

onehotencoder = OneHotEncoder(categories='auto')
iris_target_onehot = onehotencoder.fit_transform(iris.target.reshape(-1,1))

# Execute the following print statement
print(iris_target_onehot.toarray()[[0, 50, 100]])


# Set first 50 row values of iris.data to Null values. Use numpy.nan
df = pd.DataFrame(iris.data)
# print(df)
df.iloc[:50, :] = np.nan

# Perform Imputation on 'iris.data' and save the transformed data in variable 'iris_imputed'.

imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean')

imputer = imputer.fit(df)
iris_imputed = imputer.transform(df)

print(iris_imputed.mean(axis=0))
