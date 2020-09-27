# ================================================================================
# Machine Learning Using Scikit-Learn | 5 | SVM
# ================================================================================
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
from sklearn.svm import SVC
import sklearn.preprocessing as preprocessing


# Load popular digits dataset from sklearn.datasets module and assign it to variable digits.
digits = datasets.load_digits()

# Split digits.data into two sets names X_train and X_test. Also, split digits.target into two sets Y_train and Y_test..

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(digits.data, digits.target, stratify=digits.target, random_state=30)
# Print the shape of X_train dataset
print(X_train.shape)

# Print the shape of X_test dataset.
print(X_test.shape)

# Build an SVM classifier from X_train set and Y_train labels, with default parameters. Name the model as svm_clf
svm_clf = SVC(gamma="auto")
svm_clf = svm_clf.fit(X_train, Y_train)


print(svm_clf.score(X_test, Y_test))

# Perform Standardization of digits.data and store the transformed data in variable digits_standardized.

standardizer = preprocessing.StandardScaler()
standardizer = standardizer.fit(digits.data)
digits_standardized = standardizer.transform(digits.data)

# Split digits.data into two sets names X_train and X_test. Also, split digits.target into two sets Y_train and Y_test..

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(digits_standardized, digits.target, stratify= digits.target, random_state=30)

# Build an SVM classifier from X_train set and Y_train labels, with default parameters. Name the model as svm_clf
svm_clf2 = SVC(gamma="auto")
svm_clf2 = svm_clf.fit(X_train, Y_train)


print(svm_clf2.score(X_test, Y_test))
