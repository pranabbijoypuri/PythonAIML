# ================================================================================
# Machine Learning Using Scikit-Learn | 6 | Clustering
# ================================================================================

import sklearn.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

# Loading iris dataset
iris = datasets.load_iris()

# Cluster iris.data set into 3 clusters using K-means with default parameters. Name the model as km_cls

km_cls = KMeans(n_clusters=3)
km_cls = km_cls.fit(iris.data)
homogeneityscore = metrics.homogeneity_score(km_cls.predict(iris.data), iris.target)
print(homogeneityscore)

# Cluster iris.data set into 3 clusters using Agglomerative clustering. Name the model as agg_cls

agg_cls = AgglomerativeClustering(n_clusters = 3)
agg_cls = agg_cls.fit(iris.data)
homogeneityscoreAggCls = metrics.homogeneity_score(agg_cls.fit_predict(iris.data), iris.target)
print(homogeneityscoreAggCls)

# Cluster iris.data set using Affinity Propagation clustering method with default parameters. Name the model as af_cls.

af_cls = AffinityPropagation(random_state=0)
af_cls = af_cls.fit(iris.data)
homogeneityscoreAfCls = metrics.homogeneity_score(agg_cls.fit_predict(iris.data), iris.target)
print(homogeneityscoreAfCls)