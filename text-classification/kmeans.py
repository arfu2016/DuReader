"""
@Project   : DuReader
@Module    : kmeans.py
@Author    : Deco [deco@cubee.com]
@Created   : 6/20/18 1:23 PM
@Desc      : 
"""
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()


def sklearn_kMeans():
    model = KMeans(n_clusters=2, random_state=0)
    print('data shape:', iris.data.shape)
    model.fit(iris.data)
    print('labels:', model.labels_)
    print('distances:', model.transform(iris.data))
    return model.cluster_centers_


if __name__ == '__main__':
    centers = sklearn_kMeans()

    print('centers:', centers)
