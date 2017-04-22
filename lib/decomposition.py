import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances, euclidean_distances
from sklearn.manifold import MDS


def scale_data(data, component_count):
    pca = PCA()
    pca.n_components = component_count
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', pca)])
    projection = pipeline.fit_transform(data)
    return {
        "pca": pca,
        "data_projections": projection
    }


def pca_reduction(dataset, categories):
    np_dataset = np.array(dataset)
    pca_data = scale_data(np_dataset, 3)
    pca = pca_data["pca"]
    projections = pca_data["data_projections"]
    pca_squared_loadings = []
    for i in pca.components_.T:
        pca_squared_loadings.append(np.sqrt(np.sum(np.square(i))))
    np_array = np.array(pca_squared_loadings)
    feature_indices = np_array.argsort()[-3:][::-1]
    return {
        "features": list(np.array(categories).take(feature_indices)),
        "projections": projections,
        "pca_squared_loadings": pca_squared_loadings
    }


def mds_reduction(dataset, type):
    if type == 'euclidean':
        mds = MDS(n_components=2, eps=1e-6)
        similarities = euclidean_distances(dataset)
        mds.fit_transform(similarities)
        return mds.embedding_
    elif type == 'correlation':
        distance_matrix = pairwise_distances(dataset, metric=type)
        mds = MDS(n_components=2, dissimilarity='precomputed', eps=1e-6)
        mds.fit_transform(distance_matrix)
        return mds.embedding_
    else:
        raise Exception('mds type not recognized')


def draw_pca_plot(dataset):
    np_dataset = np.array(dataset)
    pca = scale_data(np_dataset, 15)["pca"]
    plt.plot(pca.explained_variance_, '--o')
    plt.axhline(y=1, color='r')
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('eigenvalue')
    plt.show()


def plot_scree(pca_squared_loadings, categories):
    y_pos = np.arange(len(categories))
    plt.bar(y_pos, pca_squared_loadings, align='center', alpha=0.5)
    plt.xticks(y_pos, categories, rotation='vertical')
    plt.ylabel('Sum Square Loadings')
    plt.title('Scree Plot')
    plt.show()