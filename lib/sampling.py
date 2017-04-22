from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import matplotlib.pyplot as plt


def k_means(dataset, cluster_size):
    vec = DictVectorizer()
    dataset_vectorized = vec.fit_transform(dataset).toarray()
    X = np.array(dataset_vectorized)
    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(X)
    return kmeans


def stratified_sampling(dataset, cluster_size):
    kmeans = k_means(dataset, cluster_size)
    cluster_with_data_index_list = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    sampled_cluster = {}
    decimated_data = []
    count = 0
    for cluster_data_index in cluster_with_data_index_list.keys():
        values = cluster_with_data_index_list[cluster_data_index]
        count = count + len(values)
        sampled_labels = np.random.choice(values, int(len(values) * 0.6)).flatten().tolist()
        data = []
        for i in sampled_labels:
            data.append(dataset[i])
        sampled_cluster[str(cluster_data_index)] = data
        decimated_data = decimated_data + data
    return decimated_data


def elbow(data):
    __k_range = range(1, 10)
    __k_means = [k_means(data, cluster_size) for cluster_size in __k_range]
    __mse = {}
    for X in __k_means:
        __mse[X.n_clusters] = X.inertia_
    plot_mse(__mse)
    return


# Selecting 4 as the value of k
def plot_mse(mse):
    __y = mse.values()
    plt.plot(range(1, 10), list(__y))
    plt.xlabel('Number of clusters')
    plt.ylabel('MSE')
    plt.title('K Means')
    plt.show()
    return


def format_dataset(dataset):
    output = []
    for x in dataset:
        output.append(x)
    return output


def label_categorical_data(dataset, category_meta_):
    output = []
    for pokemon in dataset:
        pokemon_row_ = []
        for feature in pokemon.keys():
            if feature in category_meta_.keys():
                feature_unique = category_meta_[feature]
                pokemon_row_.append(float(feature_unique.index(pokemon[feature])))
            else:
                if pokemon[feature].__class__.__name__ == 'str' and len(pokemon[feature]) == 0:
                    pokemon_row_.append(999999)
                else:
                    pokemon_row_.append(float(pokemon[feature]))
        output.append(pokemon_row_)
    return output

