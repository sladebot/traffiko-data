from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import matplotlib.pyplot as plt


def k_means(dataset, cluster_size):
    X = np.array(dataset)
    print("Running k means with cluster size - " + str(cluster_size))
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
        sampled_labels = np.random.choice(values, int(len(values) * 0.4)).flatten().tolist()
        data = []
        for i in sampled_labels:
            data.append(dataset[i])
        sampled_cluster[str(cluster_data_index)] = data
        decimated_data = decimated_data + data
    return decimated_data


def elbow(data):
    print("Starting k means !")
    __k_range = range(1, 10)
    print("Starting k means")
    __k_means = [k_means(data, cluster_size) for cluster_size in __k_range]
    print("Ran k means")
    __mse = {}
    for X in __k_means:
        print("Processing in for loop ..... ")
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
    for entry in dataset:
        __entry = []
        for feature in entry.keys():
            if feature in category_meta_.keys():
                feature_unique = category_meta_[feature]
                __entry.append(feature_unique.index(entry[feature]))
            else:
                if entry[feature].__class__.__name__ == 'str' and len(entry[feature]) == 0:
                    __entry.append(999999)
                else:
                    __entry.append(entry[feature])
        output.append(__entry)
        # print(len(output))
    return output

