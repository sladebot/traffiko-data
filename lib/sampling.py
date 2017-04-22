from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import matplotlib.pyplot as plt


def k_means(dataset, cluster_size):
    X = np.array(dataset)
    print("Running k means with cluster size - " + str(cluster_size))
    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(X)
    return kmeans


def stratified_sampling(dataset, cluster_size, categoric_meta_):
    labelled_dataset = np.array(label_categorical_data(dataset, categoric_meta_))
    kmeans = k_means(labelled_dataset, cluster_size)
    cluster_with_data_index_list = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    decimated_data = []
    count = 0
    for cluster_data_index in cluster_with_data_index_list.keys():
        values = cluster_with_data_index_list[cluster_data_index]
        count = count + len(values)
        sampled_labels = np.random.choice(values, int(len(values) * 0.4)).flatten().tolist()
        data = []
        for i in sampled_labels:
            # change dataset labels to real data from categoric meta
            # for feature_name in categoric_meta_.keys():
            #     _feature_index = categoric_meta_[feature_name]['feature_index']
            #     label_value = int(dataset[i][_feature_index])
            #     dataset[i][_feature_index] = categoric_meta_[feature_name]['uniques'][label_value]
            data.append(dataset[i])
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
                feature_unique = category_meta_[feature]['uniques']
                __entry.append(feature_unique.index(entry[feature]))
            else:
                if entry[feature].__class__.__name__ == 'str' and len(entry[feature]) == 0:
                    __entry.append(999999)
                else:
                    __entry.append(entry[feature])
        output.append(__entry)
        print(len(output))
    return output

