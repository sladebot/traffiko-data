from flask import Flask
from flask_script import Manager, Server
from pymongo import MongoClient
import numpy as np
from bson.json_util import dumps


from lib.sampling import stratified_sampling, elbow, label_categorical_data, format_dataset
from lib.decomposition import pca_reduction, mds_reduction, draw_pca_plot, plot_scree


app = Flask(__name__, template_folder='templates')
app = Flask(__name__, static_url_path='/static')
# app.config.from_object('app.settings')

manager = Manager(app)
server = Server(host="0.0.0.0", port=5000)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'Accidents'
RAW_COLLECTION = 'Collision'

FIELDS = {"Attack": True}
CATEGORICAL_ATTRIBUTES = ["DATE", "TIME", "BOROUGH", "ZIP CODE", "ON STREET NAME",
                          "CROSS STREET NAME", "OFF STREET NAME", "CONTRIBUTING FACTOR VEHICLE 1",
                          "CONTRIBUTING FACTOR VEHICLE 2", "CONTRIBUTING FACTOR VEHICLE 3", "CONTRIBUTING FACTOR VEHICLE 4",
                          "CONTRIBUTING FACTOR VEHICLE 5", "VEHICLE TYPE CODE 1", "VEHICLE TYPE CODE 2", "VEHICLE TYPE CODE 3", "VEHICLE TYPE CODE 4",
                          "VEHICLE TYPE CODE 5"]

# This has been computed from by optimizing K-Means with elbow method.
ELBOW_K_SIZE = 3

connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
db = connection[DBS_NAME]
data_collection = db[RAW_COLLECTION]


record = data_collection.find_one({}, {"_id": False})
categories = list(record.keys())

categoric_meta_ = {}
for category in CATEGORICAL_ATTRIBUTES:
    category_uniques = data_collection.distinct(category)
    categoric_meta_[category] = {
        'uniques': category_uniques,
        'feature_index': categories.index(category)
    }


def response_from_pymongo(cursor_dataset):
    response = []
    for data in cursor_dataset:
        response.append(data)
    return response


def insert_pca_to_mongo(top_three_attributes, type):
    __filter = {"_id": False}
    for attribute in top_three_attributes:
        __filter[attribute] = True
    read_collection = db[type + "_sampled"]
    data = read_collection.find({}, __filter)
    reduced_data = []
    for row in data:
        reduced_data.append(row)
    write_collection = db[type + "_pca_reduced_data"]
    write_collection.drop()
    write_collection.insert(reduced_data)


def insert_mds_to_mongo(data, filename, type):
    collection = db[type + "_" + filename + "_data"]
    collection.drop()
    formatted_dataset = []
    for row in data:
        formatted_dataset.append({
            "x": row[0],
            "y": row[1]
        })
    collection.insert(formatted_dataset)


def insert_stratified_samples_to_mongo(data):
    db["stratified_sampled"].drop()
    for document in data:
        try:
            db["stratified_sampled"].insert(document)
        except:
            pass
            # print("Found duplicate records - ")


@manager.command
def plot_k_means_elbow():
    print("K-means optimization using elbow method")
    dataset = data_collection.find({}, {"_id": False})
    print("Read data formatting it.")
    dataset_formatted = format_dataset(dataset)
    print("Finished formatting data")
    labelled_dataset = np.array(label_categorical_data(dataset_formatted, categoric_meta_))
    print("Labelled Data done ! ")
    elbow(labelled_dataset)


@manager.command
def do_stratified_sampling():
    dataset = data_collection.find({}, {"_id": False})
    dataset_formatted = format_dataset(dataset)
    sampled_cluster = stratified_sampling(dataset_formatted, ELBOW_K_SIZE, categoric_meta_)
    insert_stratified_samples_to_mongo(sampled_cluster)


@manager.command
def do_pca():
    types = ["random", "stratified"]
    for type in types:
        dataset = db[type + "_sampled"].find({}, {"_id": False})
        labelled_dataset = label_categorical_data(dataset, categoric_meta_)
        pca_data = pca_reduction(labelled_dataset, categories)
        top_three_features = pca_data["features"]
        insert_pca_to_mongo(top_three_features, type)


@manager.command
def do_mds():
    mds_types = ["euclidean", "correlation"]
    types = ["random", "stratified"]

    for type in types:
        dataset = db[type + "_pca_reduced_data"].find({}, {"_id": False})
        labelled_dataset = label_categorical_data(dataset, categoric_meta_)
        for mds_type in mds_types:
            mds_scaled_dataset = mds_reduction(labelled_dataset, mds_type)
            insert_mds_to_mongo(list(mds_scaled_dataset), mds_type, type)


@manager.command
def plot_pca():
    dataset = db["stratified_sampled"].find({}, {"_id": False})
    labelled_dataset = label_categorical_data(dataset, categoric_meta_)
    draw_pca_plot(labelled_dataset)


@manager.command
def stratified_plot_scree_plot():
    dataset = db["stratified_sampled"].find({}, {"_id": False})
    labelled_dataset = label_categorical_data(dataset, categoric_meta_)
    pca_data = pca_reduction(labelled_dataset, categories)
    pca_squared_loadings = pca_data["pca_squared_loadings"]
    plot_scree(pca_squared_loadings, categories)

@manager.command
def random_plot_scree_plot():
    dataset = db["random_sampled"].find({}, {"_id": False})
    labelled_dataset = label_categorical_data(dataset, categoric_meta_)
    pca_data = pca_reduction(labelled_dataset, categories)
    pca_squared_loadings = pca_data["pca_squared_loadings"]
    plot_scree(pca_squared_loadings, categories)