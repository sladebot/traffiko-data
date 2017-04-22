from flask import Flask, request, render_template
from flask_script import Manager, Server
from flask_assets import Environment, Bundle
from flask_bower import Bower
from pymongo import MongoClient
import numpy as np
from bson.json_util import dumps
import json


from lib.sampling import stratified_sampling, elbow, label_categorical_data, format_dataset
from lib.decomposition import pca_reduction, mds_reduction, draw_pca_plot, plot_scree



app = Flask(__name__, template_folder='templates')
app = Flask(__name__, static_url_path='/static')
# app.config.from_object('app.settings')

manager = Manager(app)
server = Server(host="0.0.0.0", port=5000)

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DBS_NAME = 'vs-asg-02'
RAW_COLLECTION = 'pokemons'

FIELDS = {"Attack": True}
CATEGORICAL_ATTRIBUTES = ["Name", "Type_1", "Type_2",
                          "isLegendary", "Color", "hasGender",
                          "Egg_Group_1", "Egg_Group_2", "hasMegaEvolution",
                          "Body_Style", "Generation", "Catch_Rate"]

# This has been computed from by optimizing K-Means with elbow method.
ELBOW_K_SIZE = 7

connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
db = connection[DBS_NAME]
data_collection = db[RAW_COLLECTION]

categoric_meta_ = {}
for category in CATEGORICAL_ATTRIBUTES:
    category_uniques = data_collection.distinct(category)
    categoric_meta_[category] = category_uniques

record = data_collection.find_one({}, {"_id": False})
categories = list(record.keys())


def response_from_pymongo(cursor_dataset):
    response = []
    for data in cursor_dataset:
        response.append(data)
    return response


def insert_random_sampled_to_mongo(data):
    collection = db["random_sampled"]
    collection.drop()
    for document in data:
        try:
            collection.insert(document)
        except:
            pass
            # print("Found duplicate records - ")


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
    dataset_formatted = format_dataset(dataset)
    elbow(dataset_formatted)


@manager.command
def do_random_sampling():
    dataset = data_collection.find({}, {"_id": False})
    formatted_data = format_dataset(dataset)
    random_sampled_indices = np.random.randint(len(formatted_data), size=int(len(formatted_data)*0.6))
    sampled_data = np.take(np.array(formatted_data), random_sampled_indices)
    insert_random_sampled_to_mongo(sampled_data.tolist())


@manager.command
def do_stratified_sampling():
    dataset = data_collection.find({}, {"_id": False})
    dataset_formatted = format_dataset(dataset)
    sampled_cluster = stratified_sampling(dataset_formatted, ELBOW_K_SIZE)
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