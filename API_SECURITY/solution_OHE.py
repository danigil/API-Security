# Imports, settings and first dataset view
import copy
from typing import Sized

import pandas
import pandas as pd
import seaborn as sns
import numpy as np
import json
import pickle
from datetime import date, datetime
import pprint

import logging

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

now = datetime.now()

current_time = now.strftime("%H-%M-%S_%d-%m-%Y")

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest, BaggingClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

from sklearn.tree import DecisionTreeClassifier

df = None
features_list = None

X = None
y = None

X_eval = None

def ret_file_name(clf_name):
    filename = f'dataset{str(dataset_number)}_{clf_name}_{current_time}'  # _{date.today()

    return filename


def ret_pretty_list_str(list):
    return pprint.pformat(list)


def global_settings():
    global df
    global dataset_number
    global test_type

    global COLUMNS_ALL
    # Set pandas to show all columns when you print a dataframe
    pd.set_option('display.max_columns', None)

    # Global setting here you choose the dataset number and classification type for the model
    dataset_number = dataset_number  # Options are [1, 2, 3, 4]
    test_type = test_type  # Options are ['label', 'attack_type']

    # Read the json and read it to a pandas dataframe object, you can change these settings
    with open(f'./datasets/dataset_{str(dataset_number)}_train.json') as file:
        raw_ds = json.load(file)
    df = pd.json_normalize(raw_ds, max_level=2)
    COLUMNS_ALL = set(df.loc[:, ~df.columns.isin(['request.Attack_Tag'])].columns)


def label_arrangements():
    global df
    # Fill the black attack tag lines with "Benign" string
    df['request.Attack_Tag'] = df['request.Attack_Tag'].fillna('Benign')
    df['attack_type'] = df['request.Attack_Tag']

    # This function will be used in the lambda below to iterate over the label columns
    # You can use this snippet to run your own lambda on any data with the apply() method
    def categorize(row):
        if row['request.Attack_Tag'] == 'Benign':
            return 'Benign'
        return 'Malware'

    df['label'] = df.apply(lambda row: categorize(row), axis=1)

    # After finishing the arrangements we delete the irrelevant column
    df.drop('request.Attack_Tag', axis=1, inplace=True)


def preprocessing():
    global df
    # Remove all NAN columns or replace with desired string
    # This loop iterates over all of the column names which are all NaN
    for column in df.loc[:, df.isna().any()].columns.tolist():
        # df.drop(column, axis=1, inplace=True)
        df[column] = df[column].fillna('None')

    # If you want to detect columns that may have only some NaN values use this:
    # df.loc[:, df.isna().any()].tolist()


# This is our main preprocessing function that will iterate over all of the chosen
# columns and run some feature extraction models

def vectorize_df(df):
    for column in COLUMNS_TO_REMOVE:
    # if column in df.columns.to_list():
        df.drop(column, axis=1, inplace=True)

    # X_columns = copy.deepcopy(df.columns.tolist())
    # X_columns.remove('label')
    # X_columns.remove('attack_type')
    # df = pd.DataFrame(oe.fit_transform(df.loc[:, ~df.columns.isin(['label', 'attack_type'])]),columns=X_columns)


    # for column in df.columns:
    #     if column not in ('label','attack_type'):
    #         df[column] = oe.fit_transform(df[column])

    return df


def label_encoding_and_feature_extraction():
    global df
    global features_list

    # list = ret_pretty_list_str(df.columns.to_list())
    # print(list)
    df = vectorize_df(df)
    # print(list)
    # list = ret_pretty_list_str(df.columns.to_list())

    features_list = df.columns.to_list()
    if 'label' in features_list:
        features_list.remove('label')
    if 'attack_type' in features_list:
        features_list.remove('attack_type')

def load_evaluation_data():
    global X_eval
    global features_list
    # Now it's your turn, use the model you have just created :)

    # Read the valuation json, preprocess it and run your model
    with open(f'./datasets/dataset_{str(dataset_number)}_val.json') as file:
        raw_ds = json.load(file)
    test_df = pd.json_normalize(raw_ds, max_level=2)

    for column in test_df.loc[:, test_df.isna().any()].columns.tolist():
        # df.drop(column, axis=1, inplace=True)
        test_df[column] = test_df[column].fillna('None')

    # Preprocess the validation dataset, remember that here you don't have the labels
    test_df = vectorize_df(test_df)

    # Predict with your model
    X_eval = test_df[features_list]

le = LabelEncoder()
ohe = OneHotEncoder()
def train_test_split_ret():

    global df
    global features_list
    global test_type

    global X
    global y

    global X_eval
    load_evaluation_data()

    global min_categories



    dataset = df.append(X_eval)


    min_categories = [(column, len(dataset[column].value_counts())) for column in df[features_list].columns]
    print(min_categories)

    # for column in columns_to_hot_encode:
    #     ohe.fit(dataset[column])

    threshold = 200
    COLUMNS_TO_ONE_HOT_ENCODE = set(map(lambda tup: tup[0],filter(lambda tup: threshold >= tup[1] > 1, [(column, len(df[column].value_counts())) for column in df.loc[:, ~df.columns.isin(('label', 'attack_type'))].columns])))
    # print(COLUMNS_TO_ONE_HOT_ENCODE)

    COLUMNS_TO_TF_IDF_ENCODE = COLUMNS_ALL - COLUMNS_TO_ONE_HOT_ENCODE - COLUMNS_TO_REMOVE

    # dataset_one_hot_columns_df = dataset[COLUMNS_TO_ONE_HOT_ENCODE]
    # dataset_tfidf_columns_df = dataset[COLUMNS_TO_TF_IDF_ENCODE]

    print(ret_pretty_list_str(list(COLUMNS_TO_ONE_HOT_ENCODE)))

    print(ret_pretty_list_str(list(COLUMNS_TO_TF_IDF_ENCODE)))

    ct = ColumnTransformer(
        transformers=[
            ("OrdinalEncoder", OrdinalEncoder(), list(COLUMNS_TO_TF_IDF_ENCODE)),
            ("OHE", OneHotEncoder(sparse_output=False), list(COLUMNS_TO_ONE_HOT_ENCODE)),
        ],
    remainder="drop")


    ct.fit(dataset)
    # print((ct.transformers[0][1]).fit_transform(dataset))
    #
    # raise Exception()
    # pd.DataFrame(codes, columns=feature_names).astype(int)

    X_eval = X_eval[features_list]
    X_eval = ct.transform(X_eval)

    # Data train and test split preparations. Here we will insert our feature list and label list.
    # Afterwards the data will be trained and fitted on the amazing XGBoost model
    # X_Train and y_Train will be used for training
    # X_test and y_test.T will be used for over fitting checking and overall score testing

    # We convert the feature list to a numpy array, this is required for the model fitting

    # for column in df[features_list].columns:

    # for count, label in np.unique(X_eval, return_counts=True):
    #     pass
    #
    # print(min_categories)


    X = df[features_list]

    X = ct.transform(X=X)
    # print(f'train_test after transform: {X}')
    # print(f'categories: {ohe.categories_}')
    # raise Exception()

    # This column is the desired prediction we will train our model on
    y = np.stack(df[test_type])

    if test_type == 'label':
        attack_types = ['Benign','Malware']
    else:
        attack_types = ['Benign',
                        'Cookie Injection',
                        'Directory Traversal',
                        'LOG4J',
                        'Log Forging',
                        'RCE',
                        'SQL Injection',
                        'XSS'
                        ]

    le.fit(attack_types)

    # print(f'before: {np.unique(y, return_counts=True)}')
    y = le.transform(y)
    # print(f'after: {np.unique(y, return_counts=True)}')

    # We split the dataset to train and test according to the required ration
    # Do not change the test_size -> you can change anything else
    return train_test_split(X, y, test_size=0.1765,random_state=42, stratify=y)

class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)

def model(X_train, y_train):
    # We choose our model of choice and set it's hyper parameters you can change anything

    # print(ret_pretty_list_str(features_list))
    # clf = MLPClassifier(hidden_layer_sizes=(18,13,8))
    # clf = CategoricalNB(min_categories=min_categories)

    # clf = LogisticRegression(solver="newton-cholesky")
    clf = HistGradientBoostingClassifier(warm_start=True)
    pipe = Pipeline([(str(type(clf)).split(".")[-1][:-2], clf)])
    param_grid = {
        "LogisticRegression__C":[1,2,5,10,15,25,50,100,300]#[0.1, 0.2, 0.3, 0.4 ,0.5, 0.6 ,0.7, 0.8, 0.9]
    }

    # search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=3)
    # search.fit(X_train, y_train)
    # print("Best parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)
    #
    # raise Exception()

    X_train = X_train


    # Train Model
    pipe.fit(X_train, y_train)
    clf = pipe
    return clf


def get_test_score(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    true_labels = y_test

    clf_report = classification_report(true_labels, predictions, digits=5)
    print(confusion_matrix(true_labels, predictions))
    return clf_report


def evaluate(clf):
    # global features_list
    # # Now it's your turn, use the model you have just created :)
    #
    # # Read the valuation json, preprocess it and run your model
    # with open(f'./datasets/dataset_{str(dataset_number)}_val.json') as file:
    #     raw_ds = json.load(file)
    # test_df = pd.json_normalize(raw_ds, max_level=2)
    #
    # for column in test_df.loc[:, test_df.isna().any()].columns.tolist():
    #     # df.drop(column, axis=1, inplace=True)
    #     test_df[column] = test_df[column].fillna('None')
    #
    # # Preprocess the validation dataset, remember that here you don't have the labels
    # test_df = vectorize_df(test_df)
    #
    # # Predict with your model
    # X = test_df[features_list].to_numpy()
    #
    # oe.
    global X_eval
    X_eval = X_eval
    y_eval = clf.predict(X_eval)

    # print(f'before: {np.unique(y_eval, return_counts=True)}')
    # y_eval = le.transform(y_eval)
    # print(f'after: {np.unique(y_eval, return_counts=True)}')
    return y_eval


def save_results(clf, test_report, predictions):
    clf_name = str(type(clf)).split(".")[-1][:-2]
    # print(predictions)
    # print(np.unique(predictions, return_counts=True))
    enc = LabelEncoder()
    np.savetxt(f'./predictions/dataset_{str(dataset_number)}_{test_type}_result.txt', enc.fit_transform(predictions),
               fmt='%2d')


if __name__ == '__main__':
    global COLUMNS_TO_ONE_HOT_ENCODE, COLUMNS_TO_TF_IDF_ENCODE, COLUMNS_TO_REMOVE, dataset_number, test_type


    dataset_number = 3
    test_type = 'label'

    # Setting features for further feature extraction by choosing columns
    # Some will be "simply" encoded via label encoding and others with HashingVectorizer


    COLUMNS_TO_REMOVE = {
        'request.headers.Host',
        'request.headers.Accept',
        'request.headers.Connection',
        'request.headers.Sec-Fetch-User',

        # 'request.headers.Cookie',
        # 'response.headers.Set-Cookie',

        # 'request.headers.Set-Cookie',
        # 'request.url',
        #
        # 'response.body',

        'request.body',
        'request.headers.Date',
        'response.headers.Content-Length',
    }


    global_settings()
    label_arrangements()
    preprocessing()
    label_encoding_and_feature_extraction()
    #print(df.head())

    X_train, X_test, y_train, y_test = train_test_split_ret()

    clf = model(X_train, y_train)
    test_report = get_test_score(clf, X_test, y_test)
    print(test_report)
    predictions = evaluate(clf)

    save_results(clf, test_report, predictions)
