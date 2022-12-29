# Imports, settings and first dataset view
import pandas as pd
import seaborn as sns
import numpy as np
import json
from datetime import date

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

df = None
features_list = None

def global_settings():
    global df
    global dataset_number
    global test_type
    # Set pandas to show all columns when you print a dataframe
    pd.set_option('display.max_columns', None)

    # Global setting here you choose the dataset number and classification type for the model
    dataset_number = dataset_number  # Options are [1, 2, 3, 4]
    test_type = test_type  # Options are ['label', 'attack_type']

    # Read the json and read it to a pandas dataframe object, you can change these settings
    with open(f'./datasets/dataset_{str(dataset_number)}_train.json') as file:
        raw_ds = json.load(file)
    df = pd.json_normalize(raw_ds, max_level=2)


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
    for column in df.columns[df.isna().any()].tolist():
        # df.drop(column, axis=1, inplace=True)
        df[column] = df[column].fillna('None')

    # If you want to detect columns that may have only some NaN values use this:
    # df.loc[:, df.isna().any()].tolist()


# This is our main preprocessing function that will iterate over all of the chosen
# columns and run some feature extraction models
def vectorize_df(df):
    le = LabelEncoder()
    h_vec = HashingVectorizer(n_features=4)

    # Run LabelEncoder on the chosen features
    for column in SIMPLE_HEADERS:
        df[column] = le.fit_transform(df[column])

    # Run HashingVectorizer on the chosen features
    for column in COMPLEX_HEADERS:
        newHVec = h_vec.fit_transform(df[column])
        df[column] = newHVec.todense()

    # Remove some columns that may be needed.. (Or not, you decide)
    for column in COLUMNS_TO_REMOVE:
        df.drop(column, axis=1, inplace=True)
    return df


def label_encoding_and_feature_extraction():
    global df
    global features_list

    df = vectorize_df(df)

    features_list = df.columns.to_list()
    features_list.remove('label')
    features_list.remove('attack_type')


def train_test_split_ret():
    global df
    global features_list
    global test_type
    # Data train and test split preparations. Here we will insert our feature list and label list.
    # Afterwards the data will be trained and fitted on the amazing XGBoost model
    # X_Train and y_Train will be used for training
    # X_test and y_test.T will be used for over fitting checking and overall score testing

    # We convert the feature list to a numpy array, this is required for the model fitting
    X = df[features_list].to_numpy()

    # This column is the desired prediction we will train our model on
    y = np.stack(df[test_type])

    # We split the dataset to train and test according to the required ration
    # Do not change the test_size -> you can change anything else
    return train_test_split(X, y, test_size=0.1765, random_state=42, stratify=y)


def model(X_train, y_train):
    # We choose our model of choice and set it's hyper parameters you can change anything
    clf = RandomForestClassifier(n_estimators=100)

    # Train Model
    clf.fit(X_train, y_train)

    return clf


def get_test_score(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    true_labels = y_test

    clf_report = classification_report(true_labels, predictions, digits=5)
    return clf_report


def evaluate(clf):
    global features_list
    # Now it's your turn, use the model you have just created :)

    # Read the valuation json, preprocess it and run your model
    with open(f'./datasets/dataset_{str(dataset_number)}_val.json') as file:
        raw_ds = json.load(file)
    test_df = pd.json_normalize(raw_ds, max_level=2)

    for column in test_df.columns[test_df.isna().any()].tolist():
        # df.drop(column, axis=1, inplace=True)
        test_df[column] = test_df[column].fillna('None')

    # Preprocess the validation dataset, remember that here you don't have the labels
    test_df = vectorize_df(test_df)

    # Predict with your model
    X = test_df[features_list].to_numpy()
    return clf.predict(X)


def save_results(clf, test_report, predictions):
    clf_name = str(type(clf)).split(".")[-1][:-2]
    with open(f'./scores/dataset{str(dataset_number)}_{clf_name}_{date.today()}.txt', mode='w') as file:
        file.write(test_report)

    # Save your preditions
    enc = LabelEncoder()
    np.savetxt(f'./predictions/dataset_{str(dataset_number)}_{test_type}_result.txt', enc.fit_transform(predictions),
               fmt='%2d')


if __name__ == '__main__':
    global SIMPLE_HEADERS, COMPLEX_HEADERS, COLUMNS_TO_REMOVE, dataset_number, test_type
    # Setting features for further feature extraction by choosing columns
    # Some will be "simply" encoded via label encoding and others with HashingVectorizer

    # On these headers we will run a "simple" BOW
    SIMPLE_HEADERS = ['request.headers.Accept-Encoding',
                      'request.headers.Connection',
                      'request.headers.Host',
                      'request.headers.Accept',
                      'request.method',
                      'request.headers.Accept-Language',
                      'request.headers.Sec-Fetch-Site',
                      'request.headers.Sec-Fetch-Mode',
                      'request.headers.Sec-Fetch-Dest',
                      'request.headers.Sec-Fetch-User',
                      'response.status',
                      ]

    # On these headers we will run HashingVectorizer
    COMPLEX_HEADERS = ['request.headers.User-Agent',
                       'request.headers.Set-Cookie',
                       'request.headers.Date',
                       'request.url',
                       'response.headers.Content-Type',
                       'response.body',
                       'response.headers.Location',
                       'request.headers.Content-Length',
                       'request.headers.Cookie',
                       'response.headers.Set-Cookie'
                       ]

    COLUMNS_TO_REMOVE = ['request.body',
                         'response.headers.Content-Length',
                         'request.headers.Date']

    dataset_number = 1
    test_type = 'label'

    global_settings()
    label_arrangements()
    preprocessing()
    label_encoding_and_feature_extraction()

    X_train, X_test, y_train, y_test = train_test_split_ret()

    clf = model(X_train, y_train)
    test_report = get_test_score(clf, X_test, y_test)
    print(test_report)
    predictions = evaluate(clf)

    save_results(clf, test_report, predictions)



