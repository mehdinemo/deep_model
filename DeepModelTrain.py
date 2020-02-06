import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pyodbc
import pandas as pd
import codecs
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


def _prepare_data(all_users_sql, user_word_sql, keywords_path, config):
    connection_string_classification = config['connection_string_classification']
    connection_string_concepts = config['connection_string_concepts']
    conn_concepts = pyodbc.connect(connection_string_concepts)
    conn_classification = pyodbc.connect(connection_string_classification)

    with codecs.open(all_users_sql, 'r', encoding='utf-8') as file:
        AllUsers_query = file.read()

    with codecs.open(user_word_sql, 'r', encoding='utf-8') as file:
        User_Word_query = file.read()

    AllUsers = pd.read_sql_query(AllUsers_query, conn_classification)

    User_Word_query = User_Word_query.format(AllUsers_query)
    User_Word = pd.read_sql_query(User_Word_query, conn_concepts)

    keywords_mn = pd.read_csv(keywords_path)

    User_Word = User_Word[User_Word['Word'].isin(keywords_mn['Words'])]
    User_Word = User_Word[User_Word['MessageCount'] >= 5]
    User_Word.reset_index(drop=True, inplace=True)
    # User_Word.drop(['MessageCount'], axis=1, inplace=True)

    data = pd.pivot_table(User_Word, values='MessageCount', index='UserID', columns='Word').reset_index()
    data.fillna(0, inplace=True)
    # data = data.set_index('UserID')

    # # drop extra features (keywords)
    # key_diff = set(data.columns) - set(keywords_mn['Words'])
    # if len(key_diff) > 0:
    #     data.drop(key_diff, axis=1, inplace=True)

    # # add features that dose'nt exit in data, with zero value
    # key_diff = set(keywords_mn['Words']) - set(data.columns)
    # if len(key_diff) > 0:
    #     for i in key_diff:
    #         data[i] = 0

    # sort by headers
    data = data.reindex(sorted(data.columns), axis=1)

    # sort by UserID
    data.sort_values(by=['UserID'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return {'data': data, 'AllUsers': AllUsers}


def _classesToBinary(n, numberofclasses):
    class_array = np.zeros(numberofclasses, dtype=int)
    class_array[int(n)] = 1

    return class_array


def _build_labels(allUsers: pd.DataFrame):
    tags_map = pd.DataFrame()
    tags_map['FK_TagId'] = allUsers['FK_TagId'].unique()
    tags_map.sort_values(by=['FK_TagId'], inplace=True)
    tags_map.reset_index(drop=True, inplace=True)
    tags_map.reset_index(inplace=True)

    same_included = pd.merge(allUsers, tags_map, how='left', left_on=['FK_TagId'], right_on=['FK_TagId'])

    numberofclasses = tags_map.shape[0]
    user_labels = pd.DataFrame(
        same_included.apply(lambda x: _classesToBinary(x['index'], numberofclasses), axis=1))
    user_labels = pd.DataFrame(user_labels[0].values.tolist())
    user_labels['FK_UserId'] = same_included['FK_UserId']

    user_labels = user_labels.groupby(['FK_UserId']).sum()
    user_labels.sort_index(inplace=True)
    # user_labels.reset_index(inplace=True)

    return {'user_labels': user_labels, 'tags_map': tags_map}


def _train_model(x_train, y_train, epochs, batch_size):
    input_dim = x_train.shape[1]
    n_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(n_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    scores = model.evaluate(x_train, y_train, verbose=0)

    return {"model": model, "scores": scores}


def _save_model(model, model_path):
    # serialize model to JSON
    json_path = model_path + 'model.json'
    h5_path = model_path + 'model.h5'
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_path)
    print("Saved model to disk")


def _calculate_accuracy(y_test, test_model_results, label_threshhold) -> dict:
    tmp = test_model_results >= label_threshhold
    tmp = tmp.astype(int)

    predict = tmp * 2
    predict = predict - 1
    predict = predict.astype(int)

    true_positive = (predict == y_test).sum()
    actual_positive = (y_test == 1).sum()
    predict_positive = (tmp == 1).sum()

    precision = true_positive / predict_positive
    recall = true_positive / actual_positive

    res = {'Test Users': y_test.shape[0], 'True Positive': true_positive, 'Actual Positive': actual_positive, 'Predict Positive': predict_positive,
           'Precision': round(precision, 2), 'Recall': round(recall, 2)}

    return res


def main():
    save_model = True

    with codecs.open('app.config.json', 'r', encoding="utf-8") as file:
        config = json.load(file)

    epochs = config['epochs']
    batch_size = config['batch_size']
    label_threshhold = config['label_threshhold']
    train_percent = config['train_percent']

    # all_users_sql = r'query/AllUsers-train.sql'
    # user_word_sql = r'query/User-Word-train.sql'
    # keywords_path = r'data\keywords_mn.csv'
    # results = _prepare_data(all_users_sql, user_word_sql, keywords_path, config)
    # data = results['data']
    # AllUsers = results['AllUsers']
    #
    # data.to_csv(r'data/user_keywords.csv', index=False)
    # AllUsers.to_csv(r'data/AllUsers.csv', index=False)

    data = pd.read_csv(r'data/user_keywords.csv')
    AllUsers = pd.read_csv(r'data/AllUsers.csv')

    AllUsers = AllUsers[AllUsers['FK_UserId'].isin(data['UserID'])]
    AllUsers.reset_index(drop=True, inplace=True)

    data.set_index('UserID', inplace=True)
    data = data.astype(int)

    results = _build_labels(AllUsers)
    user_labels = results['user_labels']
    tags_map = results['tags_map']

    keywords = pd.DataFrame(data.columns, columns=['keywords'])
    keywords.to_csv(r'data\keywords.csv', index=False)
    tags_map.to_csv(r'data\tags_map.csv', index=False)

    # choose train and test data by random
    train_data = data.sample(frac=train_percent)
    test_data = data[~data.index.isin(train_data.index)]

    train_labels = user_labels[user_labels.index.isin(train_data.index)]
    test_labels = user_labels[user_labels.index.isin(test_data.index)]

    # sort train and test data and labels by UserID (index)
    train_data.sort_index(inplace=True)
    test_data.sort_index(inplace=True)
    train_labels.sort_index(inplace=True)
    test_labels.sort_index(inplace=True)

    x_train = train_data.values
    y_train = train_labels.values

    x_test = test_data.values
    y_test = test_labels.values

    print('train model')
    results = _train_model(x_train, y_train, epochs, batch_size)
    model = results['model']
    scores = results['scores']
    print(f'scores of models: {scores}')

    model_path = r'model/'
    if save_model:
        _save_model(model, model_path)

    if x_test.size != 0:
        test_model_results = model.predict(x_test)
        res = _calculate_accuracy(y_test, test_model_results, label_threshhold)
        for k, v in res.items():
            print(k, v)


if __name__ == '__main__':
    main()
