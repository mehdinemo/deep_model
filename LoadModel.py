import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import codecs
import json
import pyodbc
import pandas as pd
from keras.models import model_from_json


def _prepare_data(all_users_sql, user_word_sql, keywords_path, config):
    connection_string_classification = config['connection_string_classification']
    connection_string_concepts = config['connection_string_concepts']
    conn_concepts = pyodbc.connect(connection_string_concepts)
    conn_classification = pyodbc.connect(connection_string_classification)

    # with codecs.open(all_users_sql, 'r', encoding='utf-8') as file:
    #     AllUsers_query = file.read()

    with codecs.open(user_word_sql, 'r', encoding='utf-8') as file:
        User_Word_query = file.read()

    # AllUsers = pd.read_sql_query(AllUsers_query, conn_classification)

    # User_Word_query = User_Word_query.format(AllUsers_query)
    User_Word = pd.read_sql_query(User_Word_query, conn_concepts)

    keywords = pd.read_csv(keywords_path)

    User_Word = User_Word[User_Word['Word'].isin(keywords['keywords'])]
    User_Word = User_Word[User_Word['MessageCount'] >= 5]
    User_Word.reset_index(drop=True, inplace=True)
    # User_Word.drop(['MessageCount'], axis=1, inplace=True)

    data = pd.pivot_table(User_Word, values='MessageCount', index='UserID', columns='Word').reset_index()
    data.fillna(0, inplace=True)
    data = data.set_index('UserID')

    # drop extra features (keywords)
    key_diff = set(data.columns) - set(keywords['keywords'])
    if len(key_diff) > 0:
        data.drop(key_diff, axis=1, inplace=True)

    # add features that dose'nt exit in data, with zero value
    key_diff = set(keywords['keywords']) - set(data.columns)
    if len(key_diff) > 0:
        for i in key_diff:
            data[i] = 0

    # sort by headers
    data = data.reindex(sorted(data.columns), axis=1)

    return {'data': data}


def _load_model(json_model_path: str, h5_model_path: str):
    # load json and create model
    json_file = open(json_model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5_model_path)

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


def main():
    with codecs.open('app.config.json', 'r', encoding="utf-8") as file:
        config = json.load(file)

    all_users_sql = r'query/Anonymous-Users.sql'
    user_word_sql = r'query/Anonymous-User-Word.sql'
    keywords_path = r'data/keywords.csv'

    # ############################## load and write data #################################
    results = _prepare_data(all_users_sql, user_word_sql, keywords_path, config)
    data = results['data']
    data.to_csv(r'data/anonymous_user_keywords.csv', index=True)

    # #################################### read data #####################################
    # data = pd.read_csv(r'data/anonymous_user_keywords.csv')
    # data.set_index('UserID', inplace=True)
    # ################################### prepare data ####################################
    # sort by UserID
    data.sort_index(inplace=True)

    data = data.astype(int)

    x = data.values

    json_model_path = r'model/model.json'
    h5_model_path = r'model/model.h5'
    model = _load_model(json_model_path=json_model_path, h5_model_path=h5_model_path)

    y = model.predict(x)
    labels = pd.DataFrame(y)
    labels.index = data.index
    labels.to_csv(r'data\anonymous_labels.csv', index=True)

    print('done')


if __name__ == '__main__':
    main()
