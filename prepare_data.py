import pyodbc
import pandas as pd
import codecs
import json
import networkx as nx
from sklearn.cluster import SpectralClustering


def main():
    with codecs.open('app.config.json', 'r', encoding="utf-8") as file:
        config = json.load(file)

    connection_string_251 = config['connection_string_251']
    conn = pyodbc.connect(connection_string_251)

    with codecs.open(r'query/khakestari_ahmadi.sql', 'r', encoding='utf-8') as file:
        khakestari_ahmadi_query = file.read()

    with codecs.open(r'query/khakestari_mn.sql', 'r', encoding='utf-8') as file:
        khakestari_mn_query = file.read()

    with codecs.open(r'query/User-Word.sql', 'r', encoding='utf-8') as file:
        User_Word_query = file.read()

    User_Word_ahmadi_query = User_Word_query.format(khakestari_ahmadi_query)
    User_Word_mn_query = User_Word_query.format(khakestari_mn_query)

    # data_ahmadi = pd.read_sql_query(User_Word_ahmadi_query, conn)
    data_mn = pd.read_sql_query(User_Word_mn_query, conn)

    keywords_mn = pd.read_csv(r'data\keywords_mn.csv')

    data_mn = data_mn[data_mn['Word'].isin(keywords_mn['Words'])]
    data_mn = data_mn[data_mn['MessageCount'] >= 5]
    data_mn.reset_index(drop=True, inplace=True)
    data_mn.drop(['MessageCount'], axis=1, inplace=True)

    Nodes = data_mn.groupby(['UserID'])['Word'].count()
    Nodes = pd.DataFrame(Nodes)
    Nodes.reset_index(inplace=True)

    # creat graph
    data_mn_merge = pd.merge(data_mn, data_mn, how='inner', left_on=['Word'], right_on=['Word'])
    edges_mn = data_mn_merge.groupby(['UserID_x', 'UserID_y']).count()
    edges_mn.reset_index(inplace=True)

    G = nx.from_pandas_edgelist(edges_mn, 'UserID_x', 'UserID_y', 'Word')
    G.remove_edges_from(G.selfloop_edges())
    Edges = nx.to_pandas_edgelist(G, source='UserID_x', target='UserID_y')
    Edges.rename(columns={'UserID_x': 'source', 'UserID_y': 'target', 'Word': 'weight'}, inplace=True)

    Edges = pd.merge(Edges, Nodes, how='left', left_on=['source'], right_on=['UserID'])
    Edges = pd.merge(Edges, Nodes, how='left', left_on=['target'], right_on=['UserID'])
    Edges.drop(['source', 'target'], axis=1, inplace=True)

    Edges['sim'] = Edges['weight'] / (Edges['Word_x'] + Edges['Word_y'] - Edges['weight'])
    Edges['dis'] = 1 - Edges['sim']

    # prune graph
    # tmp = pd.DataFrame(Edges['sim'].sort_values(ascending=False))
    # tmp.reset_index(drop=True, inplace=True)
    # thresh = int(0.2 * len(tmp))
    # thresh = tmp.iloc[thresh]['sim']
    #
    # Ed = Edges[Edges['sim'] >= thresh]
    # Ed.reset_index(drop=True, inplace=True)
    # Ed.to_csv(r'data/Prune_Edges.csv', columns=['UserID_x', 'UserID_y', 'sim'], index=False)

    # similarity and distance graph
    G_sim = nx.from_pandas_edgelist(Edges, 'UserID_x', 'UserID_y', 'sim')
    # nx.write_edgelist(G_sim, r'data/sim_graph.csv')

    G_dis = nx.from_pandas_edgelist(Edges, 'UserID_x', 'UserID_y', 'dis')
    # nx.write_edgelist(G_sim, r'data/dis_graph.csv')

    # ######################## run DBSCAN ########################
    # dbs_data = DBSCAN(eps=17, min_samples=10, metric='precomputed').fit(nx.adjacency_matrix(G_data))
    #
    # ######################## Spectral Clustering ########################
    spect_cluster = SpectralClustering(n_clusters=10, affinity='precomputed', n_init=100).fit(
        nx.adjacency_matrix(G_dis))
    spectral_cluster_res = pd.DataFrame(columns=['Concept', 'Class'])
    spectral_cluster_res['UserID'] = G_dis.nodes
    spectral_cluster_res['Class'] = spect_cluster.labels_
    spectral_cluster_res.to_csv(r'data/spectral_res.csv', index=False)
    # ######################## Run Louvain #########################
    # partitions = community.best_partition(G)

    ######################## Save Files ########################
    # Save Edge List
    # nx.write_edgelist(G, 'graph.csv')





    # data_ahmadi_group = data_ahmadi[data_ahmadi['MessageCount'] >= 5].groupby(['Word'])['UserID'].count()
    # data_ahmadi_group = pd.DataFrame(data_ahmadi_group).reset_index()

    # data_mn_group = data_mn[data_mn['MessageCount'] >= 5].groupby(['Word'])['UserID'].count()
    # data_mn_group = pd.DataFrame(data_mn_group).reset_index()

    # data_ahmadi_group.to_csv(r'data/data_ahmadi_group.csv', index=False)
    # data_mn_group.to_csv(r'data/data_mn_group.csv', index=False)

    print('done')


if __name__ == '__main__':
    main()
