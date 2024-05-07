import numpy as np
from sklearn.datasets import load_svmlight_file
import pandas as pd
import pickle


def get_data():
    labels_by_query = []
    features_docs_by_query = []
    all_queries_ids = []
    # for f in range(5):
    for f in range(1):
        # fold = 1
        fold = f + 1
        # path = f'D:\Colecoes\BD\mq2007\S{fold}.txt'
        # path = f'D:\Colecoes\BD\web10k\Fold{fold}\\Norm.test.txt'
        path = f'D:\Colecoes\BD\yahoo-c14\l2r\yahoo\\Norm.test.txt'
        data = load_svmlight_file(path, query_id=True)

        queries_ids = np.array(data[2])
        ant = queries_ids[0]
        all_queries_ids.extend(np.unique(queries_ids))

        temp_labels_by_query = []
        temp_features_docs_by_query1 = []

        for i in range(queries_ids.size):
            if ant != queries_ids[i]:
                labels_by_query.append(temp_labels_by_query)
                features_docs_by_query.append(temp_features_docs_by_query1)

                temp_labels_by_query = []
                temp_features_docs_by_query1 = []

            temp_labels_by_query.append(data[1][i])
            # temp_features_docs_by_query1.append(data[0][i].toarray().reshape(-1)[85])
            temp_features_docs_by_query1.append(data[0][i].toarray().reshape(-1))
            ant = queries_ids[i]

        labels_by_query.append(temp_labels_by_query)
        features_docs_by_query.append(temp_features_docs_by_query1)

    return features_docs_by_query, all_queries_ids


features_docs_by_query, qids = get_data()
qids = np.array(qids)

X = []
for query in features_docs_by_query:
    X.append(np.mean(query, axis=0))

from sklearn.cluster import KMeans



# home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\mlp\web10k'
home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\mlp\yahoo'
# home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\yahoo'
metric = 'lndcg_10'

# df_risk_values = pd.read_csv(home + "/" + 'geoRisk.tsv', sep="\t")[metric].values
# df_non_values = pd.read_csv(home + "/" + 'pointwise_rmse.tsv', sep="\t")[metric].values

# df_risk_values = pd.read_csv(home + "/" + 'geoRiskSpearmanLoss.tsv', sep="\t")[metric].values
# df_non_values = pd.read_csv(home + "/" + 'spearmanLossmulti.tsv', sep="\t")[metric].values

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except:
        x = []
    return x

def dcg(true_relevance, pred_relevance, k=5, gains='linear', use_numpy=False):
    np_true_relevance = np.array(true_relevance)
    if not use_numpy:
        args_pred = [i[0] for i in sorted(enumerate(pred_relevance), key=lambda p: p[1], reverse=True)]
    else:
        args_pred = np.argsort(pred_relevance)[::-1]
    if np_true_relevance.shape[0] < k:
        k = np_true_relevance.shape[0]
    order_true_relevance = np.take(np_true_relevance, args_pred[:k])

    if gains == "exponential":
        gains = 2 ** order_true_relevance - 1
    elif gains == "linear":
        gains = order_true_relevance
    else:
        raise ValueError("Invalid gains option.")

    discounts = np.log2(np.arange(k) + 2)
    return np.sum(gains / discounts)


def ndcg(true_relevance, pred_relevance, k=5, no_relevant=True, gains='linear', use_numpy=False):
    dcg_atk = dcg(true_relevance, pred_relevance, k, gains, use_numpy)
    idcg_atk = dcg(true_relevance, true_relevance, k, gains, use_numpy)
    if idcg_atk == 0 and no_relevant: return 1.0
    if idcg_atk == 0 and not no_relevant: return 0.0
    return dcg_atk / idcg_atk


def mNdcg(true_relevance, pred_relevance, k=5, no_relevant=False, gains='linear', use_numpy=False):
    np_true_relevance = np.array(true_relevance)
    num_queries = np_true_relevance.shape[0]
    return [ndcg(true_relevance[i], pred_relevance[i], k, no_relevant, gains, use_numpy) for i in range(num_queries)]


# means_of_dir1_by_querie = []
# x5 = load_data("../plots/mq2007/30.dat")
x5 = load_data("../plots/115.dat")
# for querie in range(len(x5)):
#     means_of_dir1_by_querie.append(np.mean(x5[querie]))
# means_of_dir1_by_querie = np.array(means_of_dir1_by_querie)
y = load_data("../plots/y.dat")
means_of_dir1_by_querie = mNdcg(y, x5, k=10000, gains='exponential', use_numpy=True)
means_of_dir1_by_querie = np.array(means_of_dir1_by_querie)

import glob
files = glob.glob(home + "/*.tsv")

# for num_cluster in [5, 10]:
for num_cluster in [5]:
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    labels = kmeans.labels_

    for i in range(num_cluster):
        mask = labels == i
        qids_of_cluster = qids[mask]

        print(np.sum(mask)/len(mask))
        # print(np.mean(means_of_dir1_by_querie[mask]))

        for file in files:
            alg = file.split("\\")[-1]
            # if "geoRisk" in alg: continue
            # if "lambdaLossmulti" in alg or "extgeoRiskSpearmanLoss" in alg or "geoRiskSpearmanLoss" in alg:
            # if "lambdaLossmulti" in alg or "geoRiskListnetLoss.tsv" == alg or "extgeoRiskSpearmanLoss" in alg:
            # if "pointwise_rmsemulti" in alg or "geoRisk" in alg or "extgeoRisk" in alg:
            if "lambdaLossmulti" in alg or "extgeoRiskListnetLoss" in alg or "geoRiskSpearmanLoss.tsv" == alg:
            # if "pointwise_rmse." in alg or "extgeoRisk" in alg or "geoRisk" in alg:
                if "ic-file" in alg: continue
                if "losses-wins" in alg: continue
                if "summed" in alg: continue

                df_risk_values = pd.read_csv(home + "/" + alg, sep="\t")[metric].values

                print(f"{alg}: " + str(int(10000*np.mean(df_risk_values[mask]))/10000))
        # print(np.mean(df_non_values[mask]))
        print("--------")
    print("####################")
