import numpy as np
from sklearn.datasets import load_svmlight_file
import pandas as pd


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


def mNdcg(true_relevance, pred_relevance, k=5, no_relevant=True, gains='linear', use_numpy=False):
    num_queries = len(true_relevance)
    return [ndcg(true_relevance[i], pred_relevance[i], k, no_relevant, gains, use_numpy) for i in range(num_queries)]

def getL(path):
    with open(path) as i:
        L = []
        for line in i:
            L.append(float(line.strip()))
        return L
if __name__ == '__main__':
    path = r"/home/silvapedro/experimento_loss_risk/BD/web30k/Fold3/Norm.test.txt"
    pathB = r"/home/silvapedro/experimento_loss_risk/resultados-w30k-temp/results/geoRiskSpearmanLossfold3-/model.predict.txt"
    X_test, y_test, qids_test = load_svmlight_file(path, query_id=True)

    # X_test = X_test.toarray()

    qids_test_count = pd.DataFrame(qids_test).groupby(0)[0].count().to_numpy()

    result = getL(pathB)
    summed_qid_count = 0
    grouped_preds_by_query = []
    grouped_rels_by_query = []
    for qid_count in qids_test_count:
        grouped_preds_by_query.append(result[summed_qid_count:summed_qid_count + qid_count])
        grouped_rels_by_query.append(y_test[summed_qid_count:summed_qid_count + qid_count])
        summed_qid_count += qid_count
    # print(mNdcg(grouped_rels_by_query, grouped_preds_by_query))
    print(np.mean(mNdcg(grouped_rels_by_query, grouped_preds_by_query, k=10, gains="exponential", no_relevant=True)))
    print(np.mean(mNdcg(grouped_rels_by_query, grouped_preds_by_query, k=5, gains="exponential", no_relevant=True)))
    print(np.mean(mNdcg(grouped_rels_by_query, grouped_preds_by_query, k=10, gains="linear", no_relevant=True)))
    print(np.mean(mNdcg(grouped_rels_by_query, grouped_preds_by_query, k=5, gains="linear", no_relevant=True)))

    print('...')
