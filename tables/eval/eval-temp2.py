import math

import numpy as np
import torch
from scipy.stats import norm


def getGeoRiskDefault(mat, alpha):
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0] * numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i] = np.sum(mat[:, i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = np.sum(mat[j, :])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd = norm.cdf(zRisk[i] / c)
        geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

    return geoRisk


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
    np_true_relevance = np.array(true_relevance)
    num_queries = np_true_relevance.shape[0]
    return [ndcg(true_relevance[i], pred_relevance[i], k, no_relevant, gains, use_numpy) for i in range(num_queries)]


def torchNdcg(ys_true, ys_pred, k=None, return_type='list'):
    def dcg(ys_true, ys_pred, k=None):
        _, argsort = torch.sort(ys_pred, descending=True, dim=0)
        ys_true_sorted = ys_true[argsort]
        ret = 0
        if not k is None:
            ys_true_sorted = ys_true_sorted[:k]
        for i, l in enumerate(ys_true_sorted, 1):
            ret += (2 ** l - 1) / np.log2(1 + i)
        return ret

    r = []
    for q in range(ys_true.shape[0]):
        ideal_dcg = dcg(ys_true[q], ys_true[q], k=k)
        pred_dcg = dcg(ys_true[q], ys_pred[q], k=k)
        r.append(pred_dcg / ideal_dcg)
    if return_type == 'tensor':
        r = torch.tensor(r)
        r[torch.isnan(r)] = 0
        # return torch.mean(r)
        return r
    return r

import os
import sys
if __name__ == "__main__":
    folder = sys.argv[1]
    filesnames = os.listdir(sys.argv[1])
    for datset in ['web10k']:
        print("######")
        h = []
        if datset == 'datay':
            folds = ['1']
        else:
            folds = ['1']
        for fold in folds:
            # for k in ['5', '10']:
            # col = 'yahoo'
            for filename in filesnames:
                metric = 'ndcg_10'
                if os.path.isdir(folder + "/" + filename) and os.path.isfile(folder + "/" + filename + "/" + f"model.predict.{metric}.txt"):
                    test_file = fr"/home/silvapedro/experimento_loss_risk/BD/default-web10k/Fold{fold}/_normalized" + r"/Norm.test.txt"
                    predictions_file = folder + "/" + filename + r"/model.predict.txt"

                    try:
                        with open(test_file) as tf:
                            with open(predictions_file) as pf:
                                q_id_anterior = -100
                                true_relevances = []
                                predicted_relevances = []
                                vec = []
                                vec_pred = []
                                cont = 0
                                for line in tf:
                                    line_pred = float(pf.readline().replace("\n", ""))
                                    splitted_line = line.split(" ")
                                    rel_d_q = float(splitted_line[0])
                                    q_id = int(splitted_line[1].split(":")[1])
                                    if q_id != q_id_anterior and cont > 0:
                                        true_relevances.append(vec)
                                        predicted_relevances.append(vec_pred)
                                        vec = [rel_d_q]
                                        vec_pred = [line_pred]
                                        q_id_anterior = q_id
                                        cont += 1
                                    else:
                                        vec.append(rel_d_q)
                                        vec_pred.append(line_pred)
                                        q_id_anterior = q_id
                                        cont += 1
                                true_relevances.append(vec)
                                predicted_relevances.append(vec_pred)

                        # predicted_relevance = np.asarray([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
                        # true_relevance = np.asarray([[0, 0, 0, 0, 1], [10, 0, 0, 0, 0]])

                        k = 10
                        # metrics_file = fr"D:\Colecoes\experimento_loss_risk\temp\tsalles\lmart2\{datset}\Fold" + fold + f"\model.predict.lndcg_{k}.txt"
                        r = mNdcg(true_relevances, predicted_relevances, k=int(k), no_relevant=False, gains='exponential', use_numpy=True)
                        # h.extend(r)
                        print(filename)
                        print(np.mean(r))
                        # with open(metrics_file, "w") as mf:
                        #     for value in r:
                        #         mf.write(str(value) + "\n")
                        # #############"
                        # k = 10
                        # metrics_file = fr"D:\Colecoes\experimento_loss_risk\temp\tsalles\lmart2\{datset}\Fold" + fold + f"\model.predict.lndcg_{k}.txt"
                        # r = mNdcg(true_relevances, predicted_relevances, k=int(k), no_relevant=False, gains='linear', use_numpy=True)
                        # print(np.mean(r))
                        # h.extend(r)
                        # with open(metrics_file, "w") as mf:
                        #     for value in r:
                        #         mf.write(str(value) + "\n")
                        # #############"
                        # k = 5
                        # metrics_file = fr"D:\Colecoes\experimento_loss_risk\temp\tsalles\lmart2\{datset}\Fold" + fold + f"\model.predict.ndcg_{k}.txt"
                        # r = mNdcg(true_relevances, predicted_relevances, k=int(k), no_relevant=False, gains='exponential', use_numpy=True)
                        # print(np.mean(r))
                        # # h.extend(r)
                        # with open(metrics_file, "w") as mf:
                        #     for value in r:
                        #         mf.write(str(value) + "\n")
                        # #############"
                        # k = 10
                        # metrics_file = fr"D:\Colecoes\experimento_loss_risk\temp\tsalles\lmart2\{datset}\Fold" + fold + f"\model.predict.ndcg_{k}.txt"
                        # r = mNdcg(true_relevances, predicted_relevances, k=int(k), no_relevant=False, gains='exponential', use_numpy=True)
                        # print(np.mean(r))
                        # # h.extend(r)
                        # with open(metrics_file, "w") as mf:
                        #     for value in r:
                        #         mf.write(str(value) + "\n")
                        # #############"
                    except:
                        pass
        print("..")
        print(np.mean(h))
