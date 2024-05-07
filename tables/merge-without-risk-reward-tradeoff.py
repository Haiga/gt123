import glob
import numpy
from analysis.l2rmeasures import getGeoRisk
import numpy as np
from scipy.stats import ttest_ind

# home = r"/home/silvapedro/experimento_loss_risk/resultados-web10k-mlp-eb-exec1/results"
home = r'D:\Colecoes\experimento_loss_risk\tables\overall\att\mq2007'
home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\yahoo'

# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'ordinal',
#            'pointwise_rmse']
# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'extgeoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'ordinal',
#            'pointwise_rmse', 'grisklmart']
methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss',
           # methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'lambdaLoss',
           'grisklmart', 'lmart',
           'lambdaLoss', 'lambdaLossmulti', 'listNet', 'listNetmulti', 'ordinal', 'ordinalmulti',
           'pointwise_rmse', 'pointwise_rmsemulti', 'spearmanLoss',
           'spearmanLossmulti',
           ]

# 'listNet', 'spearmanLoss', 'ordinal', 'pointwise_rmse']
metrics = ['lndcg_10', 'lndcg_5']
# metrics = ['ndcg_10', 'ndcg_5']
# reps = [1, 2, 3, 4, 5]
reps = [1]


def read_lines(path):
    r = []
    print(path)
    with open(path, 'r') as p:
        for line in p:
            r.append(float(line.strip().replace("\n", "")))
    return r


methods_dics = {}

# for rep in [1]:

for rep in reps:

    for metric in metrics:
        # for fold in [1, 2, 3, 4, 5]:
        for fold in ['']:
            mat = []
            size = 0
            for method in methods:

                if method not in methods_dics:
                    methods_dics[method] = {}
                if metric not in methods_dics[method]:
                    methods_dics[method][metric] = []

                if "*" in method:
                    max_r = 0
                    r = []
                    for sub in ["ListnetLoss", "SpearmanLoss"]:
                        fold_name = method.replace("*", sub) + "fold" + str(rep) + "-"
                        # fold_name = method + "fold" + str(rep) + ""
                        # path = home + "/" + str(rep) + "/results/" + fold_name + "/model.predict." + metric + ".txt"
                        path = home + "/" + fold_name + "/model.predict." + metric + ".txt"
                        temp_r = read_lines(path)
                        if np.mean(temp_r) > max_r:
                            max_r = np.mean(temp_r)
                            r = temp_r

                else:
                    fold_name = method + "fold" + str(rep) + "-"
                    # fold_name = method + "fold" + str(rep) + ""
                    # path = home + "/" + str(rep) + "/results/" + fold_name + "/model.predict." + metric + ".txt"
                    path = home + "/" + fold_name + "/model.predict." + metric + ".txt"
                    r = read_lines(path)

                mat.append(r)
                size = len(r)
                methods_dics[method][metric].extend(r)
            # print(rep)
            cont = 0
            georisk = getGeoRisk(np.array(mat).T, alpha=4)
            for method in methods:
                if "georisk5" + metric not in methods_dics[method]:
                    methods_dics[method]["georisk5" + metric] = []
                r = [georisk[cont]] * size
                methods_dics[method]["georisk5" + metric].extend(r)
                cont += 1

g_metrics = []
for metric in metrics:
    g_metrics.append("georisk5" + metric)

metrics.extend(g_metrics)


def getMean(metric):
    soma = []
    for method in methods:
        if soma == []:
            soma = np.array(methods_dics[method][metric])
        else:
            soma += np.array(methods_dics[method][metric])
    return soma / len(methods_dics)




metric = 'lndcg_10'
mean = getMean(metric)

methods = [
    'geoRiskSpearmanLoss', 'extgeoRiskSpearmanLoss',
    # methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'lambdaLoss',lambdaLoss', 'listNet', 'ordinal', 'pointwise_rmse', 'spearmanLoss',

    # 'lambdaLossmulti','pointwise_rmse',
    'lambdaLossmulti',
    # 'pointwise_rmse',
    # 'grisklmart', 'lmart'
]
for method in methods:
    reawrd = np.sum(
        (np.array(methods_dics[method][metric]) - mean) * ((np.array(methods_dics[method][metric]) - mean) > 0))
    degrad = -np.sum(
        (np.array(methods_dics[method][metric]) - mean) * ((np.array(methods_dics[method][metric]) - mean) < 0))

    num_wins = np.sum((np.array(methods_dics[method][metric]) - mean) > 0)
    num_losses = np.sum(((np.array(methods_dics[method][metric]) - mean) < 0))
    # num_losses_g20 = np.sum(((np.array(methods_dics[method][metric]) - mean) < 0) * ((np.array(methods_dics[method][metric]) - mean) > 0.2 * mean))
    num_losses_g20 = np.sum(((np.array(methods_dics[method][metric]) - mean) * 100) / mean > 20)

    print(method, end="\t")
    print(np.mean(methods_dics[method][metric]), end="\t")
    print(reawrd/num_wins, end="\t")
    print(degrad/num_losses, end="\t")
    print((reawrd/num_wins) / (degrad/num_losses), end="\t")
    print(num_wins, end="\t")
    print(num_losses, end="\t")
    print(num_wins / num_losses, end="\t")
    print(num_losses_g20, end="\n")
