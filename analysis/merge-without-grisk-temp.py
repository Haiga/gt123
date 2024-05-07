from l2rmeasures import getGeoRisk
import numpy as np

# Junta os resultados de múltiplos folds de uma mesma configuração, cada config gera um novo arquivo


# home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\web10k2\results"
# home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\dropout-geral"
# home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\resultados-web10k-dropoutgrisk-literature3\results"
# home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\resultados-web10k-baseline-external\results"
home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\regularizer\resultados-web10k-regularizer\results - Copia2"

# methods = ['geoRiskSpearmanLossFx--0', 'geoRiskListnetLossFx--0',
#            'geoRiskSpearmanLossFx--1', 'geoRiskListnetLossFx--1']

# methods = ['geoRiskSpearmanLossFx--1', 'geoRiskListnetLossFx--1']
# methods = ['lambdaLoss']
# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'ordinal',
#            'pointwise_rmse', 'grisklmart', 'trisklmart']
methods = ['lambdaLoss', 'listNet', 'pointwise_rmse','geoRiskSpearmanLoss', 'geoRiskListnetLoss',
           'lambdaLoss-GL-0.01', 'lambdaLoss-GS-100',
           'pointwise_rmse-GL-1', 'pointwise_rmse-GS-100',
           'listNet-GL-100', 'listNet-GS-100',
           ]

# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss']

metrics = ['lndcg_10', 'lndcg_5']


# metrics = ['ndcg_10', 'ndcg_5']


def read_lines(path):
    r = []
    print(path)
    with open(path, 'r') as p:
        for line in p:
            r.append(float(line.strip().replace("\n", "")))
    return r


methods_dics = {}
# for rep in [46]:
# for rep in [1, 2]:
# for rep in [1, 2, 3, 4, 5]:
# for rep in [1, 2, 3]:
# for rep in [33,34,35,36,37,38,39]:
# for rep in [93, 94, 95, 97]:
# for rep in [93, 94, 95, 96, 97]:
for rep in [1, 2, 3, 4, 5]:
    # for rep in [12, 46, 1212, 830, 1000]:
    # for rep in [12, 46]:
    # for rep in [1212]:
    # for rep in [12, 46]:
    # for rep in [46]:
    # for rep in [42, 43, 44]:
    # for rep in [42]:
    for metric in metrics:
        # for fold in [1, 2, 3, 4, 5]:
        for fold in ['']:
            mat = []
            size = 0
            for method in methods:

                # fold_name = method.replace("Fx", "fold" + str(rep))
                fold_name = method + "fold" + str(rep) + "-"
                if "-GL-" in method:
                    fold_name = method.split("-GL-")[0] + "regfold" + str(rep) + "-GL-" + method.split("-GL-")[1]
                elif "-GS-" in method:
                    fold_name = method.split("-GS-")[0] + "regfold" + str(rep) + "-GS-" + method.split("-GS-")[1]

                # path = home + "/" + str(rep) + "/results/" + fold_name + "/model.predict." + metric + ".txt"
                path = home + "/" + fold_name + "/model.predict." + metric + ".txt"
                if method not in methods_dics:
                    methods_dics[method] = {}
                if metric not in methods_dics[method]:
                    methods_dics[method][metric] = []
                r = read_lines(path)
                mat.append(r)
                size = len(r)
                methods_dics[method][metric].extend(r)
            # print(rep)
            cont = 0
            georisk = getGeoRisk(np.array(mat).T, alpha=5)
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

total_size = len(methods_dics[methods[0]][metrics[0]])

for method in methods:
    path = home + "/" + method + ".tsv"
    with open(path, "w") as fo:
        for metric in metrics:
            fo.write(metric + "\t")
        fo.write("\n")
        for i in range(total_size):
            for metric in metrics:
                fo.write(str(methods_dics[method][metric][i]) + "\t")
            fo.write("\n")

# for metric in metrics:
#     mat = []
#     for method in methods:
#         mat.append(methods_dics[method][metric])
#     georisk = getGeoRisk(np.array(mat).T, alpha=5)
#     print(georisk)
