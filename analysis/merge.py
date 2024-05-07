from l2rmeasures import getGeoRisk
import numpy as np

# Junta os resultados de múltiplos folds de uma mesma configuração, cada config gera um novo arquivo

home = "D:\\Colecoes\\experimento_loss_risk\\execucao-2302\\analise1"
methods = ["geoRiskListnetLossfold-2-2", "geoRiskListnetLossfold-2-3", "lambdaLossfold", "listNetfold"]
metrics = ['lndcg_10', 'lndcg_5']


def read_lines(path):
    r = []
    with open(path, 'r') as p:
        for line in p:
            r.append(float(line.strip().replace("\n", "")))
    return r


methods_dics = {}
for rep in [1, 2]:
    for metric in metrics:
        for fold in [1, 2, 3, 4, 5]:
            mat = []
            size = 0
            for method in methods:
                fold_name = method.replace("fold", "fold" + str(fold))
                path = home + "/" + str(rep) + "/results/" + fold_name + "/model.predict." + metric + ".txt"
                if method not in methods_dics:
                    methods_dics[method] = {}
                if metric not in methods_dics[method]:
                    methods_dics[method][metric] = []
                r = read_lines(path)
                mat.append(r)
                size = len(r)
                methods_dics[method][metric].extend(r)

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
