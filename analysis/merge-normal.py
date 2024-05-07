import os

from l2rmeasures import getGeoRisk
import numpy as np

# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results/"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-ml5k\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0903\\resultados-web10k-tuning\\results/"
home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\multilayer\resultados-web10k-multilayer\results"
home = r"D:\Colecoes\experimento_loss_risk\tuned-datay-mq2007\resultados-datay-mlp-tuned\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0903\\resultados-ml5k-tuning\\results"
methods = os.listdir(home)
# methods = ["geoRiskListnetLossfold-2-2", "geoRiskListnetLossfold-2-3", "lambdaLossfold", "listNetfold"]
metrics = ['lndcg_10', 'lndcg_5', 'ndcg_10', 'ndcg_5']

removing = []
for method in methods:
    fold_name = method
    path = home + "/" + fold_name + "/model.predict." + "lndcg_10" + ".txt"
    if not os.path.isfile(path):
        removing.append(method)

for r in removing:
    methods.remove(r)

def read_lines(path):
    r = []
    with open(path, 'r') as p:
        for line in p:
            r.append(float(line.strip().replace("\n", "")))
    return r


methods_dics = {}

for metric in metrics:
    mat = []
    size = 0
    for method in methods:
        fold_name = method
        path = home + "/" + fold_name + "/model.predict." + metric + ".txt"
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
