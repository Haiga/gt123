from l2rmeasures import getGeoRisk
import numpy as np

# Junta os resultados de múltiplos folds de uma mesma configuração, cada config gera um novo arquivo

# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1203\\resultados-ml5k-completo2\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1203\\resultados-web10k-completo2\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-fim-1303\\resultados-web10k-completo2\\results"

# methods = ["lambdaLoss", "listNet", "spearmanLoss", "geoRiskSpearmanLoss", "geoRiskListnetLoss", "geoRiskLambdaLoss"]
# methods = ["lambdaLoss", "listNet", "spearmanLoss", "geoRiskSpearmanLoss", "geoRiskListnetLoss", "pointwise_rmse", "geoRiskLambdaLoss"]
# methods = ["lambdaLoss", "listNet", "spearmanLoss", "geoRiskSpearmanLoss", "geoRiskListnetLoss", "geoRiskLambdaLoss"]
# methods = ["lambdaLoss", "listNet", "spearmanLoss", "geoRiskSpearmanLoss", "geoRiskListnetLoss", "geoRiskLambdaLoss"]

# home = "D:\\Colecoes\\experimento_loss_risk\\grouped\\web10k\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\grouped\\ml20m\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\grouped\\yahoo\\results"
# home = r"D:\Colecoes\experimento_loss_risk\geral\web10k\results"
# home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\multilayer\resultados-web10k-multilayer2\results"
# home = r"D:\Colecoes\experimento_loss_risk\tuned-multilayer\resultados-web10k-multilayer8\results"
home = r"D:\Colecoes\experimento_loss_risk\tuned-datay-mq2007\resultados-datay-mlp-tuned\results"
home = r"D:\Colecoes\experimento_loss_risk\tuned-datay-mq2007\resultados-mq2007-mlp-tuned\results"
home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-mq2007-final-att2"
home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-mq2007-final-mlp2"
home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-datay-final-mlp2"
home = r"C:\Users\pedro\Downloads\mq2007"
# home = r"D:\Colecoes\experimento_loss_risk\geral\ml20m\results"
# home = r"D:\Colecoes\experimento_loss_risk\geral\yahoo\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1703\\ml20m\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1703\\web10k\\results"
# methods = ["lambdaLoss", "listNet", "geoRiskSpearmanLoss", "geoRiskListnetLoss", "geoRiskLambdaLoss", "pointwise_rmse", "ordinal"]
# methods = ["lambdaLoss", "listNet", "geoRiskSpearmanLoss", "geoRiskListnetLoss", "pointwise_rmse", "ordinal"]
methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'ordinal',
           'pointwise_rmse']
methods = ['grisklmart']
# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'pointwise_rmse']
# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet']
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
# for rep in [1]:
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
                fold_name = method + "Fold" + str(rep) + ""
                # fold_name = method + "fold" + str(rep) + ""
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
    path = home + "/" + 'grisklmart2' + ".tsv"
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
