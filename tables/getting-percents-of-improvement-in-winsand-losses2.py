import glob
import numpy
from analysis.l2rmeasures import getGeoRisk
from createLossesWinsFiles2 import getData, processLossesWins
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from tables.createLossesWinsFiles2 import doIt


# plt.rc('text', usetex=True)


# for dataset in ['mq2007', 'yahoo', 'web10k']:
# for dataset in ['web10k','mq2007', 'yahoo']:
#     for tt in ['att', 'mlp']:

def getNames(array_origin):
    names = []
    for name in array_origin:
        r = ''
        if 'Best' in name:
            r = name
            names.append(r)
            continue
        if 'Spear' in name:
            r += 'RiskLoss SP'
        if 'List' in name:
            r += 'RiskLoss CS'
        if 'ext' in name:
            r += '+EB'
        else:
            r += '+DO'
        names.append(r)
    return names


def getNames2(name):
    if name == 'pointwise_rmse':
        return 'MSE'
    if name == 'lambdaLoss':
        return 'NDCGLoss2++'
    if name == 'listNet':
        return 'ListNet'
    if name == 'ordinal':
        return 'Ordinal'
    if name == 'spearmanLoss':
        return 'SpearmanLoss'
    if name == 'grisklmart':
        return 'GeoRisk-LM'
    return name

for metric in ['NDGC-10', 'NDCG-5']:
    for dataset in ['mq2007', 'web10k', 'yahoo']:
        # for dataset in ['web10k', 'yahoo']:
        # for dataset in ['web10k']:
        for tt in ['att', 'mlp']:
            # for tt in ['mlp']:
            # for tt in ['att']:

            # home = r"D:\Colecoes\experimento_loss_risk\tables\overall" + f"\\{tt}" + f"\\{dataset}"
            home = r"D:\Projetos Códigos\PycharmProjects\pythonProject\merged2" + f"\\{tt}" + f"\\{dataset}"

            methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss',
                       'lambdaLoss',
                       'listNet', 'ordinal', 'pointwise_rmse', 'spearmanLoss', 'grisklmart']
            if tt == 'mlp':
                methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss',
                           'lambdaLoss', 'listNet', 'ordinal', 'pointwise_rmse', 'spearmanLoss']
            metrics = ['lndcg_10', 'lndcg_5']
            # metrics = ['ndcg_10', 'ndcg_5']

            # print("Inicio")

            # home = r"D:\Colecoes\experimento_loss_risk\tables\overall\att\mq2007"

            baseline = "summed.tsv"

            files = glob.glob(home + "/*.tsv")

            if home + '\\' + 'grisklmart.tsv' in files:
                files.remove(home + '\\' + 'grisklmart.tsv')
                files.append(home + '\\' + 'grisklmart.tsv')

            data = getData(home, files, baseline)
            data, mean, diff = processLossesWins(data)


            lnd = 5
            padd = -1

            if '10' in metric:
                m = 0
            else:
                m = 1

            heights = []
            names_methods_heights = []
            for current in ['wins', 'losses', 'g20']:
                if current == 'wins':
                    i = 1
                    bestwin = 0
                    bestwing = 0
                else:
                    i = 0
                    bestwin = 10000000000
                    bestwing = 10000000000

                risks_values = []
                x = ''
                xg = ''
                for method in methods:
                    if 'geoRisk' not in method:
                        if current == 'wins':
                            bestwinb = bestwin
                            bestwin = max([bestwin, data[method + '.tsv'][metric][i]])
                            if bestwinb != bestwin:
                                x = method
                        else:
                            if current == 'g20':
                                bestwinb = bestwin
                                bestwin = min([bestwin, diff[method + '.tsv']['loss_g_20'][m]])
                                if bestwinb != bestwin:
                                    x = method
                            else:
                                bestwinb = bestwin
                                bestwin = min([bestwin, data[method + '.tsv'][metric][i]])
                                if bestwinb != bestwin:
                                    x = method
                    else:
                        if current == 'wins':
                            bestwinb = bestwing
                            bestwing = max([bestwing, data[method + '.tsv'][metric][i]])
                            if bestwinb != bestwing:
                                xg = method
                        else:
                            if current == 'g20':
                                bestwinb = bestwing
                                bestwing = min([bestwing, diff[method + '.tsv']['loss_g_20'][m]])
                                if bestwinb != bestwing:
                                    xg = method
                            else:
                                bestwinb = bestwing
                                bestwing = min([bestwing, data[method + '.tsv'][metric][i]])
                                if bestwinb != bestwing:
                                    xg = method
                if current == 'wins':
                    p = ((bestwing - bestwin) * 100) / bestwin
                else:
                    p = ((bestwin - bestwing) * 100) / bestwin

                print(metric + "\t" + dataset + "\t" + tt + "\t" + current + "\t" + getNames([xg])[0] + "\t" + getNames2(x) + "\t" + str(round(p, 2)))
                # risks_values.append(bestwin)
                # nomes_metodos = np.append(np.array(methods)[0:4], 'Best Non-RiskLoss (' + getNames2(x) + ')')
                # # nomes_metodos = np.append(np.array(methods)[0:4], 'Best Non-RiskLoss')
                #
                # risks_values_positions = np.argsort(risks_values)
                # if current != 'wins':
                #     risks_values_positions = risks_values_positions[::-1]
                # risks_values = np.array(risks_values)[risks_values_positions]
                # nomes_metodos_ordenados = nomes_metodos[risks_values_positions]
                #
                # heights.extend(risks_values)
                # names_methods_heights.extend(nomes_metodos_ordenados)
