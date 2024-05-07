import glob
import numpy
from analysis.l2rmeasures import getGeoRisk
import numpy as np
from scipy.stats import ttest_ind

# home = r"/home/silvapedro/experimento_loss_risk/resultados-web10k-mlp-eb-exec1/results"
from tables.createLossesWinsFiles2 import doIt

# home = r'D:\Colecoes\experimento_loss_risk\tables\overall\att\mq2007'
home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\yahoo'
# home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\mlp\yahoo'

# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'ordinal',
#            'pointwise_rmse']
# methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'extgeoRiskListnetLoss', 'spearmanLoss', 'lambdaLoss', 'listNet', 'ordinal',
#            'pointwise_rmse', 'grisklmart']
methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss',
           # methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'lambdaLoss',
           'lambdaLoss', 'listNet', 'ordinal', 'pointwise_rmse', 'spearmanLoss',
           'lambdaLossmulti', 'listNetmulti', 'ordinalmulti', 'pointwise_rmsemulti', 'spearmanLossmulti0.1',
           'grisklmart']

methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss',
           # methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'lambdaLoss',
           'grisklmart', 'lmart',
           'lambdaLoss', 'lambdaLossmulti', 'listNet', 'listNetmulti', 'ordinal', 'ordinalmulti',
           'pointwise_rmse', 'pointwise_rmsemulti', 'spearmanLoss',
           'spearmanLossmulti',
           ]

# methods = ['extgeoRisk*', 'geoRisk*',
#            # methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'lambdaLoss',
#            # 'grisklmart', 'lmart',
#            'lambdaLoss', 'lambdaLossmulti', 'listNet', 'listNetmulti', 'ordinal', 'ordinalmulti',
#            'pointwise_rmse', 'pointwise_rmsemulti', 'spearmanLoss',
#            'spearmanLossmulti',
#            ]
# methods = ['extgeoRisk*', 'geoRisk*',
#            # methods = ['geoRiskSpearmanLoss', 'geoRiskListnetLoss', 'lambdaLoss',
#            'grisklmart', 'lmart',
#            'lambdaLossmulti', 'listNetmulti', 'ordinalmulti',
#            'pointwise_rmse',
#            'spearmanLossmulti',
#            ]
# 'listNet', 'spearmanLoss', 'ordinal', 'pointwise_rmse']
metrics = ['lndcg_10', 'lndcg_5']
# metrics = ['ndcg_10', 'ndcg_5']
reps = [1]
# reps = [1, 2, 3, 4, 5]
do_sorted = False
alpha = 4
# last = True
subs = ["ListnetLoss", "SpearmanLoss"]


# subs = ["SpearmanLoss"]


# subs = ["SpearmanLoss"]

# do_sorted = True

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
                    for sub in subs:
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
                    if "lambdaLossmulti" in method or "ordinalmulti" in method or "spearmanLossmulti" in method:
                        path = home + "/" + fold_name + "/model2.predict." + metric + ".txt"
                    else:
                        path = home + "/" + fold_name + "/model.predict." + metric + ".txt"
                    r = read_lines(path)

                mat.append(r)
                size = len(r)
                methods_dics[method][metric].extend(r)
            # print(rep)
            cont = 0
            georisk = getGeoRisk(np.array(mat).T, alpha=alpha)
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

# r = ttest_ind(methods_dics[methods[3]]["georisk5" +metrics[0]], methods_dics[methods[-1]]["georisk5" +metrics[0]])
# print(r)
#
# print(methods[-1])
# print(methods[3])
# x =2/0
if "extgeoRisk" in methods:
    vvs = ['extgeoRisk*', 'geoRisk*']
else:
    vvs = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss']
for vv in vvs:
    print(vv)
    for method in methods:
        if method in vvs:
            continue
        else:
            print(method, end="\t")
            for metric in metrics:
                v, pvalue = ttest_ind(methods_dics[vv][metric], methods_dics[method][metric])
                # if pvalue < 0.05 and v > 0:
                if pvalue < 0.05 and np.mean(methods_dics[vv][metric]) > np.mean(methods_dics[method][metric]):
                    print("x\t", end="")
                else:
                    print("*\t", end="")
        print("")

for method in methods:
    path = home + "/" + method + ".tsv"
    with open(path.replace("*", ""), "w") as fo:
        for metric in metrics:
            fo.write(metric + "\t")
        fo.write("\n")
        for i in range(total_size):
            for metric in metrics:
                fo.write(str(methods_dics[method][metric][i]) + "\t")
            fo.write("\n")

measures = ['NDCG-10', 'NDCG-5', 'GeoRisk-NDCG-10', 'GeoRisk-NDCG-5']


def rankMeasure(name):
    if name.startswith('GeoRisk'):
        return True
    else:
        return True


def calculateIC(rawData, h):
    result = {}
    for measure in h:
        A = 0
        Q = 0
        k = 0
        for value in rawData[measure]:
            k += 1
            oldA = A
            A += (value - A) / k
            Q += (value - oldA) * (value - A)
        if k == 0:
            result[measure] = [float('nan'), float('nan')]
        else:
            std = numpy.sqrt(Q / (k - 1))
            z = 1.96  # https://en.wikipedia.org/wiki/Confidence_interval
            # http://www.dummies.com/education/math/statistics/checking-out-statistical-confidence-interval-critical-values/
            CI = z * (std / numpy.sqrt(k))
            result[measure] = [A, CI]
    return result


def getData(filePattern):
    # global ignoreIndDM, ignoreRiskDM, ignoreSumDM
    files = glob.glob(filePattern)
    data = []
    for fileName in files:
        # alg = getAlgName(fileName)
        alg = fileName.split("\\")[-1].replace(".tsv", "")
        if "ic-file" in alg:
            continue
        if "losses-wins" in alg:
            continue
        if "resumo" in alg:
            continue
        if "summed" in alg:
            continue
        # if 'Risk*O' in alg or 'SO-RiskRank' in alg: continue
        # if ignoreIndDM and '(Ind' in alg: continue
        # if ignoreRiskDM and 'Risk)' in alg: continue
        # if ignoreSumDM and 'SUM)' in alg: continue
        print(f'- getData: {fileName}')
        file = open(fileName, 'r')
        file.readline()  # removendo o cabeçalho
        rawData = {}
        for m in measures: rawData[m] = []
        for line in file:
            values = line.strip().split()
            for v in range(len(measures)):
                rawData[measures[v]].append(float(values[v].replace(',', '.').replace('-',
                                                                                      '')))  # métricas negativadas para maximização no jMetal
        icData = calculateIC(rawData, measures)
        values = [alg]
        for measure in measures:
            values.append(icData[measure])
        data.append(values)
    return data


def rankData(data):
    print('- rankData')
    newData = [d[:] for d in data]
    for col in range(1, len(measures) + 1):
        print('-- sorting by ' + measures[col - 1])
        newData.sort(key=lambda x: x[col][1], reverse=False)  # IC sempre crescente
        newData.sort(key=lambda x: x[col][0], reverse=True)  # métricas são todas maximizadas, sempre decrescente
        num = 1
        qtd = 1
        indexDiscount = 0
        (value, ic) = (newData[0][col][0], newData[0][col][1])
        for index in range(1, len(newData)):
            (newValue, newIc) = (newData[index][col][0], newData[index][col][1])
            tie = ((value - ic) - (newValue + newIc) < 0)
            if not tie:
                (value, ic) = (newValue, newIc)
                rank = num / qtd
                for i in range(index - qtd, index):
                    newData[i][col].append(rank)
                num = index - indexDiscount + 1
                qtd = 1
                if index == len(newData) - 1:
                    rank = num / qtd
                    newData[index][col].append(rank)
            else:
                num += index - indexDiscount + 1
                qtd += 1
                if index == len(newData) - 1:
                    rank = num / qtd
                    for i in range(index - qtd + 1, index + 1):
                        newData[i][col].append(rank)
    for index in range(0, len(newData)):
        sumRanks = 0
        for col in range(1, len(measures) + 1):
            if rankMeasure(measures[col - 1]): sumRanks += newData[index][col][2]
        newData[index].append(sumRanks)
    newData.sort(key=lambda x: x[0], reverse=False)
    newData.sort(key=lambda x: x[-1], reverse=False)
    return newData


def writeSorted(arq, header, data, pareto):
    arq.write('Context\tAlgorithm\t')
    arq.write('\t'.join(header))
    arq.write('\n')
    for row in data:
        arq.write('\t'.join(str(v).replace('.', ',') for v in row))
        if pareto != None:
            arq.write('\t' + '\t'.join(pareto[row[0]]))
        arq.write('\n')


def saveFile(file, data):
    print('- saveFile: ' + file)
    arq = open(file, 'w')
    arq.write('Alg')
    print('Alg', end='')
    for col in measures:
        arq.write(f'\t{col}')
        print(f'\t{col}', end='')
    # arq.write('\n')
    # for col in measures:
    #     # arq.write(f'\tMean\tIC\tRank')
    #     # print(f'\tMean\tIC\tRank', end='')
    #     arq.write(f'\tMean')
    #     print(f'\tMean', end='')
    arq.write('\tOverall Ranking\n')
    print('\tOverall Ranking\n', end='')
    # for alg in data:

    if do_sorted:
        for method in methods:
            for alg in data:
                if alg[0] == method.replace("*", ""):
                    break
            arq.write(alg[0])
            print(alg[0], end='')

            for i in range(1, len(alg) - 1):
                valueRound = round(alg[i][0], 4)
                bold = True
                for alg2 in data:
                    if alg[i][0] < alg2[i][0]:
                        bold = False
                st = ''
                if bold:
                    st = 'b'
                # arq.write((f'\t{alg[i][0]}\t{alg[i][1]}\t{alg[i][2]}').replace('.', ','))
                # print((f'\t{alg[i][0]}\t{alg[i][1]}\t{alg[i][2]}').replace('.', ','), end='')
                arq.write((f'\t' + st + f'{valueRound:7.4f} ({alg[i][2]})').replace('.', ','))
                print((f'\t' + st + f'{valueRound:7.4f} ({alg[i][2]})').replace('.', ','), end='')

            arq.write((f'\t{alg[-1]}\n').replace('.', ','))
            print((f'\t{alg[-1]}\n').replace('.', ','), end='')
        arq.close()
    else:
        for alg in data:
            arq.write(alg[0])
            print(alg[0], end='')

            for i in range(1, len(alg) - 1):
                valueRound = round(alg[i][0], 4)
                bold = True
                for alg2 in data:
                    if alg[i][0] < alg2[i][0]:
                        bold = False
                st = ''
                if bold:
                    st = 'b'
                # arq.write((f'\t{alg[i][0]}\t{alg[i][1]}\t{alg[i][2]}').replace('.', ','))
                # print((f'\t{alg[i][0]}\t{alg[i][1]}\t{alg[i][2]}').replace('.', ','), end='')
                arq.write((f'\t' + st + f'{valueRound:7.4f} ({alg[i][2]})').replace('.', ','))
                print((f'\t' + st + f'{valueRound:7.4f} ({alg[i][2]})').replace('.', ','), end='')

            arq.write((f'\t{alg[-1]}\n').replace('.', ','))
            print((f'\t{alg[-1]}\n').replace('.', ','), end='')
        arq.close()


outFile = f'{home}/{"ic-file-only.tsv"}'

data = []
data.extend(getData(home + f'/*.tsv'))
rankedData = rankData(data)
saveFile(outFile, rankedData)
print("Fim")
doIt(home)
