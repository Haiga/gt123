'''
Produces rankings for algorithms.
'''

# Avaliar Usuários segmentados:
# - Criar arquivos identificando os usuários para cada métrica (NDCG, EPD, EILD)
# -- abaixo da média / acima da média
# -- empates / vitória / derrota contra para Risk (IndSUM) em relaçã a mf-PEH (IndSUM)
import os
import sys, glob

# devem estar na ordem em que aparecem nos arquivos
import numpy as np

measures = ['NDGC-10', 'NDCG-5']

# Algoritmos avaliados, em ordem
algorithms = ['Risk (IndSUM)', 'Risk (SUM)',
              'Risk (IndRisk)', 'Risk (Risk)',
              'SO-Risk', 'SO-Rank',
              'mf-PEH (IndRisk)', 'mf-PEH (IndSUM)', 'mf-PEH (SUM)']


def getAlgName(fileName):
    algName = ''
    # MO Strategy
    if 'bpr' in fileName:
        return 'BPR'
    if 'ncf' in fileName:
        return 'NCF'

    if 'SO_RANK_' in fileName:
        algName += 'SO-Rank'
    elif 'SO_RISK_' in fileName:
        algName += 'SO-Risk'
    elif 'SO_RISK-RANK_' in fileName:
        algName += 'SO-RiskRank'
    elif 'RISK_' in fileName:
        algName += 'Risk'
    elif 'RISK-RANK_' in fileName:
        algName += 'Risk*O'
    else:
        algName += 'mf-PEH'
    # Decision strategy
    if 'Ind-GeoRisk' in fileName:
        algName += ' (IndRisk)'
    elif 'Ind-SUM' in fileName:
        algName += ' (IndSUM)'
    elif 'GeoRisk' in fileName:
        algName += ' (Risk)'
    elif 'SUM' in fileName:
        algName += ' (SUM)'
    return algName


def readFromFile(fileName):
    print(f'- readFromFile: {fileName}')
    file = open(fileName, 'r')
    rawData = {}
    file.readline()  # ignoring header
    uid = 0
    for line in file:
        line = line.strip().replace(',', '.')
        values = line.split()
        # uid = int(values[0])
        rawData[uid] = [float(v) for v in values[0:2]]
        uid += 1
    return rawData


def getData(home, files, baseline):
    diffs = {}

    baselineValues = readFromFile(home + "/" + baseline)

    for file in files:
        alg = file.split("\\")[-1]
        if baseline == alg: continue
        if "ic-file" in alg: continue
        if "losses-wins" in alg: continue
        if alg not in diffs:
            diffs[alg] = {}
            ##
            diffs[alg]['loss_g_20'] = {}
            for i in range(len(measures)):
                diffs[alg]['loss_g_20'][i] = 0
            ##
        algValues = readFromFile(file)
        for uid in algValues:
            if uid not in baselineValues:
                continue
            if uid not in diffs[alg]: diffs[alg][uid] = []
            for i in range(len(measures)):
                ##
                size_diff = algValues[uid][i] - baselineValues[uid][i]
                if size_diff < 0:
                    p = ((size_diff * -1) * 100) / baselineValues[uid][i]
                    if p > 20:
                        diffs[alg]['loss_g_20'][i] += 1
                ##
                diffs[alg][uid].append(algValues[uid][i] - baselineValues[uid][i])
    return diffs


def processLossesWins(data):
    histogram = {}
    mean = {}
    for alg in data:
        histogram[alg] = {}
        mean[alg] = {}
        for m in measures: histogram[alg][m] = [0, 0]  # losses x wins
        for m in measures: mean[alg][m] = [0, 0]  # losses x wins
        for uid in data[alg]:
            if uid == 'loss_g_20': continue
            for m in range(len(measures)):
                if data[alg][uid][m] < 0:
                    histogram[alg][measures[m]][0] += 1
                    mean[alg][measures[m]][0] += abs(data[alg][uid][m])
                elif data[alg][uid][m] > 0:
                    histogram[alg][measures[m]][1] += 1
                    mean[alg][measures[m]][1] += data[alg][uid][m]
        for m in range(len(measures)):
            mean[alg][measures[m]][0] /= histogram[alg][measures[m]][0]
            mean[alg][measures[m]][1] /= histogram[alg][measures[m]][1]
    return histogram, mean, data


def saveFile(outFile, data, mean, diff):
    file = open(f'{outFile}.tsv', 'w')
    algs = []
    for alg in data:
        algs.append(alg)
    # for alg in algorithms:
    #     if alg in data: algs.append(alg)
    algsStr = '\t\t'.join(algs)
    file.write(f'Criterion\tResult\t{algsStr}\n')
    for m in range(len(measures)):
        file.write(f'{measures[m]}\tWins')
        for alg in algs:
            losses = data[alg][measures[m]][0]
            wins = data[alg][measures[m]][1]
            if wins != 0:
                percent = 100 * (wins - losses) / losses
            else:
                percent = 0
            file.write((f'\t{wins}\t{percent}').replace('.', ','))
        file.write('\n\tLosses')
        for alg in algs:
            losses = data[alg][measures[m]][0]
            file.write(f'\t{losses}\t')
        file.write('\n\tImprovement')
        for alg in algs:
            degrad = mean[alg][measures[m]][0]
            improv = mean[alg][measures[m]][1]
            percent = 100 * (improv - degrad) / degrad
            file.write((f'\t{improv}\t{percent}').replace('.', ','))
        file.write('\n\tDegradation')
        for alg in algs:
            degrad = mean[alg][measures[m]][0]
            file.write((f'\t{degrad}\t').replace('.', ','))
        ####
        file.write('\n\tPerda > 20%')
        for alg in algs:
            d = diff[alg]['loss_g_20'][m]
            file.write((f'\t{d}\t').replace('.', ','))
        ####
        file.write('\n')
    file.close()


if __name__ == '__main__':
    print("Inicio")

    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1003\\resultados-web10k-completo\\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1203\\resultados-web10k-completo2\\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-2302\\analise1"

    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1203\\resultados-web10k-completo2\\results"

    # home = "D:\\Colecoes\\experimento_loss_risk\\grouped\\yahoo\\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\grouped\\web10k\\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1703-losswins\\yahoo\\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1703-losswins\\ml20m\\results"
    # home = r"D:\Colecoes\experimento_loss_risk\geral\web10k\results"
    home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\resumo-external-completo\results"
    home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-mq2007-final-att2"
    home = r"D:\Colecoes\experimento_loss_risk\tuned-multilayer\resultados-web10k-multilayer8\results"
    home = r"D:\Colecoes\experimento_loss_risk\temp1\resultados-mq2007-final-mlp"
    home = r"D:\Colecoes\experimento_loss_risk\geral\web10k\results"
    home = r"D:\Colecoes\experimento_loss_risk\geral\yahoo\results"
    home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-datay-final-mlp2"#changed after ...
    home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\regularizer\resultados-web10k-regularizer\results - Copia"

    # home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-mq2007-final-mlp2"
    # home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-datay-final-mlp2"
    # home = r"D:\Colecoes\experimento_loss_risk\geral\ml20m\results"
    # home = r"D:\Colecoes\experimento_loss_risk\geral\yahoo\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\execucao-1703-losswins\\web10k\\results"
    # home = "D:\\Colecoes\\experimento_loss_risk\\grouped\\ml20m\\results"
    home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-2\web10k'
    home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-2\yahoo-2'
    # baseline = "pointwise_rmse.tsv"
    # baseline = "ordinal.tsv"
    # baseline = "spearmanLoss.tsv"
    baseline = "summed.tsv"
    # baseline = "geoRiskListNetLoss.tsv"

    files = glob.glob(home + "/*.tsv")
    cont = 0

    for file in files:
        alg = file.split("\\")[-1]
        if baseline == alg: continue
        if "ic-file" in alg: continue
        if "losses-wins" in alg: continue

        cont += 1
        fileb = open(file)
        header = fileb.readline()  # ignoring header
        summs = []
        for line in fileb:
            line = line.strip().replace(',', '.')
            values = line.split()
            # uid = int(values[0])
            summs.append([float(v) for v in values[0:2]])
        if cont == 1:
            summs2 = np.array(summs)
        else:
            summs2 = summs2 + np.array(summs)
        fileb.close()
    summs2 = summs2 / cont
    with open(home + "/" + baseline, "w") as fo:
        fo.write(header.replace("\n", "") + "\n")
        for line in summs2:
            fo.write(str(line).replace("]", "").replace("[", "").replace(",", "\t").strip(" ") + "\n")
    outFile = f'{home}/losses-wins'

    data = getData(home, files, baseline)
    data, mean, diff = processLossesWins(data)
    saveFile(outFile, data, mean, diff)
    print("Fim")
