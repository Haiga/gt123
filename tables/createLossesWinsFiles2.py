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
    # print(f'- readFromFile: {fileName}')
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
    algsStr = '\t'.join(algs)
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

            bold = True
            for alg2 in algs:
                if wins < data[alg2][measures[m]][1]:
                    bold = False
            st = ''
            if bold:
                st = 'b'
                file.write((f'\t{st} {wins}').replace('.', ','))
            else:
                file.write((f'\t{wins}').replace('.', ','))
        file.write('\n\tLosses')
        for alg in algs:
            losses = data[alg][measures[m]][0]

            bold = True
            for alg2 in algs:
                if -1 * losses < -1 * data[alg2][measures[m]][0]:
                    bold = False
            st = ''
            if bold:
                st = 'b'
                file.write(f'\t{st} {losses}')
            else:
                file.write(f'\t{losses}')
        file.write('\n\tImprovement')
        for alg in algs:
            degrad = mean[alg][measures[m]][0]
            improv = mean[alg][measures[m]][1]
            percent = 100 * (improv - degrad) / degrad
            bold = True
            for alg2 in algs:
                if improv < mean[alg2][measures[m]][1]:
                    bold = False
            st = ''
            if bold:
                st = 'b'
            improv2 = round(improv, 4)
            file.write((f'\t{st}{improv2:7.4f}').replace('.', ','))
        file.write('\n\tDegradation')
        for alg in algs:
            degrad = mean[alg][measures[m]][0]
            bold = True
            for alg2 in algs:
                if -1*degrad < -1*mean[alg2][measures[m]][0]:
                    bold = False
            st = ''
            if bold:
                st = 'b'
            degrad2 = round(degrad, 4)
            file.write((f'\t{st}{degrad:7.4f}').replace('.', ','))
        ####
        file.write('\n\tPerda > 20%')
        for alg in algs:
            d = diff[alg]['loss_g_20'][m]

            bold = True
            for alg2 in algs:
                if -1*d < -1*diff[alg2]['loss_g_20'][m]:
                    bold = False
            st = ''
            if bold:
                st = 'b'

                file.write((f'\t{st} {d}').replace('.', ','))
            else:
                file.write((f'\t{d}').replace('.', ','))
        ####
        file.write('\n')
    file.close()


def doIt(home):
    # print("Inicio")

    # home = r"D:\Colecoes\experimento_loss_risk\tables\overall\att\mq2007"

    baseline = "summed.tsv"

    files = glob.glob(home + "/*.tsv")
    # if home + '\\' + 'grisklmart.tsv' in files:
    #     files.remove(home + '\\' + 'grisklmart.tsv')
    #     files.append(home + '\\' + 'grisklmart.tsv')
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
    # print("Fim")
# doIt(home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-4\web10k")
# doIt(home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-4\yahoo")
# doIt(home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-4\mq2007")
# doIt(home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-4\mlp\web10k")
# doIt(home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-4\mlp\yahoo")
# doIt(home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-4\mlp\mq2007")