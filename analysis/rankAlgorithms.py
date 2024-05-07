'''
Produces rankings for algorithms.
'''

import sys, os, glob, numpy

# Lê todos os arquivos .tsv de uma pasta (cada arquivo contendo colunas de métricas)
# Cada um desses arquivos representa uma configuração de execução
# Ordena essas configurações com cálculo estatístico
# Rodar após executar o arquivo merge.py ou merge-normal.py

# devem estar na ordem em que aparecem nos arquivos
measures = ['NDCG-10', 'NDCG-5', 'GeoRisk-NDCG-10', 'GeoRisk-NDCG-5']


# measures = ['NDCG-10', 'NDCG-5', 'GeoRisk-NDCG-10']
# measures = ['NDCG-10', 'NDCG-5']
# measures = ['NDCG-10']


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
    for col in measures:
        arq.write(f'\t{col}\t\t')
    arq.write('\n')
    for col in measures:
        arq.write(f'\tMean\tIC\tRank')
    arq.write('\tOverall Ranking\n')
    for alg in data:
        arq.write(alg[0])
        for i in range(1, len(alg) - 1):
            arq.write((f'\t{alg[i][0]}\t{alg[i][1]}\t{alg[i][2]}').replace('.', ','))
        arq.write((f'\t{alg[-1]}\n').replace('.', ','))
    arq.close()


if __name__ == '__main__':
    print('Parameters: ' + str(sys.argv))
    print("Inicio")

    # home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\resumo\results"
    # home = r"D:\Colecoes\experimento_loss_risk\neuralgrisklmart\resultados-web10k-lmartgrisk-exec-2\results"
    # home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\dropout-geral"
    # home = r"D:\Colecoes\experimento_loss_risk\dropout-exec\resumo-external-completo\results"
    # home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\multilayer\resultados-web10k-multilayer2\results"
    home = r"D:\Colecoes\experimento_loss_risk\tuned-datay-mq2007\resultados-datay-mlp-tuned\results"
    home = r"D:\Colecoes\experimento_loss_risk\tuned-datay-mq2007\resultados-mq2007-mlp-tuned\results"
    home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-mq2007-final-att2"
    home = r"D:\Colecoes\experimento_loss_risk\temp1\resultados-mq2007-final-mlp"
    home = r"D:\Colecoes\experimento_loss_risk\geral\yahoo\results"
    home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-datay-final-mlp2"
    home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\regularizer\resultados-web10k-regularizer\results - Copia2"
    # home = r"/home/silvapedro/experimento_loss_risk/resultados-w1-endpoint/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w1-endpoint2/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3-endpoint/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3-endpoint2/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3-endpoint3/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-y-endpoint2-externalbaseline/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-mq2007-startpoint-14/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3-endpoint3/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3c-startpoint/results"
    # home = r"/home/silvapedro/experimento_loss_risk/resultados-y-endpoint3-externalbaseline/results"
    # home = r"/home/silvapedro/experimento_loss_risk/resultados-mq2007-endpoint-externalbaseline/results"
    # home = r"/home/silvapedro/experimento_loss_risk/resultados-w30k-temp/results"
    # home = r"/home/silvapedro/experimento_loss_risk/resultados-y-endpoint2/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w30k-mlp-temp/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w30k-temp/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-mq2007-endpoint2-externalbaseline/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-mq2007-startpoint2-14/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-y-endpoint-mlp-eb-2/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-mq2007-endpoint3-mlp-eb-0/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3-endpoint4/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-w3-endpoint-full2/results"
    home = r"/home/silvapedro/experimento_loss_risk/resultados-web10k-mlp-eb-exec1/results"
    # home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-mq2007-final-mlp2"
    # home = r"D:\Colecoes\experimento_loss_risk\temp\resultados-datay-final-mlp2"
    # home = r"D:\Colecoes\experimento_loss_risk\reg-multilayer\regularizer\resultados-web10k-regularizer\results"
    outFile = f'{home}/{"ic-file-only.tsv"}'

    data = []
    # data.extend(getData(home + f'\\*.tsv'))
    data.extend(getData(home + f'/*.tsv'))
    rankedData = rankData(data)
    saveFile(outFile, rankedData)
    print("Fim")
