import pandas as pd
import numpy as np
import scipy.stats

# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results"
home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-ml5k\\results"
datafile = home + "\\" + "resumo.tsv"
# data = pd.read_csv(datafile, sep="\t")

f = open(datafile)
lines = f.readlines()[1:]

cols = ["Loss", "typeret", "alpha", "correl", "usebaseline", "lndcg10"]
infos = {}
for col in cols:
    infos.setdefault(col, {})
infos.pop(cols[-1])
y = []
for line in lines:
    h = line.replace("\n", "").split("\t")
    if h[0] == "geoRiskSpearman": continue
    if h[0] == "geoRiskLambda": continue
    y.append(float(h[-1]))
    for j in range(len(h) - 1):
        if h[j] not in infos[cols[j]]:
            infos[cols[j]][h[j]] = {}
            infos[cols[j]][h[j]]["sum"] = 0
            infos[cols[j]][h[j]]["tot"] = 0
        infos[cols[j]][h[j]]["sum"] += float(h[-1])
        infos[cols[j]][h[j]]["tot"] += 1

for fact in infos:
    for level in infos[fact]:
        infos[fact][level]['mean'] = infos[fact][level]['sum'] / infos[fact][level]['tot']
global_mean = np.mean(y)

for fact in infos:
    for level in infos[fact]:
        infos[fact][level]['efeito'] = infos[fact][level]['mean'] - global_mean

cont = 0
erro = []
y_hat = []
for line in lines:
    h = line.replace("\n", "").split("\t")
    if h[0] == "geoRiskSpearman": continue
    if h[0] == "geoRiskLambda": continue
    # y_hat.append(float(h[-1]))
    soma = global_mean
    for j in range(len(h) - 1):
        soma += infos[cols[j]][h[j]]["efeito"]
    y_hat.append(soma)
    erro.append(y[cont] - soma)
    cont += 1

tam = len(y)

print(tam)

y = np.array(y)
y_hat = np.array(y_hat)
erro = np.array(erro)

sse = np.sum(erro * erro)
ssy = np.sum(y * y)

sso = tam * global_mean * global_mean

sst = ssy - sso
mse = sse / tam

percent_explicavel = 0
sum_ss = 0
infos.pop('Loss')
att = infos.keys()

informations = {}
for fact in att:
    informations[fact] = {}
    q = 0
    for level in infos[fact]:
        q += (tam / len(infos[fact])) * infos[fact][level]["efeito"] * infos[fact][level]["efeito"]
    informations[fact]["ss"] = q
    informations[fact]["ms"] = q / (len(infos[fact]) - 1)
    informations[fact]["ms/mse"] = informations[fact]["ms"] / mse
    informations[fact]["f"] = scipy.stats.f.cdf(0.95, len(infos[fact]) - 1, tam)
    informations[fact]["sig"] = informations[fact]["ms/mse"] > informations[fact]["f"]
    informations[fact]["perc_exp"] = q / sst
    percent_explicavel += q / sst
    sum_ss += q

print(informations)
