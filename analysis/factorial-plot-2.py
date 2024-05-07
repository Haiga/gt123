import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from matplotlib import rc

home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-ml5k\\results"
datafile = home + "\\" + "resumo.tsv"
data = pd.read_csv(datafile, sep="\t")

f = open(datafile)
lines = f.readlines()[1:]

rc('text', usetex=True)

cols = ["Loss", "typeret", "alpha", "correl", "usebaseline", "lndcg10"]
infos = {}
for col in cols:
    infos.setdefault(col, {})
infos.pop(cols[-1])
u = []
for line in lines:
    h = line.replace("\n", "").split("\t")

    for j in range(len(h) - 1):
        if h[0] == "geoRiskSpearman": continue
        if h[0] == "geoRiskLambda": continue
        if h[j] not in infos[cols[j]]:
            infos[cols[j]][h[j]] = {}
            infos[cols[j]][h[j]]["sum"] = 0
            infos[cols[j]][h[j]]["tot"] = 0
            infos[cols[j]][h[j]]["values"] = []
        infos[cols[j]][h[j]]["sum"] += float(h[-1])
        infos[cols[j]][h[j]]["values"].append(float(h[-1]))
        infos[cols[j]][h[j]]["tot"] += 1
        u.append(float(h[-1]))

home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-ml5k\\results"
datafile = home + "\\" + "resumo.tsv"
data = pd.read_csv(datafile, sep="\t")

f = open(datafile)
lines = f.readlines()[1:]

rc('text', usetex=True)

cols = ["Loss", "typeret", "alpha", "correl", "usebaseline", "lndcg10"]
infos = {}
for col in cols:
    infos.setdefault(col, {})
infos.pop(cols[-1])
u2 = []
for line in lines:
    h = line.replace("\n", "").split("\t")

    for j in range(len(h) - 1):
        if h[0] == "geoRiskSpearman": continue
        if h[0] == "geoRiskLambda": continue
        if h[j] not in infos[cols[j]]:
            infos[cols[j]][h[j]] = {}
            infos[cols[j]][h[j]]["sum"] = 0
            infos[cols[j]][h[j]]["tot"] = 0
            infos[cols[j]][h[j]]["values"] = []
        infos[cols[j]][h[j]]["sum"] += float(h[-1])
        infos[cols[j]][h[j]]["values"].append(float(h[-1]))
        infos[cols[j]][h[j]]["tot"] += 1
        u2.append(float(h[-1]))
plt.boxplot([u, u2])
plt.show()