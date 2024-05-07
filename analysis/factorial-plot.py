import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from matplotlib import rc

# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results"
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

# plt.boxplot()
y = []
names = []
positions = []
p = 0
counts = []
positions_labels = []


def my_sort(lista):
    if lista == ["10", "2", "5"]:
        return ["2", "5", "10"]
    return sorted(lista)

infos.pop('Loss')
for factor in infos:
    start = p
    # print(infos[factor].keys())
    print(my_sort(list(infos[factor].keys())))
    for lvl in my_sort(list(infos[factor].keys())):
        y.append(infos[factor][lvl]["values"])
        names.append(factor + "-" + lvl)
        positions.append(p)
        counts.append(0)
        end = p
        p += 1
    positions_labels.append((end + start) / 2)
    counts.append(1)
    p += 1.5
# ax = sns.boxplot(data=y)
fig, ax = plt.subplots()
bp = ax.boxplot(y, positions=positions, patch_artist=True)

# for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
for element in ['boxes', 'fliers', 'means', 'medians', 'caps', 'whiskers']:
    plt.setp(bp[element], color='k')

# colors = ['#505050', '#787878', '#C0C0C0']
colors = ['#505050', '#C0C0C0', '#787878']
cunt = 0
index = 0

for box in bp['fliers']:
    box.set(color='k', linewidth=0.05, markersize=1.5)

for box in bp['boxes']:
    if counts[cunt] == 1:
        cunt += 1
        index = 0

    colour = colors[index]
    # change outline color
    box.set(color='k', linewidth=1)
    # change fill color
    box.set(facecolor=colour)
    # change hatch
    # box.set(hatch='/')
    index += 1
    cunt += 1

plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handleheight'] = 1.125

# colors = ['#505050', '#787878', '#C0C0C0']
p1 = patches.Patch(color=colors[0], label=r'\textit{AdaptedLambda}')
p2 = patches.Patch(color=colors[1], label=r'\textit{AdaptedListNet}')
p3 = patches.Patch(color=colors[2], label='Spearman')

p4 = patches.Patch(color=colors[0], label=r'\textit{Default-Risk}')
p5 = patches.Patch(color=colors[1], label=r'\textit{Diff-risk-ideal}')
p6 = patches.Patch(color=colors[2], label=r'\textit{Diff-risk-ideal} $^2$')

p7 = patches.Patch(color=colors[0], label='2')
p8 = patches.Patch(color=colors[1], label='5')
p9 = patches.Patch(color=colors[2], label='10')

p10 = patches.Patch(color=colors[0], label='Cosseno')
p11 = patches.Patch(color=colors[1], label='Spearman')

p12 = patches.Patch(color=colors[0], label='Não')
p13 = patches.Patch(color=colors[1], label='Sim')

v = -0.2
legends = [
    # plt.legend(handles=[p1, p2, p3], loc=1, bbox_to_anchor=(0.23, 1), prop={'size': 8}),
    # plt.legend(handles=[p1, p2], loc=1, bbox_to_anchor=(0.23, 1), prop={'size': 8}),
    # plt.legend(handles=[p1], loc=1, bbox_to_anchor=(0.23, 1), prop={'size': 8}),
    plt.legend(handles=[p4, p5, p6], loc=1, bbox_to_anchor=(0.44 + v, 1), prop={'size': 8}),
    plt.legend(handles=[p7, p8, p9], loc=1, bbox_to_anchor=(0.65 + v, 1), prop={'size': 8}),
    plt.legend(handles=[p10, p11], loc=1, bbox_to_anchor=(0.94 + v, 1), prop={'size': 8}),
    plt.legend(handles=[p12, p13], loc=1, bbox_to_anchor=(1.125 + v, 1), prop={'size': 8}),
    # plt.legend(handles=[blue_patch, blue_patch], loc=1, bbox_to_anchor=(0.6, 0.5)),
]
for l in legends:
    plt.gca().add_artist(l)

# plt.xticks(positions, names, rotation=90)
# nn = [r"\textit{Effectiveness-Loss}", "Retorno", r'$\alpha$', 'Correlação',
#       r"Usa \textit{dropout}" + "\n" + r"como \textit{baseline}"]
nn = ["Retorno", r'$\alpha$', 'Correlação',
      r"Usa múltiplas" + "\n" + r"predições como" +"\n"+ r"\textit{baselines}"]
plt.xticks(positions_labels, nn)
print(positions_labels)

plt.ylim((0.75, 0.91))
# plt.ylim((0.28, 0.53))
plt.xlim((-1, 15))
plt.yticks((0.75, 0.80, 0.85, 0.9))
# plt.yticks((0.30, 0.35, 0.4, 0.45, 0.50))
plt.ylabel("NDCG@10")

plt.tight_layout()
# plt.show()
plt.savefig("factor.png", dpi=200)
# plt.savefig("factor_web10k.png", dpi=200)

# g = sns.catplot(y="lndcg10", data=data, kind="box",height=4, aspect=.7)
#
# plt.show()
