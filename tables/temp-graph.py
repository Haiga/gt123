import glob
import numpy
from analysis.l2rmeasures import getGeoRisk
from createLossesWinsFiles2 import getData, processLossesWins
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tables.createLossesWinsFiles2 import doIt

# for dataset in ['mq2007', 'yahoo', 'web10k']:
# for dataset in ['web10k','mq2007', 'yahoo']:
#     for tt in ['att', 'mlp']:
for dataset in ['web10k']:
    for tt in ['att']:

        home = r"D:\Colecoes\experimento_loss_risk\tables\overall"+f"\\{tt}"+f"\\{dataset}"

        methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss', 'lambdaLoss',
                   'listNet', 'ordinal', 'pointwise_rmse', 'spearmanLoss', 'grisklmart']
        if tt == 'mlp':
            methods = ['extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss', 'geoRiskListnetLoss', 'geoRiskSpearmanLoss',
                       'lambdaLoss', 'listNet', 'ordinal', 'pointwise_rmse', 'spearmanLoss']
        metrics = ['lndcg_10', 'lndcg_5']
        # metrics = ['ndcg_10', 'ndcg_5']

        print("Inicio")

        # home = r"D:\Colecoes\experimento_loss_risk\tables\overall\att\mq2007"

        baseline = "summed.tsv"

        files = glob.glob(home + "/*.tsv")

        if home + '\\' + 'grisklmart.tsv' in files:
            files.remove(home + '\\' + 'grisklmart.tsv')
            files.append(home + '\\' + 'grisklmart.tsv')

        data = getData(home, files, baseline)
        data, mean, diff = processLossesWins(data)

        cont = 0
        metric = 'NDGC-10'
        bestwin = 0
        width = 0.1
        # mm = 1#1 maximize
        mm = 0
        if mm == 0:
            bestwin = 10000000
        elif mm == 1:
            bestwin = 0

        ws = []
        ps = [cont]
        for method in methods:
            if 'geoRisk' not in method:
                if mm == 0:
                    bestwin = min([bestwin, data[method + '.tsv'][metric][mm]])
                elif mm == 1:
                    bestwin = max([bestwin, data[method + '.tsv'][metric][mm]])
                # bestwin = max([bestwin, data[method + '.tsv'][metric][1]])
                # bestwin = min([bestwin, data[method + '.tsv'][metric][1]])
            else:
                # wins = data[method + '.tsv'][metric][1]
                # losses = data[method + '.tsv'][metric][0]
                # plt.bar(cont, data[method + '.tsv'][metric][mm], width)
                ws.append(data[method + '.tsv'][metric][mm])
                cont += width
                ps.append(cont)
        # sorted(ws)
        # fig, ax = plt.subplots(figsize=(10,5))
        fig, ax = plt.subplots()
        ws = np.array(ws)
        pos_order = np.argsort(ws)
        if mm == 0:
            pos_order = pos_order[::-1]
        names = np.array(methods)[0:4][pos_order]
        colours = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
        colours = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
        tt = 1
        if tt == 1:
            ax.barh(ps, np.append(ws[pos_order], bestwin), width, color=colours,  edgecolor='white', linewidth=8)
        else:
            ax.bar(ps, np.append(ws[pos_order], bestwin), width, color=colours, edgecolor='white', linewidth=8)
        # plt.bar(ps, np.append(np.sort(ws), bestwin), width, color=colours,  edgecolor='black')

        ax.bar_label(ax.containers[0], padding=-4)
        # plt.bar(cont, bestwin, 0.1)
        names2 = []
        for name in names:
            r = ''
            if 'Spear' in name:
                r += 'GRriskSP'
            if 'List' in name:
                r += 'GRriskCS'
            if 'ext' in name:
                r += '+EB'
            names2.append(r)
        if tt == 1:
            ax.set_xlim(right=6900)
            ax.set_yticks(ps)
            ax.set_yticklabels(np.append(names2, 'Best Non-RS'))
        else:
            ax.set_ylim(top=6900)
            ax.set_xticks(ps)
            ax.set_xticklabels(np.append(names2, 'Best Non-RS'))
        # mpl.rc('hatch', color='w', linewidth=4.5)

        #######################

        cont = 0
        metric = 'NDGC-10'
        bestwin = 0
        width = 0.1
        # mm = 1#1 maximize
        mm = 1
        if mm == 0:
            bestwin = 10000000
        elif mm == 1:
            bestwin = 0

        ws = []
        ps = [cont]
        for method in methods:
            if 'geoRisk' not in method:
                if mm == 0:
                    bestwin = min([bestwin, data[method + '.tsv'][metric][mm]])
                elif mm == 1:
                    bestwin = max([bestwin, data[method + '.tsv'][metric][mm]])
                # bestwin = max([bestwin, data[method + '.tsv'][metric][1]])
                # bestwin = min([bestwin, data[method + '.tsv'][metric][1]])
            else:
                # wins = data[method + '.tsv'][metric][1]
                # losses = data[method + '.tsv'][metric][0]
                # plt.bar(cont, data[method + '.tsv'][metric][mm], width)
                ws.append(data[method + '.tsv'][metric][mm])
                cont += width
                ps.append(cont)
        # sorted(ws)
        # fig, ax = plt.subplots(figsize=(10,5))
        # fig, ax = plt.subplots()
        ws = np.array(ws)
        pos_order = np.argsort(ws)
        if mm == 0:
            pos_order = pos_order[::-1]
        names = np.array(methods)[0:4][pos_order]
        colours = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
        colours = ['#ff7f0e', '#ff7f0e', '#ff7f0e', '#ff7f0e', '#ff7f0e']
        tt = 1
        ps = ps[-1] + np.array(ps) + width
        if tt == 1:
            ax.barh(ps, np.append(ws[pos_order], bestwin), width, color=colours, edgecolor='white', linewidth=8)
        else:
            ax.bar(ps, np.append(ws[pos_order], bestwin), width, color=colours, edgecolor='white', linewidth=8)
        # plt.bar(ps, np.append(np.sort(ws), bestwin), width, color=colours,  edgecolor='black')

        ax.bar_label(ax.containers[0], padding=-4)
        # plt.bar(cont, bestwin, 0.1)
        names2 = []
        for name in names:
            r = ''
            if 'Spear' in name:
                r += 'GRriskSP'
            if 'List' in name:
                r += 'GRriskCS'
            if 'ext' in name:
                r += '+EB'
            names2.append(r)
        if tt == 1:
            ax.set_xlim(right=6900)
            ax.set_yticks(ps)
            ax.set_yticklabels(np.append(names2, 'Best Non-RS'))
        else:
            ax.set_ylim(top=6900)
            ax.set_xticks(ps)
            ax.set_xticklabels(np.append(names2, 'Best Non-RS'))


        plt.tight_layout()
        plt.show()
        # x = 2/0
