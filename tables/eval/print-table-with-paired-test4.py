from scipy.stats import ttest_rel, ttest_ind
import numpy as np
import pandas as pd
import os


def getName(sub):
    if 'grisklmart' in sub:
        return 'GeoRisk-LM'
    if 'lmart' in sub:
        return 'LambdaMart'


    if 'extgeoRiskListnetLoss' in sub:
        return 'RiskLoss+EB'
    if 'extgeoRiskSpearmanLoss' in sub:
        return 'RiskLoss+EB'

    if 'geoRiskListnetLoss' in sub:
        return 'RiskLoss+DO'
    if 'geoRiskSpearmanLoss' in sub:
        return 'RiskLoss+DO'


    if 'extgeoRisk' in sub:
        return 'RiskLoss+EB'
    if 'geoRisk' in sub:
        return 'RiskLoss+DO'



    if 'lambdaLoss' in sub:
        return 'NDCGLoss2++'
    if 'listNet' in sub:
        return 'ListNet'
    if 'ordinal' in sub:
        return 'Ordinal'
    if 'pointwise_rmse' in sub:
        return 'MSE'
    if 'spearmanLoss' in sub:
        return 'SpearmanLoss'



# for arch in ['', 'mlp']:
for arch in ['']:
    # for arch in ['mlp']:
    for dataset in ['yahoo', 'web10k', 'mq2007']:
    # for dataset in ['mq2007']:
        # for dataset in ['yahoo']:
        # arch = ''
        # dataset = 'yahoo'
        if arch == '':
            home = fr'D:\Colecoes\experimento_loss_risk\tables-apresent-4\{dataset}'
        else:
            home = fr'D:\Colecoes\experimento_loss_risk\tables-apresent-4\{arch}\{dataset}'

        methods_risk = ['geoRiskListnetLoss', 'geoRiskSpearmanLoss', 'extgeoRiskListnetLoss', 'extgeoRiskSpearmanLoss',
                        ]

        if not os.path.isfile(home + "/" + methods_risk[0] + '.tsv'):
            methods_risk = ['geoRisk', 'extgeoRisk',
                            ]
        # methods_def = ['lambdaLoss', 'lambdaLossmulti', 'listNet', 'listNetmulti', 'ordinal', 'ordinalmulti',
        #                'pointwise_rmse', 'pointwise_rmsemulti', 'spearmanLoss',
        #                'spearmanLossmulti', ]
        methods_def = ['lambdaLoss', 'listNet', 'ordinal',
                       'pointwise_rmse', 'spearmanLoss']

        methods_tree = ['grisklmart', 'lmart', ]
        methods_tree = ['grisklmart', ]
        if arch == 'mlp':
            methods_tree = []


        def get_arr(file, metric_index):
            if 'georisk' in metric_index:
                return np.unique(pd.read_csv(file + ".tsv", sep="\t")[metric_index].values)
            return pd.read_csv(file + ".tsv", sep="\t")[metric_index].values


        def get_best_in_metric(metric_index):
            best_mean = 0
            method_with_best_mean = ''
            for i in range(len(methods_def)):
                arr = get_arr(home + "/" + methods_def[i], metric_index)
                if np.mean(arr) > best_mean:
                    method_with_best_mean = methods_def[i]
                    best_mean = np.mean(arr)

            for i in range(len(methods_risk)):
                arr = get_arr(home + "/" + methods_risk[i], metric_index)
                if np.mean(arr) > best_mean:
                    method_with_best_mean = methods_risk[i]
                    best_mean = np.mean(arr)

            return method_with_best_mean


        # def get_best_def_in_metric(metric_index):
        #     best_mean = 0
        #     method_with_best_mean = ''
        #     for i in range(len(methods_def)):
        #         arr = get_arr(home + "/" + methods_def[i], metric_index)
        #         if np.mean(arr) > best_mean:
        #             method_with_best_mean = methods_def[i]
        #             best_mean = np.mean(arr)
        #     return method_with_best_mean

        def get_best_mean_in_metric(metric_index):
            best_mean = 0
            # method_with_best_mean = ''
            for i in range(len(methods_def)):
                arr = get_arr(home + "/" + methods_def[i], metric_index)
                if np.mean(arr) > best_mean:
                    # method_with_best_mean = methods_def[i]
                    best_mean = np.mean(arr)
            for i in range(len(methods_risk)):
                arr = get_arr(home + "/" + methods_risk[i], metric_index)
                if np.mean(arr) > best_mean:
                    # method_with_best_mean = methods_def[i]
                    best_mean = np.mean(arr)
            for i in range(len(methods_tree)):
                arr = get_arr(home + "/" + methods_tree[i], metric_index)
                if np.mean(arr) > best_mean:
                    # method_with_best_mean = methods_def[i]
                    best_mean = np.mean(arr)
            return best_mean


        with open('tables-paired-test4/' + arch + '-' + dataset + '.tsv', 'w+') as fo:
            for i in range(len(methods_risk)):
                fo.write(getName(methods_risk[i]) + "\t")
                for metric in ['lndcg_10', 'lndcg_5', 'georisk5lndcg_10', 'georisk5lndcg_5']:
                    arr_risk = get_arr(home + "/" + methods_risk[i], metric)
                    # method_best = get_best_def_in_metric(metric)
                    method_best = get_best_in_metric(metric)
                    print(method_best)
                    arr_def = get_arr(home + "/" + method_best, metric)

                    fo.write("%.4f" % np.mean(arr_risk))
                    # if np.mean(arr_risk) > np.mean(arr_def):
                    if True:
                        if ttest_rel(arr_risk, arr_def)[1] > 0.05:
                            fo.write("*")

                    if len(methods_tree) > 0:
                        arr_glmart = get_arr(home + "/" + methods_tree[0], metric)
                        # arr_lmart = get_arr(home + "/" + methods_tree[1], metric)
                        arr_lmart = get_arr(home + "/" + methods_tree[0], metric)#############
                        # if np.mean(arr_risk) > np.mean(arr_glmart):
                        if True:
                            if ttest_rel(arr_risk, arr_glmart)[1] < 0.05:
                                fo.write("+")

                        # if np.mean(arr_risk) > np.mean(arr_lmart):
                        if True:
                            if ttest_rel(arr_risk, arr_lmart)[1] < 0.05:
                                fo.write("^")
                    if np.mean(arr_risk) == get_best_mean_in_metric(metric):
                        fo.write("* b")
                    fo.write("\t")
                fo.write("\n")

            for i in range(len(methods_def)):

                write = True
                # if i % 2 == 0:
                #     if np.mean(get_arr(home + '/' + methods_def[i + 1], 'lndcg_10')) > np.mean(
                #             get_arr(home + '/' + methods_def[i], 'lndcg_10')):
                #         write = False
                # else:
                #     if np.mean(get_arr(home + '/' + methods_def[i - 1], 'lndcg_10')) > np.mean(
                #             get_arr(home + '/' + methods_def[i], 'lndcg_10')):
                #         write = False
                if write:
                    fo.write(getName(methods_def[i]) + "\t")
                    for metric in ['lndcg_10', 'lndcg_5', 'georisk5lndcg_10', 'georisk5lndcg_5']:
                        method_best = get_best_in_metric(metric)
                        arr_risk = get_arr(home + "/" + methods_def[i], metric)
                        fo.write("%.4f" % np.mean(arr_risk))

                        arr_def = get_arr(home + "/" + method_best, metric)
                        if True:
                            if ttest_rel(arr_risk, arr_def)[1] > 0.05:
                                fo.write("*")
                        if method_best == methods_def[i]:
                            fo.write("*")

                        if np.mean(arr_risk) == get_best_mean_in_metric(metric):
                            fo.write(" b")

                        if len(methods_tree) > 0:
                            arr_glmart = get_arr(home + "/" + methods_tree[0], metric)
                            # arr_lmart = get_arr(home + "/" + methods_tree[1], metric)######
                            arr_lmart = get_arr(home + "/" + methods_tree[0], metric)
                            # if np.mean(arr_risk) > np.mean(arr_glmart):
                            if True:
                                if ttest_rel(arr_risk, arr_glmart)[1] < 0.05:
                                    fo.write("+")

                            # if np.mean(arr_risk) > np.mean(arr_lmart):
                            if True:
                                if ttest_rel(arr_risk, arr_lmart)[1] < 0.05:
                                    fo.write("^")
                        fo.write("\t")
                    fo.write("\n")

            for i in range(len(methods_tree)):
                fo.write(getName(methods_tree[i]) + "\t")
                for metric in ['lndcg_10', 'lndcg_5', 'georisk5lndcg_10', 'georisk5lndcg_5']:
                    arr_risk = get_arr(home + "/" + methods_tree[i], metric)
                    fo.write("%.4f" % np.mean(arr_risk))
                    if np.mean(arr_risk) == get_best_mean_in_metric(metric):
                        fo.write(" b")
                    fo.write("\t")
                fo.write("\n")
