import itertools

import pandas as pd
import statsmodels.api as sm
import statsmodels
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
# import itertools


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov


home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results"
datafile = home + "\\" + "resumo.tsv"
data = pd.read_csv(datafile, sep="\t")

# formula = 'ndcg10 ~ C(Loss) + C(typeret) + C(alpha) + C(correl) + C(usebaseline) '

names = ["Loss", "typeret", "alpha", "correl"]
form = ""
for i in range(len(names)):
    C = itertools.combinations(names, i + 1)
    for c in C:
        if len(c) == 1:
            form += "C(" + c[0] + ") + "
        else:
            for cin in c:
                # print(c)
                form += "C(" + cin + "):"
            form = form[:-1] + " + "
print(form[:-3])

formula = "lndcg10 ~ " + form[:-3]
# formula = "lndcg10 ~ " + "C(Loss) + C(typeret) + C(alpha) + C(correl) + C(usebaseline) + C(Loss):C(typeret) + C(Loss):C(alpha) + C(Loss):C(correl) + C(Loss):C(usebaseline) + C(typeret):C(alpha) + C(typeret):C(correl) + C(typeret):C(usebaseline) + C(alpha):C(correl) + C(alpha):C(usebaseline) + C(correl):C(usebaseline) + C(Loss):C(typeret):C(alpha) + C(Loss):C(typeret):C(correl) + C(Loss):C(typeret):C(usebaseline) + C(Loss):C(alpha):C(correl) + C(Loss):C(alpha):C(usebaseline) + C(Loss):C(correl):C(usebaseline) + C(typeret):C(alpha):C(correl) + C(typeret):C(alpha):C(usebaseline) + C(typeret):C(correl):C(usebaseline) + C(alpha):C(correl):C(usebaseline) + C(Loss):C(typeret):C(alpha):C(correl) + C(Loss):C(typeret):C(alpha):C(usebaseline) + C(Loss):C(typeret):C(correl):C(usebaseline) + C(Loss):C(alpha):C(correl):C(usebaseline) + C(typeret):C(alpha):C(correl):C(usebaseline) + C(Loss):C(typeret):C(alpha):C(correl):C(usebaseline)"

model = ols(formula, data=data).fit()
aov_table = statsmodels.api.stats.anova_lm(model, typ=2)

eta_squared(aov_table)
omega_squared(aov_table)
print(aov_table)

res = model.resid
fig = sm.qqplot(res, line='s')
plt.show()

#
# from pyvttbl.base import DataFrame
#
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results"
# datafile = home + "\\" + "resumo.tsv"
# df = DataFrame()
# df.read_tbl(datafile)
# df['id'] = range(len(df['len']))
#
# # print(df.anova('lndcg10', sub='id', bfactors=['supp', 'dose']))
# print(df.anova('lndcg10', sub='id', bfactors=["Loss", "typeret", "alpha", "correl", "usebaseline"]))
