def openM(path):
    with open(path) as fi:
        fi.readline()
        vs = []
        for line in fi:
            v = float(line.split("\t")[0].strip())
            vs.append(v)
    return vs

from scipy.stats import ttest_rel, ttest_ind
import numpy as np
gc = openM("geoRiskListnetLoss.tsv")
s = openM("spearmanLoss.tsv")

print(ttest_rel(gc, s))
print(ttest_ind(gc, s))