import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file

home = r'D:\Colecoes\experimento_loss_risk\tables-apresent-3\mq2007'
metric = 'lndcg_10'

df_risk_values = pd.read_csv(home + "/" + 'geoRisk.tsv', sep="\t")[metric].values
df_non_values = pd.read_csv(home + "/" + 'pointwise_rmse.tsv', sep="\t")[metric].values

# mask = (df_risk_values - df_non_values) > (0.999999 * df_non_values)
# mask = (df_risk_values - df_non_values)
# mask = (df_non_values - df_risk_values) > (0.999999 * df_risk_values)
mask = (df_non_values - df_risk_values)

top10 = np.argsort(mask)[::-1][:10]
all_queries_ids = []
for i in range(5):
    path = f'D:\Colecoes\BD\mq2007\S{i+1}.txt'
    data = load_svmlight_file(path, query_id=True)

    queries_ids = np.array(data[2])
    # ant = queries_ids[0]
    all_queries_ids.extend(np.unique(queries_ids))

all_queries_ids = np.array(all_queries_ids)
# print(np.sum(mask))
# print(all_queries_ids[mask])
# qids = all_queries_ids[mask]
qids = all_queries_ids[top10]
with open('D:\Colecoes\BD\mq2007\\10000.topics') as fi:
    for line in fi:
        if int(line.split(":")[0]) in qids:
            print(line)
