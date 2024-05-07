import math
import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import norm

def read_and_prit_mean(path):
    # pd.read_csv(path)[0].values
    return np.mean(pd.read_csv(path).values.reshape(-1))

if __name__ == "__main__":
    home = r"D:\Colecoes\experimento_loss_risk\tables-apresent-3\mlp\mq2007"
    x = os.listdir(home)
    print(x)
    for file in x:
        if os.path.isdir(home + "/" + file):
            if os.path.isfile(home + "/" + file + "/model.predict.lndcg_10.txt"):
                r = read_and_prit_mean(home + "/" + file + "/model.predict.lndcg_10.txt")
                print(f"1-:{file}:\t{r}")
            if os.path.isfile(home + "/" + file + "/model2.predict.lndcg_10.txt"):
                r = read_and_prit_mean(home + "/" + file + "/model2.predict.lndcg_10.txt")
                print(f"2-:{file}:\t{r}")