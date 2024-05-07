import sys
import os
import numpy as np

folder = sys.argv[1]
# folder = "."
filesnames = os.listdir(folder)

def myp(metric):
    resultados = {}
    for filename in filesnames:
        if os.path.isdir(folder + "/" + filename):
            method = filename.split("fold")[0]
            if method not in resultados:
                resultados[method] = []
            with open(folder + "/" + filename + "/" + f"model2.predict.{metric}.txt") as fi:
                for line in fi:
                    # line = line.strip()
                    resultados[method].append(float(line.strip()))

    for method in resultados.keys():
        print(f"{method} :{np.mean(resultados[method])}")

for m in ["lndcg_10", "lndcg_5", "ndcg_10", "ndcg_5"]:
    print(m)
    myp(m)
    print("-")