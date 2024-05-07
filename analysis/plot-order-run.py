# dados dois arquivos de métrica por consulta, ordena as métricas e plota ambos

def read_lines(path):
    r = []
    with open(path, 'r') as p:
        for line in p:
            r.append(float(line.strip().replace("\n", "")))
    return r


my = read_lines("model.predict.lndcg_10.txt")
list = read_lines("list.model.predict.lndcg_10.txt")

import numpy as np
import matplotlib.pyplot as plt

x = range(len(my))
my = np.sort(my)
list = np.sort(list)

plt.plot(x, my, '--', label="my", markersize=5, color="#aaf0d1")
plt.plot(x, list, '--', label="list")
plt.xlim((400, 500))

plt.legend()
plt.show()
