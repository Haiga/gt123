import glob

from scipy import stats
import numpy as np


def read_lines(path):
    r = []
    with open(path, 'r') as p:
        p.readline()
        ps = p.readlines()
        for line in ps:
            metrics = [float(x) for x in line.strip().replace("\n", "").split("\t")]
            metrics = metrics[0:2]
            r.append(metrics)
    return r


def getData(files):
    rs = []
    for file in files:
        rs.append(read_lines(file))

    return rs


home = "D:\\Colecoes\\experimento_loss_risk\\execucao-2302\\analise1\\t-test-paired"
outFile = f'{home}/{"ttest-file.tsv"}'
data = []
names_files = glob.glob(home + f'\\*.tsv')
new_names_files = []
for name_file in names_files:
    if "ttest-file.tsv" == name_file.split("\\")[-1]:
        continue
    new_names_files.append(name_file)

data.extend(getData(new_names_files))

data = np.array(data)

for m in range(len(data[0, 0])):
    for i in range(len(data)):
        print(new_names_files[i].split("\\")[-1] + "\t", end="")
        for j in range(len(data)):
            if i == j:
                print("-\t", end="")
            else:
                t, p_value = stats.ttest_rel(data[i, :, m], data[j, :, m], alternative="greater")
                diff = np.mean(data[i, :, m]) - np.mean(data[j, :, m])
                if p_value < 0.05:
                    print("e", end="")
                print(str(diff) + "\t", end="")
        print("\n", end="")

    print("\n", end="")
    print("\n", end="")


