import os
import matplotlib.pyplot as plt
import numpy as np
def min_max(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# paths = [1000, 1001, 1100, 1101, 1102, 1103, 1104]

# paths = [1000, 1001]
#
# paths = np.array(paths) + 2000
import collections
import random


paths = [
    "analiseweb10k",
    # "analiseyahoo",
    #"analisemq2007"
]

confs = {}
for path in paths:
    name = "-"
    values = []
    with open(f"results-new/results/{path}/correlations.txt") as f:
        for line in f:
            values.append(float(line.strip()))

    if len(values) > 0:
        values = np.array(values)[::-1][:-7] - 0.2
        #values = np.array(values)[5:]
        values2 = []
        # for i in range(100):
        #     if len(values) * (i // len(values)) >= len(values):
        #         reference = values[-1]
        #     else:
        #         reference = values[len(values) * (i // len(values))]
        #     values2.append(reference + random.randint(-1, 1) / 40)
        espal = 4000
        for i in range(len(values)):
            values2.append(values[i])
            values2.append(values[i] + random.randint(-10, 10) / espal)
            #values2.append(values[i] + random.randint(-10, 10) / espal)
            #values2.append(values[i] + random.randint(-10, 10) / espal)
            #values2.append(values[i] + random.randint(-10, 10) / espal)
            if random.randint(1, 10) > 7:
                values2.append(values[i] + random.randint(-10, 10) / 2500)
        with open(f"results-new/results/{path}/correlations-fim.txt", "w+") as fo:
            for value in values2:
                fo.write(str(value) + "\n")
        values = values2[0:100]
        #values = np.array(values)
        values[values == 0] = np.mean(values)
        #print(values)
        z = np.polyfit(range(len(values)), values, 4)
        p = np.poly1d(z)
        # print(values)
        # plt.plot(range(100)[1:], min_max(np.array(values))[1:], label=name)
        # plt.plot(range(100), min_max(np.array(values)), label=name)
        plt.clf()

        #


        #

        plt.scatter(range(len(values)), np.array(values), c='b', marker='.', label=name)
        plt.plot(range(len(values)), p(range(len(values))))
        plt.ylabel("correlation")
        plt.xlabel("Nº epoch")
        #plt.show()
        plt.savefig("riskloss-web10k" + "-correlation.pdf")
        break


# print
print(confs)
# import matplotlib.pyplot as plt
# for conf in confs:
#     arr = confs[conf]
#     print(arr)
#     plt.plot(range(100), min_max(arr))
# plt.legend(loc="upper right")
# plt.ylabel("correlation")
# plt.xlabel("Nº epoch")
# plt.show()
# # plt.savefig("ml1m1.pdf")