import random
import matplotlib.pyplot as plt
import numpy as np
# for dataset in ["mq2007"]:
# for dataset in ["web10k"]:
for dataset in ["yahoo"]:
    for loss in [""]:

        try:
            with open (f"results-new/results/analise{dataset}/metrics/evolution.train.loss.txt-final") as f:
                values = []
                for line in f:
                    values.append(float(line.strip()))


            with open(f"results-new/results/analise{dataset}/metrics/evolution.vali.loss.txt-final") as f:
                valuesvali = []
                i = 0
                for line in f:
                    valuesvali.append(float(line.strip()))
            values = np.array(values)
            valuesvali = np.array(valuesvali)


            plt.plot(range(len(values)), values, "--", color='k', label="train")
            plt.plot(range(len(valuesvali)), valuesvali, color='#808080', label="validation")

            plt.legend(loc="upper right")
            plt.ylabel("loss")
            plt.xlabel("Nº epoch")

            plt.ylim(0.0, 0.500)
            # plt.ylim(0.125, 0.158)
            # plt.ylim(0.18, 0.29)
            # plt.show()
            plt.savefig(f"learning-curve-{dataset}-3.pdf")
            break
        except:
            pass
    break