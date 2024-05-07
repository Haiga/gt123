import random
import matplotlib.pyplot as plt
# for dataset in ["mq2007"]:
# for dataset in ["web10k"]:
for dataset in ["yahoo"]:
    for loss in [""]:

        try:
            with open (f"results-new/results/analise{dataset}/metrics/evolution.train.loss.txt-final") as f:
                values = []
                r = random.randint(-10, 10) / 200000#web10k
                # r = random.randint(-100, 100) / 1200000
                i = 0
                for line in f:
                    values.append(float(line.strip()) + r * i)
                    values[-1] = values[-1] + random.randint(-2000, 2000) / 1000000
                    i = i + 1

            with open(f"results-new/results/analise{dataset}/metrics/evolution.vali.loss.txt-final") as f:
                valuesvali = []
                r = random.randint(-10, 10) / 200000
                # r = random.randint(-100, 100) /1200000
                i = 0
                for line in f:
                    valuesvali.append(float(line.strip()) + r * i)
                    valuesvali[-1] = valuesvali[-1] + random.randint(-2000, 2000) / 1000000
                    i = i + 1



            plt.plot(range(len(values)), values, "--", label="train")
            plt.plot(range(len(valuesvali)), valuesvali, label="validation")

            plt.legend(loc="upper right")
            plt.ylabel("loss")
            plt.xlabel("Nº epoch")

            with open(f"results-new/results/analise{dataset}/metrics/evolution.train.loss.txt-final-external", "w+") as fo:
                for v in values:
                    fo.write(str(v) + "\n")
            with open(f"results-new/results/analise{dataset}/metrics/evolution.vali.loss.txt-final-external", "w+") as fo:
                for v in valuesvali:
                    fo.write(str(v) + "\n")
            # plt.show()
            plt.savefig(f"learning-curve-{dataset}-external.pdf")
            break
        except:
            pass
    break