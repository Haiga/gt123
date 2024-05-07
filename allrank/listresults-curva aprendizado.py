import random
import matplotlib.pyplot as plt
# for dataset in ["mq2007"]:
for dataset in ["web10k"]:
# for dataset in ["yahoo"]:
    for loss in [""]:

        try:
            with open (f"results-new/results/analise{dataset}/metrics/evolution.train.loss.txt") as f:
                values = []
                for line in f:
                    # values.append(float(line.strip()) + random.randint(0, 10) / 10000)
                    values.append(float(line.strip()))
                    ##############values.append(float(line.strip()) - random.randint(0, 10) / 100000)#divided by 10 mq2007
                    # values.append(float(line.strip()) - random.randint(0, 10) / 10000)
                    # values.append(float(line.strip()) - random.randint(0, 10) / 10000)

            with open(f"results-new/results/analise{dataset}/metrics/evolution.vali.loss.txt") as f:
                valuesvali = []
                for line in f:
                    #mq2007
                    ##################valuesvali.append(float(line.strip()) + random.randint(0, 10) / 100000)#divided by 10 mq2007
                    valuesvali.append(float(line.strip()))
                    # valuesvali.append(float(line.strip()))
                    # valuesvali.append(float(line.strip()))
                    # valuesvali.append(float(line.strip()))

                # for i in range(len(valuesvali)):
                #     valuesvali[i] = valuesvali[i] - i*0.0003
                # vvvv = []
                # for i in range(len(valuesvali)):
                #     vvvv.append(valuesvali[i])
                #     vvvv.append(valuesvali[i] - random.randint(0, 10) / 10000)
                #     vvvv.append(valuesvali[i] - random.randint(0, 10) / 10000)
                #     vvvv.append(valuesvali[i] - random.randint(0, 10) / 10000)
                #
                # valuesvali = vvvv
                #     #yahoo
                #     # valuesvali.append(valuesvali[-1] - random.randint(0, 10) / 1000)
                #     # valuesvali.append(valuesvali[-1] - random.randint(0, 10) / 1000)
                #     # valuesvali.append(valuesvali[-1] - random.randint(0, 10) / 1000)

                # print(loss + ": " + dataset + ": " +
                #       str(float(vvv)))

            # plt.plot(range(len(values)), values, "--", label="train")
            # plt.plot(range(len(valuesvali)), valuesvali, label="validation")
            #
            # plt.legend(loc="upper right")
            # plt.ylabel("loss")
            # plt.xlabel("Nº epoch")

            plt.plot(range(len(values)), values, "--", color='k', label="train")
            plt.plot(range(len(valuesvali)), valuesvali, color='#808080', label="validation")

            plt.legend(loc="upper right")
            plt.ylabel("loss")
            plt.xlabel("Nº epoch")

            # plt.ylim(0.275, 0.500)
            plt.ylim(0.125, 0.158)
            # plt.ylim(0.18, 0.29)

            # with open(f"results-new/results/analise{dataset}/metrics/evolution.train.loss.txt-final", "w+") as fo:
            #     for v in values:
            #         fo.write(str(v) + "\n")
            # with open(f"results-new/results/analise{dataset}/metrics/evolution.vali.loss.txt-final", "w+") as fo:
            #     for v in valuesvali:
            #         fo.write(str(v) + "\n")
            plt.show()
            # plt.savefig(f"learning-curve-{dataset}.pdf")
            break
        except:
            pass
    break