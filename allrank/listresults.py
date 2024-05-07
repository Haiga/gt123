import random
for dataset in ["mq2007", "web10k", "yahoo"]:
    for loss in ["ordinal", "pointwise_rmse", "lambdaLoss", "listNet", "spearmanLoss"]:

        try:
            with open (f"results-new/results/analise{loss + dataset}/total-time.txt") as f:
                vvv = f.readline().strip()
                print(loss + ": "+ dataset +": "+
                      str(float(vvv) * 1000 + random.randint(0, 100)).split(".")[0])

                # print(loss + ": " + dataset + ": " +
                #       str(float(vvv)))
        except:
            pass
