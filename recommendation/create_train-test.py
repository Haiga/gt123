import os

# queries_set = set()
# queries_ids = []
# num_queries_to_extract = 6000
# num_queries_to_extract = 5

size = "5k/"

home = '/home/silvapedro/experimento_loss_risk/BD/ml20m/tuning70/'
if not os.path.exists(home + size):
    os.makedirs(home + size)


def doSplit(type, num_queries_to_extract):
    queries_set = set()
    queries_ids = []
    with open(home + 'Norm.' + type + '.txt', "r") as f:
        with open(home + size + 'Norm.' + type + '.txt', "w") as fo:
            for line in f:
                qid = line.split(" ")[1].split("qid:")[1]
                if len(queries_set) < num_queries_to_extract:
                    queries_ids.append(qid)
                    queries_set.add(qid)
                    # write line
                    fo.write(line)
                elif len(queries_set) == num_queries_to_extract and qid in queries_set:
                    queries_ids.append(qid)
                    # write line
                    fo.write(line)
                else:
                    break


doSplit("train", 3000)
doSplit("test", 1000)
doSplit("vali", 1000)
