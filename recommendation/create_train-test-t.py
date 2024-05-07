import os

# queries_set = set()
# queries_ids = []
# num_queries_to_extract = 6000
# num_queries_to_extract = 5

home = 'D:\\Colecoes\\BD\\out\\temp-features\\features\\dev\\'
if not os.path.exists(home + '10k/'):
    os.makedirs(home + '10k/')


def doSplit(type, num_queries_to_extract):
    queries_set = set()
    queries_ids = []
    with open(home + '10k/' + 'Norm.' + "test" + '.txt', "r") as f:
        with open(home + '10k/' + 'Norm.' + "vali" + '.txt', "w") as fo:
            with open(home + '10k/' + 'Norm.' + "test2" + '.txt', "w") as fo2:
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
                        fo2.write(line)


doSplit("train", 30)
# doSplit("test", 2000)
# doSplit("vali", 2000)
