import numpy as np

home = "/home/silvapedro/experimento_loss_risk/BD/ml20m/tuning70"
keys_file = "keys.csv"
users = {}
cont = 0
users_arr = []
with open(home + "/" + keys_file, "r") as fi:
    for line in fi:
        line0 = line.split(",")[0]
        line1 = line.split(",")[1]
        if int(line0) not in users:
            users.setdefault(int(line0), [])
        users[int(line0)].append(int(line1))
        cont += 1
        users_arr.append(int(line0))
# print(max(users_arr))
# print(min(users_arr))
# print(len(users))
# print(cont)
num_items_by_users = []
for u in users:
    num_items_by_users.append(len(users[u]))

a = np.array(list(users.keys()))
np.random.shuffle(a)

train = a[0:(int(0.6 * len(users)))]
valid = a[(int(0.6 * len(users))):(int(0.8 * len(users)))]
test = a[(int(0.8 * len(users))):]

u_train = set()
for u in train:
    u_train.add(u)

u_test = set()
for u in test:
    u_test.add(u)

u_vali = set()
for u in valid:
    u_vali.add(u)

# print("---")
# print(u_train.intersection(u_vali))
# print(u_train.intersection(u_test))
# print(u_vali.intersection(u_test))
# print("---")


trainf = open(home + "/train-" + keys_file, "w")
testf = open(home + "/test-" + keys_file, "w")
valif = open(home + "/vali-" + keys_file, "w")

with open(home + "/" + keys_file, "r") as fi:
    for line in fi:
        line0 = int(line.split(",")[0])
        # line1 = line.split(",")[1]
        if line0 in u_train:
            trainf.write(line)
        elif line0 in u_vali:
            valif.write(line)
        elif line0 in u_test:
            testf.write(line)
trainf.close()
testf.close()
valif.close()


trainf = open(home + "/Norm.train.txt", "w")
testf = open(home + "/Norm.test.txt", "w")
valif = open(home + "/Norm.vali.txt", "w")

with open(home + "/" + keys_file, "r") as fi:
    with open(home + "/" + "FWLS-all.skl", "r") as fi2:
        for line in fi:
            qid = line.split(",")[0]
            line2 = fi2.readline()
            if line == "\n" or line2 == "\n":
                break
            spl = line2.replace("\n", "").split(" ")
            spl_f = spl[1:]

            u = spl[0] + " qid:" + qid + " "
            for feat in spl_f:
                s = feat.split(":")
                try:
                    x = float(s[1])
                    u += s[0] + f":{x:.4f} "
                except:
                    u += s[0] + f":0 "
            result_line = u[:-1] + "\n"

            line0 = int(qid)
            # line1 = line.split(",")[1]
            if line0 in u_train:
                trainf.write(result_line)
            elif line0 in u_vali:
                valif.write(result_line)
            elif line0 in u_test:
                testf.write(result_line)

trainf.close()
testf.close()
valif.close()