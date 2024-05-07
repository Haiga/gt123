import numpy as np
import pandas as pd
def getarr(path):
    with open(path) as inp:
        data = []
        for line in inp:
            data.append(float(line.strip()))
        return data


if __name__ == "__main__":
    home = "/home/silvapedro/experimento_loss_risk/BD/web30k"
    for fold in ['1', '2', '3', '4', '5']:
        for role in ['train', 'test', 'vali']:
            x1 = None
            for baseline in ['lmart', 'adarank']:
                if x1 is None:
                    x1 = np.array(getarr(home + "/Fold" + fold + "/" + baseline + "-" + role + ".csv"))
                else:
                    x2 = np.array(getarr(home + "/Fold" + fold + "/" + baseline + "-" + role + ".csv"))
                    x1 = np.vstack((x1, x2))

            pd.DataFrame(x1.T).to_csv(home + "/Fold" + fold + "/" + "baseline" + ".Norm." + role + ".txt", index=False, header=False)

    home = "/home/silvapedro/experimento_loss_risk/BD/datay"
    for fold in ['']:
        for role in ['train', 'test', 'vali']:
            x1 = None
            for baseline in ['lmart', 'adarank']:
                if x1 is None:
                    x1 = np.array(getarr(home + "" + fold + "/" + baseline + "-" + role + ".csv"))
                else:
                    x2 = np.array(getarr(home + "" + fold + "/" + baseline + "-" + role + ".csv"))
                    x1 = np.vstack((x1, x2))

            pd.DataFrame(x1.T).to_csv(home + "" + fold + "/" + "baseline" + ".Norm." + role + ".txt", index=False,
                                      header=False)

# x = [1, 3, 5]
# x2 = [2, 3, 6]
#
# x3 = np.vstack((x, x2))
# print(x3)
# x4 = [1, 3, 5]
#
# x5 = np.vstack((x3, x4))
# print(x5)
#
# pd.DataFrame(x5.T).to_csv('tets.csv', index=False, header=False)
