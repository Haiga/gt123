import json
from allrank.mymain import run

for fold in range(1, 2):
# for fold in range(1, 2):
    for loss in ['approxNDCGLoss', 'myApproxGeoLoss', 'myApproxGeoNDCGLoss']:
    # for loss in ['myApproxGeoNDCGLoss']:
        f = open("temp_config.json", "r")
        r = json.load(f)
        f.close()

        # r["data"]["path"] = r["data"]["path"].replace("Fold1", "Fold" + str(fold))
        spt = r["data"]["path"].split("/")
        spt[-1] = "Fold" + str(fold)
        r["data"]["path"] = "/".join(spt)  # "Fold" + str(fold))
        # r["data"]["path"] = "Fold" + str(fold)
        r["loss"]["name"] = loss
        # r["training"]["epochs"] = fold + 1
        r["training"]["epochs"] = 30
        f = open("temp_config.json", "w+")
        json.dump(r, f)
        f.close()

        run(id=loss + "_" + str(fold))

    try:
        print("PRINTING RESULT UNTIL NOW")
        f2 = open("resultado.txt")
        for line in f2:
            print(line)
        f2.close()
    except:
        print("not working ...")
# 1 tabela dessa para o Georisk
# Nº epochs == 30
# Loss-NDCG  Loss-Geo-NDCG Loss-Geo
# F1 r1  r2  r3
# F2 r1  r2  r3
# F3 r1  r2  r3
# F4 r1  r2  r3
# F5 r1  r2  r3

# 1 tabela dessa para o NDCG@10
# Nº epochs == 30
# Loss-NDCG  Loss-Geo-NDCG Loss-Geo
# F1 r1  r2  r3
# F2 r1  r2  r3
# F3 r1  r2  r3
# F4 r1  r2  r3
# F5 r1  r2  r3
