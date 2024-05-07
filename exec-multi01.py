import json
from allrank.mymain import myrun
import multiprocessing

path_dataset = "/home/silvapedro/experimento_loss_risk/BD/web30k/Fold"
name_exp = "web30k-exec-no-2/"

lists = []
for fold in [4]:
    for seed in [42]:
        for lr in [0.001]:
            for vu in [0.1]:
                n_fold = str(fold + 1)
                for loss in ['spearmanLossmulti', 'listNetmulti', 'lambdaLossmulti', 'pointwise_rmsemulti',
                             'ordinalmulti']:
                    loss = loss.replace("multi", "")

                    with open(
                            '/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/temp_config_ml.json') as json_file:
                        data = json.load(json_file)

                    metrics = {'ndcg_5': 0.0, 'ndcg_10': 0.0, 'georisk_10': 0.0, 'lndcg_5': 0.0, 'lndcg_10': 0.0, }
                    data['expected_metrics']['val'] = metrics
                    data['metrics'] = ['ndcg_5', 'ndcg_10', 'georisk_10', 'lndcg_5', 'lndcg_10']
                    data['data']['path'] = path_dataset + n_fold
                    data['loss']['name'] = loss
                    data["model"]["fc_model"]["dropout"] = 0.1
                    data["model"]["fc_model"]["sizes"] = [128, 256, 128]
                    # data['loss']['args']['u'] = vu
                    data['model']['fc_model']['input_norm'] = False
                    # data['model']['fc_model']['activation'] = ""
                    data['optimizer']['args']["lr"] = lr

                    if "geoRisk" in loss:
                        data['training']['epochs'] = 100
                    else:
                        data['training']['epochs'] = 50

                    if ("pointwise_rmse" in loss) or ("ordinal" in loss):
                        data['loss']['args']['normalized'] = 0  # web10k foi normalizada

                    if "ordinal" in loss:
                        data['loss']['args']['normalized'] = 0  # web10k foi normalizada
                        data['model']['post_model']['d_output'] = 5  # multi saidas para ordinal

                    if "spearmanLossmulti" in loss:
                        data['loss']['args']['u'] = vu

                    with open("/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                              'w') as outfile:
                        json.dump(data, outfile)

                    name = ""

                    try:
                        # myrun(loss + str(lr) + "fold" + n_fold + "-" + name,
                        myrun(loss + "fold" + n_fold + "-" + name,
                              "/home/silvapedro/experimento_loss_risk/resultados-" + name_exp,
                              "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                              # useCuda=False, seed=seed, num_baselines=5)
                              useCuda=False, seed=seed, num_baselines=1)
                    except Exception as e:
                        print("*****************************ERROR***********************************")
                        print(loss + "fold" + n_fold + "-" + name)
                        print(str(e))
                        print("*****************************ERROR***********************************")
                        # raise e
                    lists.append([loss + "fold" + n_fold + "-" + name,
                                  "/home/silvapedro/experimento_loss_risk/resultados-" + name_exp,
                                  "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                                  False, seed, 5])

# with multiprocessing.Pool(processes=2) as pool:
##     a = executor.submit(LocalEval, list_of_args)
# results = pool.map(myrun, lists)
# for r in results:
#     print(r)
