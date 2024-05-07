import json
from allrank.mymain import myrun


path_dataset = "/home/silvapedro/experimento_loss_risk/BD/mq2007/Fold"
name_exp = "mq2007-mlp-eb-exec0/"

# for size in [[64, 128, 64], [128, 64]]:
for size in [[64, 128, 64]]:
    for input_norm in [False]:
        for act in [""]:
            for fold in [2]:
                for seed in [1, 2, 3, 4, 5]:
                    for lr in [0.001]:
                        n_fold = str(fold + 1)
                        for loss in ['geoRiskSpearmanLoss', 'geoRiskListnetLoss']:

                            with open(
                                    '/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/temp_config2ml.json') as json_file:
                                data = json.load(json_file)

                            metrics = {'ndcg_5': 0.0, 'ndcg_10': 0.0, 'georisk_10': 0.0, 'lndcg_5': 0.0,
                                       'lndcg_10': 0.0, }
                            data['expected_metrics']['val'] = metrics
                            data['metrics'] = ['ndcg_5', 'ndcg_10', 'georisk_10', 'lndcg_5', 'lndcg_10']
                            data['data']['path'] = path_dataset + str(seed)
                            data['loss']['name'] = "spearmanLoss"
                            data['optimizer']['args']['lr'] = lr
                            epoch = 50
                            if loss == 'geoRiskSpearmanLoss':
                                # seed = 33
                                epoch = 50
                            if loss == 'geoRiskListnetLoss':
                                # seed = 34
                                epoch = 51
                            if loss == 'spearmanLoss':
                                # seed = 35
                                epoch = 31
                            if loss == 'lambdaLoss':
                                # seed = 36
                                epoch = 34
                            if loss == 'listNet':
                                # seed = 37
                                epoch = 32
                            if loss == 'pointwise_rmse':
                                # seed = 38
                                epoch = 39
                            if loss == 'ordinal':
                                # seed = 39
                                epoch = 27
                            epoch = epoch - 13
                            data["model"]["fc_model"]["sizes"] = size
                            if act != "":
                                data["model"]["fc_model"]["activation"] = act
                            # data["model"]["fc_model"]["dropout"]
                            data["model"]["fc_model"]["input_norm"] = input_norm

                            if "geoRisk" in loss:
                                data['training']['epochs'] = epoch
                            else:
                                data['training']['epochs'] = epoch


                            with open(
                                    "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                                    'w') as outfile:
                                json.dump(data, outfile)

                            name = ""
                            # for s in size:
                            #     name += str(s) + "."
                            # name += "-" + str(act)
                            # name += "-" + str(input_norm)
                            # name += "-" + str(lr)

                            try:
                                myrun(loss + "fold" + str(seed) + "-" + name,
                                      "/home/silvapedro/experimento_loss_risk/resultados-" + name_exp,
                                      "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                                      useCuda=False, seed=epoch+100, num_baselines=0)
                            except Exception as e:
                                print("*****************************ERROR***********************************")
                                print(loss + "fold" + n_fold + "-" + name)
                                print(e)
                                print("*****************************ERROR***********************************")
