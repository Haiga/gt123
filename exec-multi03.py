import json
from allrank.mymain import myrun
import multiprocessing

path_dataset = "/home/silvapedro/experimento_loss_risk/BD/datay"
name_exp = "datay-mlp-multi-drop-new02/"

lists = []
for fold in [0]:
    for seed in [43]:
        for lr in [0.001]:
            n_fold = str(fold + 1)
            for loss in ['spearmanLossmulti', 'listNetmulti', 'lambdaLossmulti', 'pointwise_rmsemulti',
                         'ordinalmulti', 'CorrScoremulti']:

                with open(
                        '/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/temp_config_ml.json') as json_file:
                    data = json.load(json_file)

                metrics = {'ndcg_5': 0.0, 'ndcg_10': 0.0, 'georisk_10': 0.0, 'lndcg_5': 0.0, 'lndcg_10': 0.0, }
                data['expected_metrics']['val'] = metrics
                data['metrics'] = ['ndcg_5', 'ndcg_10', 'georisk_10', 'lndcg_5', 'lndcg_10']
                data['data']['path'] = path_dataset
                data['loss']['name'] = loss

                if "geoRisk" in loss:
                    data['training']['epochs'] = 100
                else:
                    data['training']['epochs'] = 3

                if ("pointwise_rmse" in loss) or ("ordinal" in loss):
                    data['loss']['args']['normalized'] = 1  # web10k foi normalizada

                if "ordinal" in loss:
                    data['loss']['args']['normalized'] = 1  # web10k foi normalizada
                    data['model']['post_model']['d_output'] = 5  # multi saidas para ordinal

                if "spearmanLossmulti" in loss:
                    data['loss']['args']['u'] = 0.1

                with open("/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                          'w') as outfile:
                    json.dump(data, outfile)

                name = ""

                try:
                    myrun(loss + "fold" + n_fold + "-" + name,
                          "/home/silvapedro/experimento_loss_risk/resultados-" + name_exp,
                          "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                          useCuda=False, seed=seed, num_baselines=5)
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
#     # a = executor.submit(LocalEval, list_of_args)
#     results = pool.map(myrun, lists)
#     for r in results:
#         print(r)
