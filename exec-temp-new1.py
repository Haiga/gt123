import json
from allrank.mymain import myrun
import multiprocessing

path_dataset = "/home/silvapedro/experimento_loss_risk/BD/default-web10k/Fold/_normalized"
name_exp = "temporary-evalaluation3/"

lists = []
cont = 200
for d_ff in [512]:
    for v in [5, 2]:
        for fold in [0]:
            for seed in [42]:
                for lr in [0.0001, 0.0005, 0.001]:
                    for vu in [0.1]:
                        n_fold = str(fold + 1)
                        for loss in ['spearmanLossmulti']:
                            loss = loss.replace("multi", "")

                            with open(
                                    '/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/w3c.json') as json_file:
                                data = json.load(json_file)

                            metrics = {'ndcg_5': 0.0, 'ndcg_10': 0.0, 'georisk_10': 0.0, 'lndcg_5': 0.0, 'lndcg_10': 0.0, }
                            data['expected_metrics']['val'] = metrics
                            data['metrics'] = ['ndcg_5', 'ndcg_10', 'georisk_10', 'lndcg_5', 'lndcg_10']
                            data['data']['path'] = path_dataset.replace("Fold", "Fold" + n_fold)
                            data['loss']['name'] = "lambdaLossmulti"
                            data['training']['epochs'] = 100
                            data['optimizer']['args']['lr'] = lr
                            data['model']['transformer']['d_ff'] = d_ff
                            with open("/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                                      'w') as outfile:
                                json.dump(data, outfile)

                            cont += 1
                            name = "--" + str(cont) + "--" + str(v)

                            try:
                                # myrun(loss + str(lr) + "fold" + n_fold + "-" + name,
                                myrun(loss + "fold" + n_fold + "-" + name,
                                      "/home/silvapedro/experimento_loss_risk/resultados-" + name_exp,
                                      "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                                      # useCuda=False, seed=seed, num_baselines=5)
                                      useCuda=False, seed=seed, num_baselines=v)
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
