import json
from allrank.mymain import myrun

path_dataset = "/home/silvapedro/experimento_loss_risk/BD/web10k/Fold"
name_exp = "web10k-mlp-eb-exec1/"

for fold in [0, 1, 2, 3, 4]:
    n_fold = str(fold + 1)
    # for loss in ['geoRiskSpearmanLoss', 'geoRiskListnetLoss']:
    for loss in ['geoRiskListnetLoss']:

        with open(
                '/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/temp_config_ml2.json') as json_file:
            data = json.load(json_file)

        metrics = {'ndcg_5': 0.0, 'ndcg_10': 0.0, 'georisk_10': 0.0, 'lndcg_5': 0.0,
                   'lndcg_10': 0.0, }
        data['expected_metrics']['val'] = metrics
        data['metrics'] = ['ndcg_5', 'ndcg_10', 'georisk_10', 'lndcg_5', 'lndcg_10']
        data['data']['path'] = path_dataset + n_fold
        data['loss']['name'] = loss
        data['loss']['args']['corr'] = 1

        with open(
                "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                'w') as outfile:
            json.dump(data, outfile)

        name = ""

        try:
            myrun(loss + "fold" + n_fold + "-" + name,
                  "/home/silvapedro/experimento_loss_risk/resultados-" + name_exp,
                  "/home/silvapedro/experimento_loss_risk/risk-loss-nn/allrank/allrank/new_config.json",
                  useCuda=False, seed=42, num_baselines=0)
            # useCuda=False, seed=42, num_baselines=0,
            # path_base="/home/silvapedro/experimento_loss_risk/resultados-mq2007-startpoint2-mlp-14/results/spearmanLossfold" + n_fold + "-/model.pkl")
        except:
            print("error")

