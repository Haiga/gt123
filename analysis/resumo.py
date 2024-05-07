import os

# Lê todos as configurações na pasta home e cria um arquivo "test.resumo" com a média das medidas de avaliação
# que estão salvos em evolution.test.metrics.txt
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-web10k-0403\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0503\\resultados-ml5k\\results"
home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0903\\resultados-ml5k-tuning\\results"
# home = "D:\\Colecoes\\experimento_loss_risk\\execucao-0903\\resultados-web10k-tuning\\results"
methods = os.listdir(home)

with open(home + "\\resumo.tsv",
          "w") as fo:
    fo.write("Loss\ttyperet\talpha\tcorrel\tusebaseline\tndcg5\tndcg10\tlndcg5\tlndcg10\n")
    for method in methods:
        if "resumo" in method: continue
        if "info" in method: continue
        if ".tsv" in method: continue
        if not method.startswith("geo"): continue
        size = 1
        name_file = home + "/" + method + "/metrics/evolution.vali.metrics.txt"
        final_line = "\n"
        if os.path.isfile(name_file):
            with open(name_file, "r") as f:
                lll = f.readlines()
                final_line = lll[-1]
                size = len(lll)
        n = method.replace("Lossfold1-", "")
        sp = n.split("-")
        loss = sp[0]
        type_ret = sp[1]
        alpha = sp[2]
        correlation = sp[3]
        use_baseline = sp[4]

        fo.write(loss + "\t" + type_ret + "\t" + alpha + "\t" + correlation + "\t" + use_baseline + "\t" + final_line)

print(methods)
