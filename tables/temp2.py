
with open(r"D:\Colecoes\experimento_loss_risk\tables\overall\icfile-ranked.tsv", "w+") as ic:
    with open(r"D:\Colecoes\experimento_loss_risk\tables\overall\loss-wins-ranked.tsv", "w+") as lw:
        for dataset in ['mq2007', 'yahoo', 'web10k']:
                for tt in ['att', 'mlp']:
                    file1 = r"D:\Colecoes\experimento_loss_risk\tables\overall"+f"\\{tt}"+f"\\{dataset}\\ic-file-only.tsv"
                    file2 = r"D:\Colecoes\experimento_loss_risk\tables\overall"+f"\\{tt}"+f"\\{dataset}\\losses-wins.tsv"
                    ic.writelines(open(file1).readlines())
                    ic.write("\n\n")
                    lw.writelines(open(file2).readlines())
                    lw.write("\n\n")