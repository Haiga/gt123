import sys
import os

if __name__ == '__main__':

    # BD = r"C:\Users\pedro\Downloads"
    # sample = "bla"
    # name_file = "FWLS-all"
    # use_features = "cbpr-32-001,IntegratedBiasMF-020".split(",")

    # BD = sys.argv[1]
    # sample = sys.argv[2]
    # name_file = sys.argv[3]
    # use_features = sys.argv[4].split(",")

    # ['FWLS-all', 'HR-all', 'STREAM-all']
    # home = "/home/reifortes/mof/"
    # ['ML20M', 'Amazon']

    # ALS_BiasedMF - 80
    # BiasedSVD - 020
    # Amazon
    #
    # ML20M
    # IntegratedBiasMF - 020
    # ItemKNN - 025

    home = "/home/reifortes/mof/"
    for BD in ['ML20M/features', 'Amazon/features']:
        for sample in ['sample70', 'tuning70']:
            for name_file in ['FWLS-all', 'HR-all', 'STREAM-all']:
                try:
                    if 'ML20M' in BD:
                        use_features = ['IntegratedBiasMF-020', 'ItemKNN-025']
                    else:
                        use_features = ['BiasedSVD-020', 'ALS_BiasedMF-080']
                    use_features = set(use_features)

                    features_file = os.path.join(home, BD, sample, name_file) + ".skl"
                    features_names_file = os.path.join(home, BD, sample, name_file) + ".features"

                    result_file = os.path.join(home, BD, sample, name_file) + ".top2.skl"
                    result_names_file = os.path.join(home, BD, sample, name_file) + ".top2.features"

                    print(features_file)
                    print(features_names_file)

                    will_use_id_features = []
                    with open(features_file, "r") as ff, open(features_names_file, "r") as fn, open(result_file,
                                                                                                    "w") as out, open(
                            result_names_file, "w") as out_names:
                        cont = 1
                        for line in fn:
                            spl = line.strip().split(": ")
                            # print(spl)
                            id_feature = spl[0]
                            name_feature = spl[1]
                            for feat in use_features:
                                if feat in name_feature:
                                    will_use_id_features.append(id_feature)
                                    out_names.write(str(cont) + ": " + name_feature + "\n")
                                    cont += 1
                                    break
                        will_use_id_features = set(will_use_id_features)
                        for line in ff:
                            spl = line.strip().split(" ")
                            new_line = spl[0]
                            cont = 1
                            for s in spl[1:]:
                                if s.split(":")[0] in will_use_id_features:
                                    new_line += " " + str(cont) + ":" + s.split(":")[1]
                                    cont += 1
                            new_line = new_line.strip() + "\n"
                            out.write(new_line)
                except:
                    print("Erro em:" + str(features_file))