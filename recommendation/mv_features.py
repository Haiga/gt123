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

                    os.rename(features_file, features_file+".old")
                    os.rename(features_names_file, features_names_file+".old")

                    os.rename(result_file, result_file.replace(".top2.skl", ".skl"))
                    os.rename(result_names_file, result_names_file.replace(".top2.features", ".features"))
                except:
                    print("Erro")
