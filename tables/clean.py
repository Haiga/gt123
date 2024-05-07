import os
import shutil

home = r'D:\Colecoes\experimento_loss_risk\tables\overall\mlp\mq2007'
# home = r'D:\Colecoes\experimento_loss_risk\tables\overall\att\web10k\Nova pasta'
files = os.listdir(home)


def rep(file, inp, oup):
    if inp in file:
        src = home + '/' + file
        dest = home + '/' + file.replace(inp, oup)
        os.rename(src, dest)

# for file in files:
#     rep(file, 'geoRisk', 'extgeoRisk')
#     rep(file, '93', '1')
#     rep(file, '94', '2')
#     rep(file, '95', '3')
#     rep(file, '96', '4')
#     rep(file, '97', '5')

def delete(file, inp):
    if inp in file:
        # os.removedirs(home + '/' + file)
        shutil.rmtree(home + '/' + file)

# for file in files:
#     delete(file, '5')
#     delete(file, '2')
#     delete(file, '3')
#     delete(file, '4')