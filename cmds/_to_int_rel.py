with open('Norm.train.txt') as fi:
    with open('Norm.train.txt2', 'w') as fo:
        for line in fi:
            spl = line.replace("nan", "0.0")
            fo.write(spl)

with open('Norm.test.txt') as fi:
    with open('Norm.test.txt2', 'w') as fo:
        for line in fi:
            spl = line.replace("nan", "0.0")
            fo.write(spl)

with open('Norm.vali.txt') as fi:
    with open('Norm.vali.txt2', 'w') as fo:
        for line in fi:
            spl = line.replace("nan", "0.0")
            fo.write(spl)
