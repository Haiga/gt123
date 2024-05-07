num_lines = 0
new_file_lines = ""
with open('Norm.trainfull.txt') as f:
    while True:
        c = f.read(1024)
        num_lines += c.count("\n")
        new_file_lines += c
        if num_lines >= 71183:
            break

with open('Norm.train.txt', "w") as f:
    lines_ = new_file_lines.split("\n")
    cont = 0
    for l in lines_:
        f.write(l + "\n")
        if cont == 71090:
            break
        cont += 1
