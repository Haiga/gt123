queries_set = set()
queries_ids = []
with open('D:\\Colecoes\\BD\\out\\temp-features\\features\\dev\\keys.csv') as f:
    for line in f:
        qid = line.split(",")[0]
        if len(queries_set) < 500:
            queries_ids.append(qid)
            queries_set.add(qid)
        elif len(queries_set) == 500 and qid in queries_set:
            queries_ids.append(qid)
        else:
            break

new_file_lines = ""
num_lines_required = len(queries_ids)
num_lines = 0
with open('D:\\Colecoes\\BD\\out\\temp-features\\features\\dev\\FWLS-all.skl') as f:
    while True:
        c = f.read(1024)
        num_lines += c.count("\n")
        new_file_lines += c
        if num_lines >= num_lines_required:
            break

with open('D:\\Colecoes\\BD\\out\\temp-features\\features\\dev\\Norm.test.txt', "w") as f:
    lines_ = new_file_lines.split("\n")
    for i in range(len(queries_ids)):
        qid = queries_ids[i]

        line2 = lines_[i]
        if line == "\n" or line2 == "\n":
            break
        spl = line2.replace("\n", "").split(" ")
        spl_f = spl[1:]

        u = spl[0] + " qid:" + qid + " "
        for feat in spl_f:
            s = feat.split(":")
            try:
                x = float(s[1])
                u += s[0] + f":{x:.4f} "
            except:
                u += s[0] + f":0 "
        result_line = u[:-1] + "\n"
        f.write(result_line)
