all_sents, all_y = list(), list()
with open("./test.txt", "r") as f:
    sent, label = list(), list()
    for line in f.readlines():
        if "-DOCSTART-" in line:
            continue
        if line == "\n":
            all_sents.append(sent)
            all_y.append(label)
            sent, label = list(), list()
        else:
            token, _, _, y = line.split()
            sent.append(token)
            label.append(y)

formated_data = list()
for i, (sent, label) in enumerate(zip(all_sents, all_y)):
     if len(sent) > 0 and len(label) > 0:
        formated_data.append(str(i) + "\t" +" ".join(sent) + "\t" + " ".join(label))

with open("./processed_conll_test.txt", "w") as writer:
    for line in formated_data:
        writer.write(line + "\n")