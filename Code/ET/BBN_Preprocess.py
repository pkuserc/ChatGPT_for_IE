import json

i = 0
data = list()
for line in map(eval, open("test.json", "r").readlines()):
    for mention in line["mentions"]:
        for one_label in mention["labels"]:
            data.append({
                "idx": i,
                "senid": line["senid"],
                "tokens": line["tokens"],
                "mentions": [{
                    "start": mention["start"], "end": mention["end"], "labels": [one_label]
                }],
                "fileid": line["fileid"]
            })
            i += 1

with open("./BBN.json", "w") as writer:
    for line in data:
        writer.write(json.dumps(data) + "\n")