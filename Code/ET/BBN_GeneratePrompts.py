import json

et_bbn_path = "../../../1.data/ET/BBN.json"

first_id2label = eval(open("../../../1.data/ET/bbn_first_id2label.txt", "r").readline())
first_label2id = { l:i for i, l in enumerate(first_id2label) }
second_id2label = eval(open("../../../1.data/ET/bbn_second_id2label.txt", "r").readline())
second_label2id = { l:i for i, l in enumerate(second_id2label) }

data = list()
for line in map(eval, open(et_bbn_path, "r").readlines()):

    # 1. Raw data infomation
    idx = line["idx"]
    sentid = line["senid"]
    sentence = " ".join(line["tokens"])
    tokens = line["tokens"]
    entity = " ".join(tokens[line["mentions"][0]["start"]:line["mentions"][0]["end"]])
    label = line["mentions"][0]["labels"][0]
    target_level = len(line["mentions"][0]["labels"][0].strip("/").split("/"))

    assert target_level in [1, 2]

    # 2. Generate Open Prompt
    open_pred = "Question: What is the type of entity '%s' in the sentence '%s'? Answer me in json format like { \"label\": the entity type } without any additional things including your explanations or notes." % (entity, sentence)
    open_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things including your notes and explanations!"
    open_reason = "Question: Tell me the reason why does the entity belong to this type?"
    open_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    open_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    # 3. Generate Close Prompt
    label_set = str(first_id2label) if target_level == 1 else str(second_id2label)

    close_pred = "Given label set: %s\nQuestion: What is the type of entity '%s' in the sentence '%s', and which category from the given label set would you use to describe this entity type? Answer me in json format like { \"label\": you choosed in the given label set } without any additional things including your notes and explanations!" % (label_set, entity, sentence)
    close_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things, including your notes and explanations!"
    close_reason = "Question: Tell me the reason why does the entity belong to this type?"
    close_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    close_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."
    close_top3_top5 = "Given label set: %s\nWhich three or five categories are the most likely for entity '%s' in sentence '%s' to belong to, based on the given set of labels? Answer me in format { \"three\": the three labels you choose in the given label set, \"five\": the five labels } without any additional things, including your notes and explanations!" % (label_set, entity, sentence)

    data.append({
        "info": {
            "idx": idx,
            "sentid": sentid,
            "sentence": sentence,
            "entity": entity,
            "label": label
        },
        "open": {
            "open_pred": open_pred,
            "open_conf": open_conf,
            "open_reason": open_reason,
            "open_reasonable": open_reasonable,
            "open_fictitious": open_fictitious
        },
        "close": {
            "close_pred": close_pred,
            "close_conf": close_conf,
            "close_reason": close_reason,
            "close_reasonable": close_reasonable,
            "close_fictitious": close_fictitious,
            "close_top3_top5": close_top3_top5
        }
    })

with open("./prompts.json", "w") as f:
    f.write(json.dumps(data, indent=4))