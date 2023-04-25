import json

ner_conll_path = "./processed_conll_test.txt"

id2label = ["PER", "LOC", "ORG", "MISC"]
label2id = { l:i for i, l in enumerate(id2label) }

def get_truth(tokens, truth):
    assert len(tokens) == len(truth)

    entity, label = list(), list()
    token_type_dict = dict()
    for i in range(len(tokens)):
        if truth[i].startswith("B-"):
            entity.append(tokens[i])
            label.append(truth[i].split("-")[-1])
        elif truth[i].startswith("I-"):
            entity.append(tokens[i])
        else:
            if len(entity) != 0:
                token_type_dict.update({ " ".join(entity): label[0] })
                entity, label = list(), list()

    if len(entity) > 0:
        token_type_dict.update({ " ".join(entity): label[0] })
    
    return token_type_dict

data = list()
for line in map(lambda x: x.strip().split("\t"), open(ner_conll_path, "r").readlines()):

    # 1. Raw data infomation
    sentid = line[0]
    sentence = line[1]
    label = get_truth(sentence.split(), line[2].split())

    # 2. Generate Open Prompt
    open_pred = "Text: %s\nQuestion: Please extract the named entity from the given text. Provide the answer in the format: [{\"Entity Name\": \"Entity Label\"}] without any additional things including your explanations or notes." % (sentence)
    open_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things including your notes and explanations!"
    open_reason = "Question: Tell me the reason why does the entity belong to this type?"
    open_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    open_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    # 3. Generate Close Prompt
    close_pred = "Given label set: %s\nText: %s\nQuestion: Please extract the named entity from the given text. Based on the given label set, provide the answer in the format: [{\"Entity Name\": \"Entity Label\"}] without any additional things including your explanations or notes." % (id2label, sentence)
    close_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things, including your notes and explanations!"
    close_reason = "Question: Tell me the reason why does the entity belong to this type?"
    close_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    close_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    data.append({
        "info": {
            "sentid": sentid,
            "sentence": sentence,
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
            "close_fictitious": close_fictitious
        }
    })

with open("./prompts.json", "w") as f:
    f.write(json.dumps(data, indent=4))