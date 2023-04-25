import json

rc_semeval_path = "./SemEval_processed.json"

id2label = ['Cause Effect', 'Instrument Agency', 'Product Producer', 'Content Container', 'Entity Origin', 'Entity Destination', 'Component Whole', 'Member Collection', 'Message Topic', 'Other']
label2id = { l:i for i, l in enumerate(id2label) }

data = list()
for line in json.load(open(rc_semeval_path, "r")):

    # 1. Raw data infomation
    idx = line["idx"]
    sentence = line["sentence"]
    head_entity = line["head_entity"]
    tail_entity = line["tail_entity"]
    label = line["relation_type"]

    # 2. Generate Open Prompt
    open_pred = "Question: What is the relationship between '%s' and '%s' in the text '%s'? Answer me in json format like { \"label\": the relation } without any additional things including your notes and explanations!" % (head_entity, tail_entity, sentence)
    open_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things including your notes and explanations!"
    open_reason = "Question: Tell me the reason why do these two entities belong to this relationship?"
    open_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    open_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    # 3. Generate Close Prompt
    close_pred = "Given label set: %s\nQuestion: What is the relationship between '%s' and '%s' in the text '%s', and which category from the given label set would you use to describe this relationship? Answer me in json format like { \"label\": you choosed in the given label set } without any additional things including your notes and explanations!" % (id2label, head_entity, tail_entity, sentence)
    close_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things, including your notes and explanations!"
    close_reason = "Question: Tell me the reason why do these two entities belong to this relationship?"
    close_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    close_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    data.append({
        "info": {
            "idx": idx,
            "sentence": sentence,
            "head_entity": head_entity,
            "tail_entity": tail_entity,
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