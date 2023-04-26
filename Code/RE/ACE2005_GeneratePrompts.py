import json

re_ace2005_path = "./ACE2005_ChatGPT_clean_all.json"

relation_type_map = { 
    "ART": "artifact", "GEN-AFF": "Gen-affiliation", "ORG-AFF": "Org-affiliation",
    "PART-WHOLE": "part-whole", "PER-SOC": "person-social", "PHYS": "physical"
}

id2label = list(relation_type_map.keys())
label2id = { l:i for i, l in enumerate(id2label) }

data = list()
for line in json.load(open(re_ace2005_path, "r")):

    # 1. Raw data infomation
    idx = line["idx"]
    sentid = line["sentid"]
    sentence = line["sentence"]
    head_entity, head_type = line["mention"]["head_entity"], line["mention"]["head_type"]
    tail_entity, tail_type = line["mention"]["tail_entity"], line["mention"]["tail_type"]
    label = line["truth"]

    # 2. Generate Open Prompt
    open_pred = "Question: What is the relationship between '%s' (Type: '%s') and '%s' (Type: '%s') in the text '%s'? Answer me in json format like { \"label\": the relation } without any additional things including your notes and explanations!" % (head_entity, head_type, tail_entity, tail_type, sentence)
    open_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things including your notes and explanations!"
    open_reason = "Question: Tell me the reason why do these two entities belong to this relationship?"
    open_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    open_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    # 3. Generate Close Prompt
    close_conf = "Question: How confident you are in making this judgment, giving it 0 to 100 percent in json format like { \"Confidence\": How confident in your mind } without any additional things, including your notes and explanations!"
    close_reason = "Question: Tell me the reason why do these two entities belong to this relationship?"
    close_reasonable = "Question: Is your reason reasonable? Just tell me yes or no."
    close_fictitious = "Question: Is your reason fictitious? Just tell me yes or no."

    data.append({
        "info": {
            "idx": idx,
            "sentid": sentid,
            "sentence": sentence,
            "head_entity": head_entity,
            "head_type": head_type,
            "tail_entity": tail_entity,
            "tail_type": tail_type,
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
            "close_pred": line["pred_close"],
            "close_conf": close_conf,
            "close_reason": close_reason,
            "close_reasonable": close_reasonable,
            "close_fictitious": close_fictitious
        }
    })

with open("./prompts.json", "w") as f:
    f.write(json.dumps(data, indent=4))