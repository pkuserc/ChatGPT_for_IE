import json
import random
import operator

cnt_all, cnt_right = 0, 0

cnt, data = 0, list()
for i, line in enumerate(map(eval, open("ACE2005_ChatGPT_answers_SAMPLE.json", "r"))):
    tokens = line["sentences"][0]

    ner_dict_truth = { " ".join(tokens[ner[0]: ner[1] + 1]): ner[-1] for ner in line["ner"][0] }
    ner_dict_pred  = { " ".join(tokens[ner[0]: ner[1] + 1]): ner[-1] for ner in line["predicted_ner"][0] }

    for pred_rel in line["predicted_relations"][0]:
        entity_shift = pred_rel[:-1]
        relationship = pred_rel[-1]

        head_entity = " ".join(tokens[entity_shift[0]: entity_shift[1] + 1])
        head_type = ner_dict_truth[head_entity] if head_entity in ner_dict_truth.keys() else "None"
        tail_entity = " ".join(tokens[entity_shift[2]: entity_shift[3] + 1])
        tail_type = ner_dict_truth[tail_entity] if tail_entity in ner_dict_truth.keys() else "None"

        for truth in line["relations"][0]:
            truth_entity_shift = truth[:-1]
            truth_relationship = truth[-1]

            if operator.eq(entity_shift, truth_entity_shift):
                if head_entity in ner_dict_truth.keys() and tail_entity in ner_dict_truth.keys():
                    if head_entity in ner_dict_pred.keys() and tail_entity in ner_dict_pred.keys():
                        if ner_dict_truth[head_entity] == ner_dict_pred[head_entity]:
                            if ner_dict_truth[tail_entity] == ner_dict_pred[tail_entity]:
                                if truth_relationship == relationship:
                                    break
            
            truth_relationship = "None"

        data.append({
            "idx": cnt,
            "sentid": i,
            "sentence": " ".join(tokens),
            "mention": {
                "head_entity": head_entity,
                "head_type": head_type,
                "tail_entity": tail_entity,
                "tail_type": tail_type
            },
            "truth": truth_relationship,
            "pred_close": relationship
        })
        cnt += 1

        if truth_relationship == relationship:
            cnt_right += 1

print(cnt_right / cnt)

open("./ACE2005_ChatGPT_clean_all.json", "w").write(json.dumps(data, indent=4))

# random.shuffle(data)

# cnt_sample_right = 0
# for i in data[:66]:
#     if i["pred_close"] == i["truth"]:
#         cnt_sample_right += 1

# print(cnt_sample_right)
# open("./ACE2005_ChatGPT_clean_sample.json", "w").write(json.dumps(data[:66], indent=4))
