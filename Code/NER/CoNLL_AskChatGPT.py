import re
import json
import time
import openai

from tqdm import tqdm

ner_conll_path = "./prompts.json"

openai.api_key = "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"

label_set = ["PER", "LOC", "ORG", "MISC"]

cnt = 0
data = list()
bar = tqdm(json.load(open(ner_conll_path, "r")))
for line in bar:
    # ------------------ #
    #        Pred
    # ------------------ #
    while True:
        try:
            # 0. ChaGPT Open Pred
            bar.set_description("0. O Pred")
            open_pred_chatgpt_ans = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": line["open"]["open_pred"]}
                ]
            )["choices"][0]["message"]["content"]

            open_pred_chatgpt_ans_processed = {
                list(item.keys())[0]: list(item.values())[0] for item in eval(
                    re.search(r'\[(.*?)\]', open_pred_chatgpt_ans).group(0)
                )
            }
            break
        except:
            time.sleep(3)
    
    while True:
        try:
            # 5. ChaGPT Close Pred
            bar.set_description("5. C Pred")
            close_pred_chatgpt_ans = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": line["close"]["close_pred"]}
                ]
            )["choices"][0]["message"]["content"]

            close_pred_chatgpt_ans_processed = {
                list(item.keys())[0]: list(item.values())[0] for item in eval(
                    re.search(r'\[(.*?)\]', close_pred_chatgpt_ans).group(0)
                )
            }
            break
        except:
            time.sleep(3)
    
    for entity, pred_close in close_pred_chatgpt_ans_processed.items():
        if entity in open_pred_chatgpt_ans_processed.keys():
            pred_open = open_pred_chatgpt_ans_processed[entity]
        else:
            pred_open = "O"
        
        if entity in line["info"]["label"].keys():
            ground_truth = line["info"]["label"][entity]
        else:
            ground_truth = "O"

        # ------------------ #
        #        Open
        # ------------------ #
        while True:
            try:
                # 1. 置信度 Conf
                bar.set_description("1. O Conf")
                open_conf_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Question: What is the type of entity '%s' in the sentence '%s'? Answer me in json format like { \"label\": the entity type } without any additional things including your explanations or notes." % (entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_open},
                        {"role": "user", "content": line["open"]["open_conf"]},
                    ]
                )["choices"][0]["message"]["content"]

                open_conf_chatgpt_ans = int(re.search("\d+", open_conf_chatgpt_ans).group())
                break
            except:
                time.sleep(3)

        while True:
            try:
                # 2. 原因 Reason
                bar.set_description("2. O Reason")
                open_reason_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Question: What is the type of entity '%s' in the sentence '%s'? Answer me in json format like { \"label\": the entity type } without any additional things including your explanations or notes." % (entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_open},
                        {"role": "user", "content": line["open"]["open_reason"]},
                    ]
                )["choices"][0]["message"]["content"]
                break
            except:
                time.sleep(3)

        while True:
            try:
                # 3. 是否合理 Reasonable
                bar.set_description("3. O Reasonable")
                open_reasonable_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Question: What is the type of entity '%s' in the sentence '%s'? Answer me in json format like { \"label\": the entity type } without any additional things including your explanations or notes." % (entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_open},
                        {"role": "user", "content": line["open"]["open_reason"]},
                        {"role": "assistant", "content": open_reason_chatgpt_ans},
                        {"role": "user", "content": line["open"]["open_reasonable"]},
                    ]
                )["choices"][0]["message"]["content"]

                if "yes" in open_reasonable_chatgpt_ans.lower():
                    open_reasonable_chatgpt_ans = 1
                    break
                elif "no" in open_reasonable_chatgpt_ans.lower():
                    open_reasonable_chatgpt_ans = 0
                    break
                else:
                    continue
            except:
                time.sleep(3)

        while True:
            try:
                # 4. 是否虚构 Fictitious
                bar.set_description("4. O Fictitious")
                open_fictitious_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Question: What is the type of entity '%s' in the sentence '%s'? Answer me in json format like { \"label\": the entity type } without any additional things including your explanations or notes." % (entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_open},
                        {"role": "user", "content": line["open"]["open_reason"]},
                        {"role": "assistant", "content": open_reason_chatgpt_ans},
                        {"role": "user", "content": line["open"]["open_fictitious"]},
                    ]
                )["choices"][0]["message"]["content"]

                if "yes" in open_fictitious_chatgpt_ans.lower():
                    open_fictitious_chatgpt_ans = 1
                    break
                elif "no" in open_fictitious_chatgpt_ans.lower():
                    open_fictitious_chatgpt_ans = 0
                    break
                else:
                    continue
            except:
                time.sleep(3)

        # ------------------ #
        #        Close
        # ------------------ #

        while True:
            try:
                # 6. 置信度 Conf
                bar.set_description("6. C Conf")
                close_conf_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Given label set: %s\nQuestion: What is the type of entity '%s' in the sentence '%s', and which category from the given label set would you use to describe this entity type? Answer me in json format like { \"label\": you choosed in the given label set } without any additional things including your notes and explanations!" % (label_set, entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_close},
                        {"role": "user", "content": line["close"]["close_conf"]},
                    ]
                )["choices"][0]["message"]["content"]

                close_conf_chatgpt_ans = int(re.search("\d+", close_conf_chatgpt_ans).group())
                break
            except:
                time.sleep(3)

        while True:
            try:
                # 7. 原因 Reason
                bar.set_description("7. C Reason")
                close_reason_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Given label set: %s\nQuestion: What is the type of entity '%s' in the sentence '%s', and which category from the given label set would you use to describe this entity type? Answer me in json format like { \"label\": you choosed in the given label set } without any additional things including your notes and explanations!" % (label_set, entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_close},
                        {"role": "user", "content": line["close"]["close_reason"]},
                    ]
                )["choices"][0]["message"]["content"]
                break
            except:
                time.sleep(3)

        while True:
            try:
                # 8. 是否合理 Reasonable
                bar.set_description("8. C Reasonable")
                close_reasonable_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Given label set: %s\nQuestion: What is the type of entity '%s' in the sentence '%s', and which category from the given label set would you use to describe this entity type? Answer me in json format like { \"label\": you choosed in the given label set } without any additional things including your notes and explanations!" % (label_set, entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_close},
                        {"role": "user", "content": line["close"]["close_reason"]},
                        {"role": "assistant", "content": close_reason_chatgpt_ans},
                        {"role": "user", "content": line["close"]["close_reasonable"]},
                    ]
                )["choices"][0]["message"]["content"]

                if "yes" in close_reasonable_chatgpt_ans.lower():
                    close_reasonable_chatgpt_ans = 1
                    break
                elif "no" in close_reasonable_chatgpt_ans.lower():
                    close_reasonable_chatgpt_ans = 0
                    break
                else:
                    continue
            except:
                time.sleep(3)

        while True:
            try:
                # 9. 是否虚构 Fictitious
                bar.set_description("9. C Fictitious")
                close_fictitious_chatgpt_ans = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Given label set: %s\nQuestion: What is the type of entity '%s' in the sentence '%s', and which category from the given label set would you use to describe this entity type? Answer me in json format like { \"label\": you choosed in the given label set } without any additional things including your notes and explanations!" % (label_set, entity, line["info"]["sentence"])},
                        {"role": "assistant", "content": "{\"label\": \"%s\"}" % pred_close},
                        {"role": "user", "content": line["close"]["close_reason"]},
                        {"role": "assistant", "content": close_reason_chatgpt_ans},
                        {"role": "user", "content": line["close"]["close_fictitious"]},
                    ]
                )["choices"][0]["message"]["content"]

                if "yes" in close_fictitious_chatgpt_ans.lower():
                    close_fictitious_chatgpt_ans = 1
                    break
                elif "no" in close_fictitious_chatgpt_ans.lower():
                    close_fictitious_chatgpt_ans = 0
                    break
                else:
                    continue
            except:
                time.sleep(3)

        answer = {
            # 1. 基本内容
            "idx": cnt,
            "sentIdx": line["info"]["sentid"],
            "sentence": line["info"]["sentence"],
            "EntityMention": entity,
            "GroundTruth": ground_truth,

            # 2. Open 场景下的回答
            "isOpenCorrect": -1,
            "OpenET": pred_open,
            "OConf": open_conf_chatgpt_ans,
            "Reason4OET": open_reason_chatgpt_ans,
            "ifR4OETAuto": open_reasonable_chatgpt_ans,
            "ifR4OETManual": -1,
            "ifR4OETFicAuto": open_fictitious_chatgpt_ans,
            "ifR4OETFicManual": -1,

            # 2. Close 场景下的回答
            "isCloseCorrect": 1 if ground_truth == pred_close else 0,
            "ClosedET": pred_close,
            "CConf": close_conf_chatgpt_ans,
            "Reason4CET": close_reason_chatgpt_ans,
            "ifR4CETAuto": close_reasonable_chatgpt_ans,
            "ifR4CETManual": -1,
            "ifR4CETFicAuto": close_fictitious_chatgpt_ans,
            "ifR4CETFicManual": -1
        }

        cnt += 1
        data.append(answer)

with open("./outputs.json", "w") as f:
    f.write(json.dumps(data, indent=4))