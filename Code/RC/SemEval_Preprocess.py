import os
import re
import json

from tqdm import tqdm

class SemEval2010Processor:
    def __init__(self):
        super().__init__()
        self.LABEL_TO_ID = {"Component-Whole(e2,e1)": "Component-Whole", 
                    "Other": "Other", 
                    "Instrument-Agency(e2,e1)": "Instrument-Agency", 
                    "Member-Collection(e1,e2)": "Member-Collection", 
                    "Cause-Effect(e2,e1)": "Cause-Effect", 
                    "Entity-Destination(e1,e2)": "Entity-Destination", 
                    "Content-Container(e1,e2)": "Content-Container", 
                    "Message-Topic(e1,e2)": "Message-Topic", 
                    "Product-Producer(e2,e1)": "Product-Producer", 
                    "Member-Collection(e2,e1)": "Member-Collection", 
                    "Entity-Origin(e1,e2)": "Entity-Origin", 
                    "Cause-Effect(e1,e2)": "Cause-Effect", 
                    "Component-Whole(e1,e2)": "Component-Whole", 
                    "Message-Topic(e2,e1)": "Message-Topic", 
                    "Product-Producer(e1,e2)": "Product-Producer", 
                    "Entity-Origin(e2,e1)": "Entity-Origin", 
                    "Content-Container(e2,e1)": "Content-Container", 
                    "Instrument-Agency(e1,e2)": "Instrument-Agency", 
                    "Entity-Destination(e2,e1)": "Entity-Destination"}
    def read(self, file_in):
        features = []
        data = []
        for line in open(file_in): 
            data.append(eval(line))

        for d in tqdm(data):

            tokens = d['token']
            
            head_entity = d['h']['name']
            tail_entity = d['t']['name']

            rel = self.LABEL_TO_ID[d['relation']]
    
            feature = {
                'inputs': tokens,
                'labels': rel,
                'head_entity':head_entity,
                'tail_entity':tail_entity,
            }
            
            features.append(feature)
            
        return features

test_file = os.path.join("SemEval2010_test.txt")
processor = SemEval2010Processor()

test_features = processor.read(test_file)

relation_ls = list()
for item in test_features:
    relation = item["labels"]
    relation_ls.append(relation)

relation_ls = list(map(lambda x: re.sub("-", " ", x), list(set(relation_ls))))
new_relation_ls = list()
for relation_type in relation_ls:
    new_relation_ls.append(relation_type)

i = 0
news = list()
for line in tqdm(test_features):
    new_item = {
        "idx": i, "sentence": " ".join(line["inputs"]),
        "head_entity": line["head_entity"],
        "tail_entity": line["tail_entity"],
        "relation_type": re.sub("-", " ", line["labels"])
    }
    news.append(new_item)
    i += 1

with open("./SemEval_processed.json", "w") as writer:
    writer.write(json.dumps(news, indent=4))