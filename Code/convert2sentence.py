import json
import os


def convert2sentence(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_path, "w", encoding="utf8") as writer:
        for line in lines:
            doc = json.loads(line)
            doc_id = doc['doc_key']
            events = doc['events']
            sentences = doc['sentences']
            ner = doc['ner']
            _sentence_start = doc['_sentence_start']
            assert len(events) == len(sentences) == len(_sentence_start) == len(ner)

            for sent_id in range(len(sentences)):
                start = _sentence_start[sent_id]
                cur_events = events[sent_id]
                words =  sentences[sent_id]
                entities = ner[sent_id]
                if len(cur_events)==0:
                    res = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'words': words,
                        'events': [],
                        'entity': [[ent[0] - start, ent[1] - start + 1, ent[2]]for ent in entities]
                    }
                else:
                    out_events = []
                    for event in cur_events:
                        out = {"trigger":{}, "argument":[]}
                        for idx, item in enumerate(event):
                            if idx==0:
                                assert len(item)==2
                                out['trigger']['text'] = words[item[0] - start]
                                out['trigger']['word_idx'] = item[0] - start
                                out['event_type'] = item[1]
                            else:
                                assert len(item)==3
                                s = item[0] - start
                                e = item[1] - start + 1
                                out["argument"].append(
                                    {
                                        "start": s,
                                        "end": e,
                                        "text": words[s: e],
                                        "role": item[2]
                                    }
                                )
                        out_events.append(out)
                    res = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'words': words,
                        'events': out_events,
                        'entity': [[ent[0] - start, ent[1] - start + 1, ent[2]]for ent in entities]
                    }
                    
                writer.write(json.dumps(res, ensure_ascii=False) + "\n")


dataset = "test"
input_path = f'./data/ACE05-E/{dataset}.json'
output_path = f'./data/ACE05-E/sentence-level_{dataset}.json'
convert2sentence(input_path, output_path)