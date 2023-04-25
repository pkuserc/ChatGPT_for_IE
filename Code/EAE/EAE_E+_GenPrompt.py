import json



num_list = ["zero", 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']



input_path = './data/ACE05-E+/test.oneie.json'
output_path = "./data/ACE05-E+/EAE_E+_gold.json"
with open(input_path, 'r') as f:
    lines = f.readlines()
EVENT_ID = 0
with open(output_path, 'w') as f:
    for line in lines:
        line = json.loads(line)
        sent_id = line['sent_id']
        events = line['event_mentions']
        words = line['tokens']
        entity_mentions = line['entity_mentions']
        entity_dict = {}
        for entity in entity_mentions:
            entity_dict[entity['id']] = {
                'text':entity['text'],
                'start':entity['start'],
                'end':entity['end'],
                'mention_type': entity['mention_type']
            }
        if not events:
            continue
        for event in events:
            for argument in event['arguments']:
                entity_id = argument.pop("entity_id")
                matched_entity = entity_dict[entity_id]
                start = matched_entity['start']
                end = matched_entity['end']
                mention_type = matched_entity["mention_type"]
                argument['start'] = start
                argument['end'] = end
            res = {
                'id': f'{sent_id}-{str(EVENT_ID)}',
                'words': words,
                'event': {
                    'trigger': event['trigger'],
                    'event_type': event['event_type'].replace(':', '.'),
                    'argument': event['arguments']
                },
            }
            EVENT_ID += 1
            f.write(json.dumps(res)+'\n')




# Generating Prompts for Open Setting
open_input_path = output_path
open_prompt_path = "./Prompt/EAE_E+_Open.json"

with open(open_input_path, 'r') as f:
    lines = f.readlines()

output={}
for line in lines:
    line = json.loads(line)
    words = line['words']
    event = line['event']
    trigger = event['trigger']['text']
    word_or_phra = 'word'
    if event['trigger']['end'] - event['trigger']['start'] > 1 and '-' not in trigger:
        word_or_phra = 'phrase'
    event_type = event['event_type']
    prompt = str(words) + '\n\n'
    prompt += f"The {word_or_phra} \"{trigger}\" in the above input list expresses the occurrence \
of a \"{event_type}\" event. An event's participants are the entities that are involved in that event. \
Participants of event can be pronouns. Identify participants of this event from the input list and assign a role for each \
participant.\n\n"
    prompt += f"Your answer should be a list of dictionaries following json \
format. Each dictionary corresponds to a participant of the \"{event_type}\" event expressed by word \
\"{trigger}\" and should consist of \"span\", \"start_word_index\", \"end_word_index\", \"role\", \
\"confidence\", \"reason\" and \"if_reasonable\" seven keys. \
If you believe there are no participants of the event in this input list, your answer should be an empty list.\
The value of \"role\" key is a string, \
indicating the role that this participant plays in the event. The value of \"start_word_index\" key \
and \"end_word_index\" keys are the index (start from zero) of the start and end word of \"span\", \
respectively, in the input list. The value of \"confidence\" key is an integer ranging from 0 to 100, \
indicating how confident you are that the \"span\" plays the \"role\" in the event. The value of \
\"reason\" key describes the reason why the \"span\" plays the \"role\" in the event. The value of \
\"if_reasonable\" key is either 0 (indicating the reason given in the \"reason\" field is not \
reasonable) or 1 (indicating the reason given in the \"reason\" field is reasonable). \
Note that your answer should only contain the json string and nothing else."

    output[line['id']] = prompt

with open(open_prompt_path, 'w') as f:
    json.dump(output, f, indent=4)



# Generating Prompts for Standard Setting
closed_input_path = output_path
closed_prompt_path = './Prompt/EAE_E+_Closed.json'

with open("./Code/description_queries_new.csv", "r") as f:
    lines = f.readlines()   # loading questions for roles from 《EE as QA》 paper
Event2Query = {}
for line in lines:
    line = line.strip()
    event_arg, query = line.split(",")
    event, arg = event_arg.split("_")
    if event not in Event2Query:
        Event2Query[event] = []
    Event2Query[event].append((arg, query))

with open(closed_input_path, 'r') as f:
    lines = f.readlines()

output={}
for line in lines:
    line = json.loads(line)
    words = line['words']
    event = line['event']
    trigger = event['trigger']['text']
    word_or_phra = 'word'
    if event['trigger']['end'] - event['trigger']['start'] > 1 and '-' not in trigger:
        word_or_phra = 'phrase'
    event_type = event['event_type']
    prompt = str(words) + 2 * '\n'
    prompt += f"The {word_or_phra} \"{trigger}\" in the above input list expresses the occurrence \
of a \"{event_type}\" event. An event's participants are the entities that are involved in that event. \
Participants of event can be pronouns. Identify participants of this event from the input list by answering following \
questions.\n\n"
    prompt += f"All questions related to the participants of the \"{event_type}\" event are as follows:\n"
    for i, query in enumerate(Event2Query[event_type]):
        prompt += f'- Question{str(i+1)}: In this "{event_type}" event expressed by {word_or_phra} "{trigger}", {query[1].lower()}\n'
    query_len = len(Event2Query[event_type])
    if query_len > 2:
        question = ' '.join([f'"Question{str(i+1)}",' for i,_ in enumerate(Event2Query[event_type][:-1])]) + f' and "Question{query_len}" {num_list[query_len]} keys'
    elif query_len == 2:
        question = '"Question1" and "Question2" two keys'
    else:
        question = '"Question1" one key'
    prompt += f"\nYour answer should be a dictionary following json format. The \
dictionary should consists of {question}. Each key in this dictionary corresponds a question for this \
event. Each value of these keys should be a list consisting of answers of this question. If there is \
no answer for a question, the list should be empty. Each answer should be a dictionary corresponding \
to a participant of the \"{event_type}\" event expressed by {word_or_phra} \"{trigger}\", and the dictionary \
should consist of \"span\", \"start_word_index\", \"end_word_index\", \"confidence\", \"reason\" and \
\"if_reasonable\" six keys. The value of \"start_word_index\" key and \"end_word_index\" keys are the \
index (start from zero) of the start and end word of \"span\", respectively, in the input list. The \
value of \"confidence\" key is an integer ranging from 0 to 100, indicating how confident you are that \
the \"span\" is the correct answer of this question. The value of \"reason\" key describes the reason \
why the \"span\" is the correct answer of this question. The value of \"if_reasonable\" key is either \
0 (indicating the reason given in the \"reason\" field is not reasonable) or 1 (indicating the reason \
given in the \"reason\" field is reasonable). Note that your answer should only contain the json string and nothing else."
    output[line['id']] = prompt

with open(closed_prompt_path, 'w') as f:
    json.dump(output, f, indent=4)
