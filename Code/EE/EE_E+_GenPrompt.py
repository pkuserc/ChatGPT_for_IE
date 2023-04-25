import json
from os import path


num_list = ["zero", 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']



input_path = './data/ACE05-E+/test.oneie.json'
output_path = "./data/ACE05-E+/EE_E+_gold.json"
with open(input_path, 'r') as f:
    lines = f.readlines()

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
        for event in events:
            event['event_type'] = event['event_type'].replace(":", ".")
            event.pop('id')
            for argument in event['arguments']:
                entity_id = argument.pop("entity_id")
                matched_entity = entity_dict[entity_id]
                start = matched_entity['start']
                end = matched_entity['end']
                mention_type = matched_entity["mention_type"]
                argument['start'] = start
                argument['end'] = end
        res = {
            'id': sent_id,
            'words': words,
            'event': events,
        }
        f.write(json.dumps(res)+'\n')




# Generating Prompts for Open Setting
open_input_path = output_path
open_prompt_path = './Prompt/EE_E+_Open.json'

with open(open_input_path, 'r') as f:
    lines = f.readlines()

output={}
for line in lines:
    line = json.loads(line)
    words = line['words']
    event = line['event']

    prompt = "Event Extraction (EE) task definition:\n"
    prompt += "Given an input list of words, identify triggers in the list and categorize their event types. An event is something  that happens. An event is a specific occurrence involving entity participants, and it can frequently be described as a change of state. A trigger is the main word or phrase that most explicitly expresses the occurrence of an event. Participants of event can be pronouns. Identify participants of each event and assign a role for each participant.\n\n"
    prompt += "The output of EE task should be a list of dictionaries following \
json format. Each dictionary corresponds to a trigger and should consists of \"trigger\", \
\"start_word_index\", \"end_word_index\", \"event_type\", \"confidence\", \
\"participants\", \"reason\" and \"if_reasonable\" eight keys. The value of \"start_word_index\" key \
and \"end_word_index\" keys are the index (start from zero) of the start and end word of \"trigger\", \
respectively, in the input list. The value of \"confidence\" key is an integer ranging from 0 to 100, \
indicating how confident you are that the \"trigger\" expresses the \"event_type\" event. The value of \"reason\" key is a string describing the reason why the \
\"trigger\" expresses the \"event_type\" event, and do not use any \" mark in this string. The value \
of \"if_reasonable\" key is either 0 (indicating the reason given in the \"reason\" field is not \
reasonable) or 1 (indicating the reason given in the \"reason\" field is reasonable). The value of \
\"participants\" key is a list of dictionaries. Each dictionary corresponds to a participant of this \
\"event_type\" event expressed by word \"trigger\" and should consist of \"span\", \"start_word_index\", \
\"end_word_index\", \"role\", \"confidence\", \"reason\" and \"if_reasonable\" seven keys. The value \
of  \"role\" key is a string, indicating the role that this participant plays in this event. The value \
of \"start_word_index\" key and \"end_word_index\" keys are the index (start from zero) of the start \
and end word of \"span\", respectively, in the input list. The value of \"confidence\" key is an \
integer ranging from 0 to 100, indicating how confident you are that the \"span\" plays the \"role\" \
in this event. The value of \"reason\" key describes the reason why the \"span\" is a participant of \
the event and why it plays the \"role\" in the event. \
Note that your answer should only contain the json string and nothing else.\n\n\
Perform EE task for the following input list, and print the output:\n"
    prompt += str(words)
    output[line['id']] = prompt

with open(open_prompt_path, 'w') as f:
    json.dump(output, f, indent=4)



# Generating Prompts for Standard Setting
closed_input_path = output_path
closed_prompt_path = './Prompt/EE_E+_Closed.json'

with open(closed_input_path, 'r') as f:
    lines = f.readlines()

output={}
for line in lines:
    line = json.loads(line)
    words = line['words']
    event = line['event']
    prompt = "Event Extraction (EE) task definition:\n"
    prompt += "Given an input list of words, identify triggers in the list and categorize their event types. An event is a specific occurrence involving participants, and it can frequently be described as a change of state. A trigger is the main word or phrase that most explicitly expresses the occurrence of an event. An event's participants are the entities that are involved in that event. Participants of event can be pronouns. Identify participants of each event in the input list and assign a role for each participant. Each event type corresponds to a set of roles.\n\n"
    prompt += "All event types and their corresponding set of roles are as follows: \n\
```json\n\
{\n\
    'Business.Declare-Bankruptcy': ['Org', 'Place'],\n\
    'Business.End-Org': ['Place', 'Org'],\n\
    'Business.Merge-Org': ['Org'],\n\
    'Business.Start-Org': ['Org', 'Place', 'Agent'],\n\
    'Conflict.Attack': ['Place', 'Target', 'Attacker', 'Instrument', 'Victim'],\n\
    'Conflict.Demonstrate': ['Entity', 'Place'],\n\
    'Contact.Meet': ['Entity', 'Place'],\n\
    'Contact.Phone-Write': ['Entity', 'Place'],\n\
    'Justice.Acquit': ['Defendant', 'Adjudicator'],\n\
    'Justice.Appeal': ['Adjudicator', 'Plaintiff', 'Place'],\n\
    'Justice.Arrest-Jail': ['Person', 'Agent', 'Place'],\n\
    'Justice.Charge-Indict': ['Adjudicator', 'Defendant', 'Prosecutor', 'Place'],\n\
    'Justice.Convict': ['Defendant', 'Adjudicator', 'Place'],\n\
    'Justice.Execute': ['Place', 'Agent', 'Person'],\n\
    'Justice.Extradite': ['Origin', 'Destination', 'Agent'],\n\
    'Justice.Fine': ['Entity', 'Adjudicator', 'Place'],\n\
    'Justice.Pardon': ['Adjudicator', 'Place', 'Defendant'],\n\
    'Justice.Release-Parole': ['Entity', 'Person', 'Place'],\n\
    'Justice.Sentence': ['Defendant', 'Adjudicator', 'Place'],\n\
    'Justice.Sue': ['Plaintiff', 'Defendant', 'Adjudicator', 'Place'],\n\
    'Justice.Trial-Hearing': ['Defendant', 'Place', 'Adjudicator', 'Prosecutor'],\n\
    'Life.Be-Born': ['Place', 'Person'],\n\
    'Life.Die': ['Victim', 'Agent', 'Place', 'Instrument', 'Person'],\n\
    'Life.Divorce': ['Person', 'Place'],\n\
    'Life.Injure': ['Victim', 'Agent', 'Place', 'Instrument'],\n\
    'Life.Marry': ['Person', 'Place'],\n\
    'Movement.Transport': ['Vehicle', 'Artifact', 'Destination', 'Agent', 'Origin', 'Place'],\n\
    'Personnel.Elect': ['Person', 'Entity', 'Place'],\n\
    'Personnel.End-Position': ['Entity', 'Person', 'Place'],\n\
    'Personnel.Nominate': ['Person', 'Agent'],\n\
    'Personnel.Start-Position': ['Person', 'Entity', 'Place'],\n\
    'Transaction.Transfer-Money': ['Giver', 'Recipient', 'Beneficiary', 'Place'],\n\
    'Transaction.Transfer-Ownership': ['Buyer', 'Artifact', 'Seller', 'Place', 'Beneficiary']\n\
}\n\
```\n\n\
The output of EE task should be a list of dictionaries following json \
format. Each dictionary corresponds to a trigger and should consists of \"trigger\", \
\"start_word_index\", \"end_word_index\", \"event_type\", \"confidence\", \
\"participants\", \"reason\" and \"if_reasonable\" eight keys. \
The value of \"start_word_index\" key and \"end_word_index\" keys are the index (start \
from zero) of the start and end word of \"trigger\", respectively, in the input list. \
The value of \"confidence\" key is an integer ranging from 0 to 100, indicating how \
confident you are that the \"trigger\" expresses the \"event_type\" event. The value of \"reason\" key \
is a string describing the reason why the \"trigger\" expresses the \"event_type\" \
event, and do not use any \" mark in this string. The value of \"if_reasonable\" key is \
either 0 (indicating the reason given in the \"reason\" field is not reasonable) or 1 \
(indicating the reason given in the \"reason\" field is reasonable). The value of \
\"participants\" key is a list of dictionaries. Each dictionary corresponds to a \
participant of the \"event_type\" event expressed by word \"trigger\" and should \
consist of \"span\", \"start_word_index\", \"end_word_index\", \"role\", \"confidence\", \
\"reason\" and \"if_reasonable\" seven keys. The value of \"role\" key is a string, \
indicating the role that this participant plays in this event. The value of \
\"start_word_index\" key and \"end_word_index\" keys are the index (start from zero) of \
the start and end word of \"span\", respectively, in the input list. \
The value of \"confidence\" key is an \
integer ranging from 0 to 100, indicating how confident you are that the \"span\" plays the \"role\" \
in this event. The value of \
\"reason\" key describes the reason why the \"span\" is a participant of the event and \
why it plays the \"role\" in the event. Note that your answer should only contain the json string and nothing else.\n\n\
Perform EE task for the following input list, and print the output:\n"
    prompt+= str(words)
    output[line['id']] = prompt

with open(closed_prompt_path, 'w') as f:
    json.dump(output, f, indent=4)