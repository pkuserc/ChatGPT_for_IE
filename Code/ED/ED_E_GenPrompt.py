import json
import os


input_path = "./data/ACE05-E/sentence-level_test.json"
output_path = "./data/ACE05-E/ED_E_gold.json"
with open(input_path, 'r') as f:
    lines = f.readlines()
EVENT_ID = 0
with open(output_path, 'w') as f:
    for line in lines:
        line = json.loads(line)
        doc_id = line['doc_id']
        sent_id = line['sent_id']
        events = line['events']
        words = line['words']
        res = {
            'id': f'{doc_id}-{sent_id}',
            'words': words,
            'events': events,
        }
        f.write(json.dumps(res)+'\n')


# Generating Prompts for Open Setting
open_input_path = output_path
open_prompt_path = './Prompt/ED_E_Open.json'
with open(open_input_path, 'r') as f:
    lines = f.readlines()
output={}
for line in lines:
    line = json.loads(line)
    prompt = "Event Detection (ED) task definition:\n\
Given an input list of words, identify triggers in the list and categorize their event types. \
An event is something  that happens. An event is a specific occurrence involving participants, \
and it can frequently be described as a change of state. A trigger is the main word that most \
clearly expresses the occurrence of an event.\n\n\
The output of ED task should be a list of dictionaries following json format. \
Each dictionary corresponds to a trigger and should consists of \"trigger\", \"word_index\", \
\"event_type\", \"confidence\", \"if_context_dependent\", \"reason\" and \"if_reasonable\" seven \
 keys. The value of \"word_index\" key is an integer indicating the index (start from zero) of the \
\"trigger\" in the input list. The value of \"confidence\" key is an integer ranging from 0 to 100, \
indicating how confident you are that the \"trigger\" expresses the \"event_type\" event. The value \
of \"if_context_dependent\" key is either 0 (indicating the event semantic is primarily expressed \
by the trigger rather than contexts) or 1 (indicating the event semantic is primarily expressed by \
contexts rather than the trigger). The value of \"reason\" key is a string describing the reason \
why the \"trigger\" expresses the \"event_type\" event, and do not use any \" mark in this string. \
The value of \"if_reasonable\" key is either 0 (indicating the reason given in the \"reason\" field \
is not reasonable) or 1 (indicating the reason given in the \"reason\" field is reasonable). \
Note that your answer should only contain the json string and nothing else.\n\n\
Perform ED task for the following input list, and print the output:\n"          
    prompt += str(line['words'])
    output[line['id']] = prompt

with open(open_prompt_path, 'w') as f:
    json.dump(output, f, indent=4)


# Generating Prompts for Standard Setting
closed_input_path = output_path
closed_prompt_path = './Prompt/ED_E_Closed.json'
with open(closed_input_path, 'r') as f:
    lines = f.readlines()

output={}
for line in lines:
    line = json.loads(line)
    prompt = "Event Detection (ED) task definition:\n\
Given an input list of words, identify all triggers in the list, and categorize each of \
them into the predefined set of event types. A trigger is the main word that most clearly \
expresses the occurrence of an event in the predefined set of event types.\n\n\
The predefined set of event types includes: [Life.Be-Born, Life.Marry, Life.Divorce, Life.Injure, \
Life.Die, Movement.Transport, Transaction.Transfer-Ownership, Transaction.Transfer-Money, \
Business.Start-Org, Business.Merge-Org, Business.Declare-Bankruptcy, Business.End-Org, \
Conflict.Attack, Conflict.Demonstrate, Contact.Meet, Contact.Phone-Write, Personnel.Start-Position\
, Personnel.End-Position, Personnel.Nominate, Personnel.Elect, Justice.Arrest-Jail, \
Justice.Release-Parole, Justice.Trial-Hearing, Justice.Charge-Indict, Justice.Sue, Justice.Convict\
, Justice.Sentence, Justice.Fine, Justice.Execute, Justice.Extradite, Justice.Acquit, \
Justice.Appeal, Justice.Pardon].\n\n\
The output of ED task should be a list of dictionaries following json format. \
Each dictionary corresponds to the occurrence of an event in the input list and should consists \
of \"trigger\", \"word_index\", \"event_type\", \"top3_event_type\", \"top5_event_type\", \
\"confidence\", \"if_context_dependent\",  \"reason\" and \"if_reasonable\" nine keys. The value \
of \"word_index\" key is an integer indicating the index (start from zero) of the \"trigger\" in \
the input list. The value of \"confidence\" key is an integer ranging from 0 to 100, indicating \
how confident you are that the \"trigger\" expresses the \"event_type\" event. The value of \
\"if_context_dependent\" key is either 0 (indicating the event semantic is primarily expressed by \
the trigger rather than contexts) or 1 (indicating the event semantic is primarily expressed by \
contexts rather than the trigger). The value of \"reason\" key is a string describing the reason \
why the \"trigger\" expresses the \"event_type\", and do not use any \" mark in this string. The \
value of \"if_reasonable\" key is either 0 (indicating the reason given in the \"reason\" field is \
not reasonable) or 1 (indicating the reason given in the \"reason\" field is reasonable). \
Note that your answer should only contain the json string and nothing else.\n\n\
Perform ED task for the following input list, and print the output:\n"
    prompt += str(line['words'])
    output[line['id']] = prompt

with open(closed_prompt_path, 'w') as f:
    json.dump(output, f, indent=4)