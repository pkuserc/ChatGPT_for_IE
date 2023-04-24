***Evaluating ChatGPT’s Information Extraction Capabilities: An Assessment of Performance, Explainability, Calibration, and Faithfulness***

[Bo Li](https://deepblue666.github.io/), Gexiang Fang, Yang Yang, Quansen Wang, [Wei Ye](https://se.pku.edu.cn/kcl/weiye/), Wen Zhao, and Shikun Zhang.


# Abstract

In this paper, we focus on assessing the overall ability of ChatGPT using 7 fine-grained information extraction (IE) tasks. Specially, we present the systematically analysis by measuring ChatGPT's performance, explainability, calibration, and faithfulness, and resulting in 15 keys from either the ChatGPT or domain experts. Our findings reveal that ChatGPT’s performance in Standard-IE setting is poor, but it surprisingly exhibits excellent performance in the OpenIE setting, as evidenced by human evaluation. In addition, our research indicates that ChatGPT provides high-quality and trustworthy explanations for its decisions. However, there is an issue of ChatGPT being overconfident in its predictions, which resulting in low calibration. Furthermore, ChatGPT demonstrates a high level of faithfulness to the original text in the majority of cases. We manually annotate and release the test sets of 7 fine-grained IE tasks contains 14 datasets to further promote the research. 

# Keys

We have collected 15 keys from the ChatGPT and domain experts. Specifically, 10 keys are extracted from ChatGPT and 5 keys involves human involvements. These keys could systemically assess ChatGPT's ability from the following four aspects: 

![keys](https://github.com/pkuserc/ChatGPT_for_IE/blob/main/Image/keys.jpg)

# Dataset
Please access the datasets used in our paper from the following resources:

#### <ins>Entity Typing(ET)<\ins>:  [BBN](https://catalog.ldc.upenn.edu/LDC2005T33), [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19)
#### <ins>Named Entity Recognition(NER)<\ins>:  [CoNLL2003](https://huggingface.co/datasets/conll2003), [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19)

# An Example

We show an input example for the event detection (ED) task to help readers understand our implementation.

|  **Input of Event Detection (ED)**  |
|  ---- |
| **<ins>Task Description:</ins>** *Given an input list of words, identify all triggers in the list, and categorize each of them into the predefined set of event types. A trigger is the main word that most clearly expresses the occurrence of an event in the predefined set of event types.* |
| **<ins>Pre-defined Label Set:<ins>** *The predefined set of event types includes: \[Life.Be-Born, Life.Marry, Life.Divorce, Life.Injure, Life.Die, Movement.Transport, Transaction.Transfer-Ownership, Transaction.Transfer-Money, Business.Start-Org, Business.Merge-Org, Business.Declare Bankruptcy, Business.End-Org, Conflict.Attack, Conflict.Demonstrate, Contact.Meet, Contact. Phone-Write, Personnel.Start-Position, Personnel.End-Position, Personnel.Nominate, Personnel. Elect, Justice.Arrest-Jail, Justice.Release-Parole, Justice.Trial-Hearing, Justice.Charge-Indict, Justice.Sue, Justice.Convict, Justice.Sentence, Justice.Fine, Justice.Execute, Justice.Extradite, Justice.Acquit, Justice.Appeal, Justice.Pardon.\]* |
| **<ins>Input and Task Requirement:<ins>** *Perform ED task for the following input list, and print the output: \[’Putin’, ’concluded’, ’his’, ’two’, ’days’, ’of’, ’talks’, ’in’, ’Saint’, ’Petersburg’, ’with’, ’Jacques’, ’Chirac’, ’of’, ’France’, ’and’, ’German’, ’Chancellor’, ’Gerhard’, ’Schroeder’, ’on’, ’Saturday’, ’still’, ’urging’, ’for’, ’a’, ’central’, ’role’, ’for’, ’the’, ’United’, ’Nations’, ’in’, ’a’, ’post’, ’-’, ’war’, ’revival’, ’of’, ’Iraq’, ’.’\] The output of ED task should be a list of dictionaries following json format. Each dictionary corresponds to the occurrence of an event in the input list and should consists of "trigger", "word_index", "event_type", "top3_event_type", "top5_event_type", "confidence", "if_context_dependent", "reason" and "if_reasonable" nine keys. The value of "word_index" key is an integer indicating the index (start from zero) of the "trigger" in the input list. The value of "confidence" key is an integer ranging from 0 to 100, indicating how confident you are that the "trigger" expresses the "event_type" event. The value of "if_context_dependent" key is either 0 (indicating the event semantic is primarily expressed by the trigger rather than contexts) or 1 (indicating the event semantic is primarily expressed by contexts rather than the trigger). The value of "reason" key is a string describing the reason why the "trigger" expresses the "event_type", and do not use any " mark in this string. The value of "if_reasonable" key is either 0 (indicating the reason given in the "reason" field is not reasonable) or 1 (indicating the reason given in the "reason" field is reasonable). Note that your answer should only contain the json string and nothing else.*  |


# Setup and Results

Please refer our paper for more details.
